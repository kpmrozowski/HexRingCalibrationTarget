# Dropped-Frame-Aware Detection Repair

**Version:** 1.0 (design)
**Status:** Approved for implementation planning
**Module (new):** `marker/repair_dropped_neighbors.{hpp,cpp}`
**Callers touched:** `aslam_cameras_circlegrid_hexring/src/CircleGridCalibInterface.cpp`,
`3rd-party/HexRingCalibrationTarget/src/subcommands/calibrate_mono.cpp`
**Related doc:** `detection_circlegrid-v2.md` (unchanged by this design)

## 1. Problem

When the camera source drops frames, the dataset on disk keeps a sequential frame
index but the underlying timestamps have gaps. The detector's Hungarian tracker,
velocity field, and H2/H4 FCG pixel-to-pixel affine recovery all assume roughly
uniform per-frame dt. A gap violates that assumption and produces two failure
modes observed on `thermal_circlegrid_eposN_1/results12`:

- **F246 → F247**: dt = 534 ms (nominal 66.7 ms, ~8 frames dropped). Identified
  count collapses from 30/35 to 19/35, velocity estimate jumps from (-2.5, -0.7)
  to (-21.7, -23.7) px/frame, Hungarian drops to standalone mode.
- **F265 → F266**: F265 produces a globally-consistent-but-wrong column-slipped
  identification (28/28 inliers at 0.3 px — passes all validation) because the
  FCG reference aged past the 20-frame recency gate. F266's FCG success then
  snaps all IDs back to truth, producing an apparent v = (-84.3, -2.8) discontinuity.

Both failure modes feed implausible measurements into downstream spline fitting,
producing spline errors.

## 2. Root Cause

Dropped frames silently violate the per-frame-dt assumption. The detector has no
way to know that prev-frame is 8× older than usual, so velocity prediction,
match radii, and the FCG-recency gate all misbehave. Once detection drifts into
a column-slipped but internally consistent state, nothing recovers it until FCG
randomly succeeds again.

## 3. Approach

Two-pass orchestration around the existing per-frame detector.

1. **Pass 1** runs today's pipeline unchanged, additionally caching per-frame
   blob centroids, identifications, method, and image timestamp.
2. **Gap detection + span building** runs between passes using timestamps.
3. **Pass 2** replays the existing `detect_and_identify_circlegrid()` on each
   drop-affected span in **reverse frame order** (newer → older), seeded by the
   trailing FCG anchor. No new detection algorithm.

This is a pure orchestration change. No edits to `detection_circlegrid-v2.md`
algorithm behaviour. The repair is also gated by a feature flag so sequences
without drops behave identically to today.

## 4. Data Flow

```
Pass 1 (per-frame loop, unchanged semantically):
  for each frame f at index i:
    decoding = detect_and_identify_circlegrid(image_i, params, board, tracker, i, out)
    blob_cache[i]   = decoding.markers_              // all rings
    ident_cache[i]  = { ordering, method, identified_count, ts_ns_i }
    decoded[i]      = decoding                       // as today

Between Pass 1 and downstream consumers:
  spans = repair::detect_drop_affected_spans(ident_cache, timestamps_ns)
  for span in spans in reverse chronological order:
    repair::repair_span(span, blob_cache, ident_cache, decoded, params, board, out)

Downstream (precalibration / kalibr calibrator):
  consumes `decoded` as today, with corrected entries overwritten in place.
```

`TrackingState` from Pass 1 is discarded at the pass boundary. Each repair span
builds its own fresh `TrackingState`.

## 5. Gap Detector

Input: `timestamps_ns[]` extracted from image filenames (confirmed on disk:
`cam0_unique/<ns>.png`).

```
median_dt = median of successive diffs
gap_factor = 2.5                  // config; triggers when dt > 2.5 × median
for i in 1..N-1:
  if (ts[i] - ts[i-1]) > gap_factor * median_dt:
    emit boundary(i)              // frame i is first post-drop frame
```

For eposN_1: median_dt ≈ 66.7 ms → threshold ≈ 167 ms. F246→F247 (534 ms)
triggers; any normal dt well under threshold does not.

## 6. Affected-Span Builder

Per gap boundary `b`:

```
span.start      = b                           // first post-drop frame
span.anchor_idx = smallest f > b such that ident_cache[f].method == "findCirclesGrid"
                  AND f - b <= max_span_len (default 20, matches H2 recency gate)
span.indices    = [b .. span.anchor_idx - 1]
```

If no FCG anchor within `max_span_len`: skip the span (log as unrecoverable).
If two gap boundaries occur within one span, the span extends to the last
anchor covering them.

Option to extend `span.start` one frame backward when the pre-drop frame's
method ∈ {`hungarian`, `last_resort_homography`, `cold_start`} — off by default,
config flag `extend_backward = false`.

## 7. Per-Span Repair (reuse existing pipeline)

The repair does **not** implement any new detection logic. It calls
`detect_and_identify_circlegrid()` on each span frame in reverse, with a fresh
`TrackingState` seeded from the anchor.

### 7.1 Anchor seeding

The anchor frame's FCG output is converted into a seeded `TrackingState`:

```
TrackingState fresh;
fresh.fcg_ever_succeeded_   = true;
fresh.last_fcg_positions_   = ident_cache[anchor].marker_pixel_positions;
fresh.last_fcg_frame_idx_   = anchor;            // so has_recent_fcg stays true
fresh.tracks_               = {};                // empty, repopulated as we walk
fresh.previous_frame_idx_   = anchor;
fresh.previous_positions_   = ident_cache[anchor].marker_pixel_positions;
```

The FCG-recency gate (20 frames) is enforced against `current_idx`, so for every
frame in the span the check `(anchor - current_idx) <= 20` holds provided
`max_span_len <= 20`. We set `max_span_len = 20` to match the existing gate.

### 7.2 Reverse replay

```
for idx = span.anchor_idx - 1 down to span.start:
  repaired = detect_and_identify_circlegrid(
      image_cache_or_reload(idx), params, board, fresh, idx, out_repair);
  decoded[idx] = repaired;                             // overwrite pass-1 result
  ident_cache[idx].method = repaired.method + "_repaired";
```

Inside the call, the detector's existing H2 (FCG affine recovery) is the primary
identification source because the seeded FCG reference is always recent and
accurate. H4 fills gaps. No new code paths.

### 7.3 Blob cache reuse

To avoid re-running binarization and blob detection, the detector gains one new
optional parameter: `const std::vector<base::MarkerRing>* prebuilt_rings`. When
non-null, the detector skips the brightness-scale loop and the blob detection
stage and substitutes the prebuilt rings directly. This is a pure performance
shortcut — the identification logic downstream is identical. (If this proves
awkward to plumb, implementation may skip the cache and re-detect. User
direction accepted both interpretations; we prefer the cache for speed.)

The brightness-scale selected on pass 1 per frame is stored alongside so the
repair uses the same scale.

### 7.4 Walking forward between gap and anchor through track state

After repairing frame `idx`, the detector naturally updates `fresh.tracks_`,
`previous_positions_`, etc. to reflect frame `idx`'s positions. Since we are
walking backwards, `previous_positions_` semantically represents
"position one frame newer than current", which is the direction the velocity
field points. The detector does not depend on a physical time sign — its
Hungarian search radius and velocity field are symmetric enough for this use,
and the primary recovery path (H2) does not use velocity at all. This is the
crux of why reverse replay works without code changes.

## 8. Integration Points

### 8.1 Detector API

`detection.hpp` gains:

```
base::ImageDecoding detect_and_identify_circlegrid(
    cv::Mat1b& input, const DetectionParameters& params, const BoardCircleGrid& board,
    identification::circlegrid::TrackingState& tracker_state, int image_idx,
    const std::filesystem::path& output_path = {},
    const std::vector<base::MarkerRing>* prebuilt_rings = nullptr);   // NEW
```

`DetectionParameters` gains:

- `bool  repair_dropped_neighbors_   = true;`   // enables pass 2
- `float repair_gap_factor_          = 2.5f;`   // gap trigger multiplier
- `int   repair_max_span_len_        = 20;`     // matches H2 recency gate
- `bool  repair_extend_backward_     = false;`

### 8.2 New module

`src/marker/repair_dropped_neighbors.{hpp,cpp}` exporting:

```
namespace marker::repair {

struct FrameCacheEntry {
    uint64_t ts_ns;
    std::vector<base::MarkerRing> rings;
    std::string method;
    int identified_count;
};

struct Span {
    int start_idx;
    int anchor_idx;
    uint64_t dt_at_gap_ns;
};

std::vector<Span> detect_drop_affected_spans(
    const std::map<int, FrameCacheEntry>& cache,
    float gap_factor, int max_span_len);

void repair_span(
    const Span& span,
    const std::map<int, FrameCacheEntry>& cache,
    const std::vector<std::filesystem::path>& image_paths,
    const DetectionParameters& params,
    const BoardCircleGrid& board,
    std::map<int, base::ImageDecoding>& decoded,
    const std::filesystem::path& output_path);

}   // namespace marker::repair
```

### 8.3 Caller changes

Both callers — `CalibrateMono::execute()` and Kalibr's
`CircleGridCalibInterface` — grow a cache structure populated in the existing
loop, then invoke `detect_drop_affected_spans` + `repair_span` once after the
loop finishes. The Kalibr-side orchestration sits around the per-frame call so
the Python-exposed observation bag receives corrected entries.

### 8.4 Timestamp source

Frames are stored as `<ns>.png`. Each caller parses the filename to populate
`ts_ns`. If a non-numeric filename is encountered, the cache entry gets
`ts_ns = 0` and the gap detector skips gap checks involving that frame.

## 9. Debug Output

- Repaired frames emit `markers-png-final-*/frame_NNNNNN.png` with a visible
  annotation `[repaired]` in the header; the original pass-1 PNG is preserved
  under `markers-png-pass1/` for comparison.
- `identified-markers/frame_NNNNNN.csv` gains column `method` suffix
  `_repaired` when pass 2 overwrote the row.
- New file `repair_report.txt` in the debug root:

```
# median_dt_ms: 66.7
# gap_factor: 2.5
# spans: 2
span 0: start=247 anchor=266 dt_gap_ms=534 repaired_frames=19 recovered_markers_total=612
span 1: ...
```

## 10. Regression Monitoring

A regression sweep compares before/after results across **all nav-operations
circlegrid datasets**. The sweep is a shell script under
`scripts/regression/repair_sweep.sh` that runs `calibrate_mono` (or the existing
Kalibr subcommand used in-house) on each dataset twice — once with
`repair_dropped_neighbors_=false` (baseline) and once with it enabled — and
diffs per-frame identified counts, detection method distribution, and the
overall calibration reprojection RMSE.

Datasets covered:

- `/home/kmro/praca/dev/datasets/nav-operations/imx219_circlegrid_nord4_1/`
- `/home/kmro/praca/dev/datasets/nav-operations/imx219_circlegrid_nord4_2/`
- `/home/kmro/praca/dev/datasets/nav-operations/thermal_circlegrid_eposN_1/`
- `/home/kmro/praca/dev/datasets/nav-operations/thermal_circlegrid_eposN_2/`
- `/home/kmro/praca/dev/datasets/nav-operations/thermal_circlegrid_eposN_3/`
- `/home/kmro/praca/dev/datasets/nav-operations/thermal_circlegrid_eposN_4/`

Acceptance criteria for the sweep:

1. On eposN_1, frames 247..265 change method to `*_repaired` and F265 no longer
   shows a column-slip (visual check on `markers-png-final-*`).
2. On every dataset, number of frames with `findCirclesGrid` method is
   unchanged (we do not remove FCG successes).
3. On every dataset, total `identified-markers` summed across frames is
   greater than or equal to baseline.
4. On every dataset, calibration reprojection RMSE does not regress by more
   than 5% versus baseline (and ideally improves on eposN_1).
5. On datasets with no timestamp gaps exceeding `gap_factor × median_dt`,
   `repair_report.txt` reports `spans: 0` and `decoded` is bit-identical to
   baseline.

The sweep output is a single CSV `repair_sweep_summary.csv` with one row per
dataset summarizing the metrics above. No baselines are committed; the sweep is
re-run locally before merge.

## 11. Non-Goals

- No new identification algorithm. No new validation thresholds beyond
  `gap_factor`, `max_span_len`, `extend_backward`.
- No support for jitter compensation inside uniformly-sampled sequences.
- No changes to the per-frame detector's identification logic.
- No unit tests (per user decision 2026-04-20). Regression sweep is the sole
  automated coverage for this feature.

## 12. Open Questions

None. All design questions were resolved during brainstorming:

- Pass structure: in-process two-pass (A).
- Affected set: all non-FCG frames inside `[drop_boundary, next_FCG]` (iii).
- Pipeline reuse: call `detect_and_identify_circlegrid()` in reverse with
  seeded TrackingState; reuse pass-1 blob centroids via optional parameter.
- Gap threshold: `2.5 × median_dt`.
- Testing: no unit tests; regression sweep across all
  `nav-operations/*circlegrid*/` datasets.

## Appendix A: Evidence from eposN_1

Timestamps around F247 (from `cam0_unique/` filenames, ns):

| idx | timestamp_ns        | dt_ms  |
|-----|---------------------|--------|
| 245 | 1772704344406477568 | 66.7   |
| 246 | 1772704344473185792 | 66.7   |
| 247 | 1772704344539932160 | 66.7   |
| 248 | 1772704345073894912 | 533.9  |  ← gap
| 249 | 1772704345140627456 | 66.7   |

Note: the index alignment between `markers-png-final-003/frame_NNNNNN.png` and
`cam0_unique/<ns>.png` will be verified during implementation; the frame that
exhibits the velocity collapse may correspond to the frame labelled F247 or
F248 in debug output depending on zero-indexing.  The gap itself (≈ 534 ms,
well above the 2.5× threshold of 167 ms) is unambiguous.
