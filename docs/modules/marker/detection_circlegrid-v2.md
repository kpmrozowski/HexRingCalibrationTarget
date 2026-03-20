# Circle Grid Detection and Identification Pipeline

**Version:** 2.0
**Module:** `marker/detection.cpp`, `identification/board_circle/identification_circle.cpp`
**Entry point:** `detection::detect_and_identify_circlegrid()`

## Changes from v1

- **H1**: Early board-to-image RANSAC after Hungarian (Section 5.6)
- **H2**: FCG pixel-to-pixel affine recovery replaces local homography when FCG reference is recent (Section 7)
- **H3**: Velocity-field cross-check for recovered markers using structural row-spacing bound (Section 7.3)
- **H4**: Post-cleanup FCG recovery with affine transform (Section 8.3)
- **180-degree flip fix**: Velocity-aware match radius + minimum-count guard (Section 4.4)
- **Brute-force gate**: Skip brute-force blob matching when FCG pixel recovery is available (Section 8.2)
- **Rotation center sign fix**: Corrected ICR computation in debug visualization
- **Affine instead of homography** for pixel-to-pixel FCG mapping (fish-eye compatibility)

## 1. Problem Statement

Detecting and identifying markers on an asymmetric circle grid calibration target from monocular camera frames. The target consists of `R x C` circles arranged in a hex-like pattern where physical positions follow:

$$x = (2c + r \bmod 2) \cdot s, \quad y = r \cdot s$$

where $s$ is the circle spacing, $r$ is the row index, and $c$ is the column index. Each marker has a unique global ID: $\text{gid} = r \cdot C + c$.

### Core Challenges

1. **Orientation ambiguity**: For grids with even row count, the grid is invariant under 180-degree rotation. A 10x5 asymmetric grid at 56 mm spacing produces a 504x504 mm board &mdash; effectively square. The homography reprojection cost for both orientations is nearly identical, making cost-based disambiguation impossible.

2. **Row-slip during fast motion**: With vertical motion of ~27 px/f and row spacing of ~33 px, after velocity compensation the inter-row gap is only ~6 px. Hungarian matching can assign markers to adjacent-row blobs. The RANSAC in Hungarian (prev-to-curr homography) cannot catch this because a general homography with 8 DOF fits both correct and row-slipped markers by varying displacement spatially.

3. **Fish-eye edge degradation**: For fish-eye cameras, the pixel-to-pixel mapping between frames is non-linear. A planar homography (8 DOF) fits well at the center where seeds cluster but extrapolates wildly at edges, causing wrong matches. An affine transform (6 DOF) extrapolates better due to fewer parameters.

4. **180-degree flip during fast motion**: The flip detection compares current gids against previous-frame gids using spatial proximity. When motion exceeds the match radius, neither orientation gets matches, and a single accidental match can trigger a false flip.

5. **Cascading swaps**: A single disappeared marker can trigger a chain: M0 disappears, M1 takes M0's spot, M6 takes M1's spot. Each link must be detected and broken.

## 2. Glossary

| Term | Meaning |
|------|---------|
| **FCG** | `findCirclesGrid` -- OpenCV's built-in function that detects a regular grid of circles and returns their centers in canonical order. When it succeeds, all markers are perfectly identified in one shot. It fails when the board is partially occluded, at extreme angles, or during motion blur. |
| **Hungarian tracking** | The Jonker-Volgenant (Hungarian) algorithm for optimal bipartite matching. Given N predicted marker positions from the previous frame and M detected blobs on the current frame, it finds the assignment that minimizes total distance. This is the primary fallback when FCG fails. |
| **Row-slip** | A misassignment where a marker is matched to the blob one grid row above or below its correct position (gid offset of +/-cols). Happens during fast vertical motion when adjacent-row blobs are close after velocity compensation. |
| **180-degree flip** | The grid has rotational symmetry -- rotating it 180 degrees maps gid X to gid (total-1-X). The pipeline must determine which orientation is correct. A flip means all markers are assigned to their 180-degree-opposite IDs. |
| **Pixel-to-pixel mapping** | A geometric transform that maps pixel coordinates from one frame directly to pixel coordinates on another frame (e.g., FCG reference frame -> current frame). Contrast with board-to-image mapping which goes from abstract board coordinates to image pixels. Pixel-to-pixel implicitly captures lens distortion since both frames are in distorted image space. |
| **Disappeared-neighbor swap** | When an edge marker goes off-screen, its hex-grid neighbor may be assigned the disappeared marker's ID by Hungarian tracking (the neighbor blob is at approximately where the disappeared marker used to be). The detection checks each identified marker against its missing neighbors' predicted positions. |
| **Last-resort homography** | A final fallback when only 4-7 markers are identified -- too few for RANSAC but enough for a least-squares homography. Projects remaining board positions and matches nearby blobs. Used only when all other methods failed to identify enough markers. |
| **H1, H2, H3, H4** | The four structural heuristics added in v2 (see Changes from v1 section). |
| **Board-to-image H** | A homography computed from known board coordinates (in mm) to detected image positions (in px). Used to validate marker assignments -- correct markers have low reprojection error (~3px), wrong markers have high error (~30px). |
| **Affine transform** | A 6-DOF linear transform (translation + rotation + scale + shear) used instead of a full 8-DOF homography for pixel-to-pixel mapping. Better edge extrapolation for fish-eye cameras because it has no perspective vanishing terms. |

## 3. Pipeline Overview

```
Input Image (grayscale)
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│ [1] BRIGHTNESS SCALE LOOP                               │
│     for β ∈ {1.0, 1.5, 0.67, 0.44, 0.3}:                │
│       • Scale: I' = clamp(β·I, 0, 255)                  │
│       • Adaptive binarization + blob detection          │
│       • Attempt findCirclesGrid (FCG)                   │
│     Select best β (most markers; prefer FCG pass)       │
│     if FCG passed -> BREAK (stop trying scales)         │
└─────────────────────────┬───────────────────────────────┘
                          │
                          ▼
              ┌───── FCG passed? ─────┐
              │ YES                   │ NO
              ▼                       ▼
┌──────────────────────┐   ┌──────────────────────────────┐
│ [2] FCG ORIENTATION  │   │ fcg_ever_succeeded_ == false?│
│  • Row-pair swap     │   │  YES -> RETURN FAILED        │
│    (cost < 0.9×orig) │   │  NO  -> continue             │
│  • 180° resolution   │   └──────────────┬───────────────┘
│    (immutable ref +  │                  │
│     track override)  │                  ▼
│  • Store FCG ref     │   ┌─────────────────────────────┐
│    positions         │   │ primary_poor?               │
│  • Set fcg_ever_     │   │ = !FCG_pass OR              │
│    succeeded_=true   │   │   identified < blobs/2      │
└──────────┬───────────┘   └──────────┬──────────────────┘
           │                          │ primary_poor AND has_previous_
           │                          ▼
           │               ┌──────────────────────────────┐
           │               │ [3] HUNGARIAN TRACKING       │
           │               │  • Cost = ‖predicted - blob‖ │
           │               │  • max_distance = 80 px      │
           │               │  • RANSAC H (prev->curr, 5px)│
           │               │  • Divergence: ratio < 0.25  │
           │               │    -> CLEAR tracker          │
           │               └──────────┬───────────────────┘
           │                          │ matched > primary?
           │                          ▼
           │               ┌─────────────────────────────┐
           │               │ [3b] H1: BOARD->IMAGE RANSAC│ <- NEW v2
           │               │  • board_coords -> image    │
           │               │  • RANSAC threshold = 8 px  │
           │               │  • Removes row-slipped      │
           │               │    markers (30px reproj)    │
           │               └──────────┬──────────────────┘
           │                          │
           │                          ▼
           │               ┌─────────────────────────────────────┐
           │               │ [4] 180° FLIP DETECTION             │  <- FIXED v2
           │               │  Guard: has_previous_ AND hungarian │
           │               │  Match radius:                      │
           │               │    r = max(50, 1.5 × ‖v_C‖)         │
           │               │  For each marker:                   │
           │               │    find nearest prev-frame within r │
           │               │    count orig vs flipped matches    │
           │               │  FLIP if:                           │
           │               │    compared ≥ 10 AND                │
           │               │    flipped > original × 3 AND       │
           │               │    flipped ≥ max(5, compared/5)     │
           │               └──────────┬──────────────────────────┘
           │                          │
           ├──────────────────────────┤  (merge FCG and tracking paths)
           │                          │
           ▼                          ▼
┌─────────────────────────────────────────────────────────┐
│ [5] BRUTE-FORCE BLOB MATCHING (cold-start fallback)     │
│  Trigger: id_count < max(8, total/3)                    │
│       AND blobs ≥ total/2                               │
│       AND NOT has_recent_fcg (≤20 frames)    <- NEW v2  │
│  if has_fcg_ref:                                        │
│    • H from FCG ref -> blobs, RANSAC 10px               │
│    • match_threshold = mean_sp × 0.25 (~33px)           │
│  else: SKIP                                             │
└──────────────────────────┬──────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│ [5b] COLD-START FALLBACK                                │
│  Trigger: id_count < max(8, total/3)                    │
│       AND blobs ≥ 3×total/4                             │
│       AND last_fcg_positions_ empty                     │
│  • Board coords -> blob centroid matching (25px init)   │
│  • RANSAC 8px, match_threshold = mean_sp × 0.2          │
└──────────────────────────┬──────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│ [5c] LAST-RESORT HOMOGRAPHY                             │
│  Trigger: 4 ≤ id_count < 8                              │
│  • Homography without RANSAC (too few points)           │
│  • match_threshold = mean_sp × 0.25                     │
└──────────────────────────┬──────────────────────────────┘
                           │
                           ▼
              ┌──── has_recent_fcg? ────┐
              │ (≤20 frames since FCG)  │
              │ YES                     │ NO
              ▼                         ▼
┌──────────────────────────┐  ┌────────────────────────────┐
│ [6] H2: FCG PIXEL-TO-    │  │ [6b] LOCAL HOMOGRAPHY      │
│   PIXEL AFFINE RECOVERY  │  │  • board->image RANSAC 5px │
│  <- NEW v2               │  │  • Global H projection     │
│  • seeds ≥ 8 required    │  │  • Local 4-neighbor fallbk │
│  • estimateAffine2D      │  │  • match = 30% spacing     │
│    RANSAC 5px            │  └─────────────┬──────────────┘
│  • match = 20% spacing   │                │
│  • H3: velocity check    │                │
│    accept if vel_err     │                │
│    < row_spacing × 0.5   │                │
│  • Iterative board->img  │                │
│    outlier removal:      │                │
│    remove worst if       │                │
│    > max(3×med, 2×P90)   │                │
│    up to 30 iterations   │                │
└──────────────┬───────────┘                │
               │                            │
               ├────────────────────────────┘  (merge)
               ▼
┌──────────────────────────────────────────────────────────┐
│ [7] FCG-REFERENCE FALLBACK                               │
│  Trigger: identified < total/2                           │
│       AND unidentified > 0                               │
│       AND has fcg_positions                              │
│       AND NOT has_recent_fcg                 <- NEW v2   │
│  • H from FCG ref -> current, RANSAC 10px                │
│  • match = mean_sp × 0.2 (20% spacing)                   │
└──────────────────────────┬───────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────┐
│ [8] DISAPPEARED-NEIGHBOR SWAP DETECTION                  │
│  Guard: has_previous_ AND method ≠ "findCirclesGrid"     │
│     AND has_180_ambiguity (even rows)                    │
│  Threshold: max(20px, 0.5 × ‖v_C‖)                       │
│  For each identified marker gid X:                       │
│    for each hex neighbor Y (6-connected):                │
│      if Y missing AND was seen within 5 frames:          │
│        predict Y position via velocity field             │
│        if ‖pos_X - predicted_Y‖ < threshold -> UNSET X   │
│  Cascade: up to 5 passes (stop when 0 rejections)        │
│  Safety: never drop below 8 identified markers           │
└──────────────────────────┬───────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────┐
│ [9] BOARD->IMAGE REPROJECTION QUALITY CHECK              │
│  • Compute H from board_coords -> image (RANSAC 5px)     │
│  • Iterative outlier removal (up to 10 passes):          │
│      threshold = max(3 × median, 2 × P90)                │
│      remove worst outlier if > threshold AND count > 8   │
│  • If mean_reproj > 20px AND count > 8:                  │
│      AGGRESSIVE RANSAC cleanup:                          │
│      keep only RANSAC inliers (threshold = 8px)          │
└──────────────────────────┬───────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────┐
│ [10] H4: POST-CLEANUP FCG RECOVERY               <- NEW v2
│  Trigger: has_fcg_ref                                    │
│       AND post_cleanup_count ≥ 8                         │
│       AND post_cleanup_count < blobs × 3/4               │
│       AND (blobs - post_cleanup_count) ≥ 4               │
│  • estimateAffine2D (RANSAC 5px) from FCG ref -> current │
│  • match = 20% spacing                                   │
│  • H3: velocity check (vel_err < row_spacing × 0.5)      │
│  • Validate vs clean board->image H:                     │
│      reject if reproj > max(5 × clean_median, 15px)      │
└──────────────────────────┬───────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────┐
│ [11] TRACKER STATE UPDATE                                │
│  • Update per-marker tracks (3-frame ring buffer)        │
│  • Fit forward/backward velocity fields (RANSAC)         │
│  • Prune tracks unseen for >10 frames                    │
│  • Store blob positions for next frame                   │
└──────────────────────────┬───────────────────────────────┘
                           │
                           ▼
                   Output: identified markers,
                   ordering matrix, debug CSV/PNG
```

## 4. Block Descriptions

### [1] Brightness Scale Loop

The calibration target may be unevenly lit (shadows, reflections). Testing multiple brightness scalings of the image ensures blob detection works even in poor lighting. The loop tries progressively darker/brighter versions until FCG succeeds or the best blob count is found.

### [2] FCG Orientation Resolution

FCG succeeded and returned all marker centers in grid order. But the order may have two ambiguities: (a) row-pairs may be swapped (OpenCV quirk with asymmetric grids), and (b) the entire grid may be 180-degree rotated. This block resolves both using an immutable reference locked on the first FCG frame, and stores the corrected positions for use by future frames.

If FCG has never succeeded on any frame yet, there is no orientation reference and no tracking history, so identification is impossible -- return failure. Otherwise, check if the result is poor enough to warrant tracking.

### [3] Hungarian Tracking

When FCG fails, we use frame-to-frame tracking: predict where each marker from the previous frame should appear on the current frame (using velocity history), then find the globally optimal assignment of predictions to detected blobs via the Jonker-Volgenant algorithm. A RANSAC homography fitted from (previous position, current position) pairs validates the matches by checking that the overall spatial transform is self-consistent. If fewer than 25% of blobs are matched, the tracker is considered diverged and its state is cleared.

### [3b] H1: Early Board-to-Image RANSAC (NEW in v2)

Hungarian matching can produce "row-slips" where markers are assigned to blobs one grid row off. These pass the prev-to-curr RANSAC because a general homography (8 DOF) can model spatially-varying displacement. But they fail a board-to-image check: if gid 25 is actually at gid 30's image position, projecting board coordinates for gid 25 gives ~30px error vs ~3px for correct markers. This immediate validation catches row-slips before they contaminate downstream recovery steps. The signal-to-noise ratio is 10x, making the threshold non-critical.

### [4] 180-Degree Flip Detection (FIXED in v2)

After Hungarian assigns gids, the entire assignment may be 180-degree flipped (all gids are total-1-gid). This block detects flips by comparing each marker's assigned gid against the nearest previous-frame marker within a velocity-scaled search radius. If most matches are flipped, all gids are corrected. The v2 fix adds two guards: (a) the match radius scales with motion speed so fast-moving boards still find matches, and (b) a minimum count of flipped matches is required to prevent single-match noise from triggering a false flip.

### [5] Brute-Force Blob Matching

When very few markers are identified (< total/3), this fallback tries to match all detected blobs against the stored FCG reference positions using a homography with a loose 33px threshold. In v2, this is skipped when a recent FCG reference exists because block [6] H2 does the same job with tighter thresholds and velocity validation.

### [5b] Cold-Start Fallback

When no FCG reference exists yet (the very beginning of the sequence, before any FCG success), this bootstraps identification from scratch by matching board coordinates directly to blob positions using centroid alignment and scale estimation.

### [5c] Last-Resort Homography

When only 4-7 markers are identified -- too few for RANSAC but enough for a least-squares homography -- this projects all remaining board positions and matches the nearest unmatched blobs. It is the final attempt before giving up on unidentified blobs.

### [6] H2: FCG Pixel-to-Pixel Affine Recovery (NEW in v2)

When a recent FCG reference exists (within 20 frames), map its stored pixel positions to the current frame using an affine transform (6 DOF) fitted from the already-identified markers as seed points. Then project all FCG positions through this affine and match each to the nearest unidentified blob. The affine is used instead of a homography (8 DOF) because it extrapolates better at image edges under fish-eye distortion -- it has no perspective vanishing terms that can go wild at corners. Each recovered marker is cross-checked against the velocity field (H3: accept only if prediction error < half the image-space row spacing) and then validated against a board-to-image homography via iterative outlier removal.

### [6b] Local Homography (fallback)

When no recent FCG reference is available, use the standard board-coordinate-to-image homography to project missing board positions and match unidentified blobs. Less accurate than pixel-to-pixel because board-to-image does not capture lens distortion.

### [7] FCG-Reference Fallback

A legacy recovery path for when less than half the markers are identified. Uses the stored FCG reference with a homography projection and 20% spacing threshold. In v2, this is gated: skipped when a recent FCG reference exists (H2 already handled it more accurately).

### [8] Disappeared-Neighbor Swap Detection

When an edge marker goes off-screen, its hex-grid neighbor (6-connected on the asymmetric grid) may inherit the disappeared marker's ID during Hungarian tracking -- the neighbor blob sits near the disappeared marker's predicted position. This block checks each identified marker against its missing neighbors: if marker X sits at the motion-compensated predicted position of its missing neighbor Y, then X has the wrong ID and is unset. Multiple cascade passes (up to 5) catch chain reactions where A takes B's ID, then C takes A's ID, and so on.

### [9] Board-to-Image Reprojection Quality Check

A global quality gate that validates ALL identified markers against a board-to-image homography. First, iteratively removes the single worst outlier if it exceeds a data-adaptive threshold (max of 3x median and 2x P90 reprojection error). If the overall mean reprojection error is still above 20px after iterative removal, an aggressive RANSAC cleanup keeps only tight inliers (8px threshold). This catches any remaining misidentifications from all upstream steps.

### [10] H4: Post-Cleanup FCG Recovery (NEW in v2)

After the quality check removed wrong markers, there may be gaps -- detected blobs that lost their IDs. This second FCG recovery pass uses the clean surviving markers as seeds for a fresh affine transform, then fills the gaps. Because the seeds are now guaranteed correct (they survived the quality check), the affine is more accurate than the first pass. Each recovered marker gets the same velocity cross-check (H3) and board-to-image reproj validation.

### [11] Tracker State Update

Save the current frame's identified markers into the tracking state for use by the next frame. This includes updating per-marker position histories (3-frame ring buffer for velocity estimation), fitting the global similarity velocity field via RANSAC (forward and backward), and pruning tracks not seen for more than 10 frames.

## 5. Brightness Scale Loop

*Unchanged from v1.*

## 6. findCirclesGrid and Row-Pair Swap

### 6.1-6.3 *Unchanged from v1 (Sections 4.1-4.3).*

### 6.4 180-Degree Flip Detection (post-Hungarian)

**Problem in v1**: The flip detection compared current gids against previous-frame gids using a fixed 50 px spatial radius. At high motion speeds (e.g., 63.7 px/f), no markers from the previous frame fall within 50 px, producing `matches_original=0`. A single accidental match in the flipped orientation (`matches_flipped=1`) triggered the condition `1 > 0*3 = TRUE`, incorrectly flipping all gids.

**Fix (v2)**: Two-part structural fix.

**Part A &mdash; Velocity-aware match radius:**

$$r_{\text{match}} = \max(50,\; 1.5 \cdot \|\mathbf{v}_C\|)$$

At $v=63.7$ px/f, the radius becomes ~95 px, ensuring correct-orientation matches are found.

**Part B &mdash; Minimum flipped-count guard:**

```
if (compared >= 10
    && matches_flipped > matches_original * 3
    && matches_flipped >= max(5, compared / 5))
```

Requires at least 5 flipped matches (or 20% of compared), preventing single-match noise from triggering a flip.

## 7. Hungarian Tracking

### 7.1-7.4 *Unchanged from v1 (Sections 5.1-5.4).*

### 7.5 Bidirectional Velocity Acceptance Check

*Unchanged from v1 (Section 5.5).*

### 7.6 H1: Early Board-to-Image RANSAC Validation [NEW]

**Problem**: Hungarian+RANSAC (prev-to-curr homography) passes row-slipped markers because the homography model can fit spatially-varying displacement. The board-to-image reproj check later catches them, but only after they've been used as seeds for downstream recovery steps.

**Solution**: Immediately after Hungarian assigns gids, compute a board-to-image RANSAC homography. Row-slipped markers have ~30 px reproj against board-to-image H (they're at gid+5's position but labeled as gid), while correct markers have <5 px. RANSAC with threshold 8 px trivially separates them.

**Why structural**: The signal-to-noise ratio is 10× (3 px correct vs 30 px row-slip). Any reasonable threshold between 5 and 25 px catches all row-slips. This is independent of motion speed, grid size, or camera model.

**Impact**: Catches 10-22 row-slipped markers per frame on fast-motion sequences.

## 8. Velocity Field: 2D Similarity Model

*Unchanged from v1 (Section 6).*

## 9. H2: FCG Pixel-to-Pixel Affine Recovery [NEW]

### 9.1 Motivation

When findCirclesGrid fails but a recent FCG reference exists (≤20 frames old), the FCG reference pixel positions can be mapped to the current frame to recover unidentified markers. This replaces the local homography re-identification (Section 7 of v1) when applicable, because:

- **FCG pixel-to-pixel mapping** implicitly includes lens distortion (both reference and current positions are in distorted image space)
- **Local homography** uses board coordinates -> image, which has systematic error from fish-eye distortion at edges

### 9.2 Affine Instead of Homography

The FCG-to-current mapping uses `cv::estimateAffine2D` (6 DOF) instead of `cv::findHomography` (8 DOF).

**Why affine is better for fish-eye pixel-to-pixel mapping:**

| Property | Homography (8 DOF) | Affine (6 DOF) |
|----------|--------------------:|----------------:|
| Parameters | 8 | 6 |
| Edge extrapolation | Unstable (perspective vanishing) | Stable (linear) |
| Fish-eye compatibility | Poor (assumes pinhole) | Better (no perspective term) |
| Rotation handling | Yes | Yes |
| Scale handling | Yes | Yes |

For nearby frames (1-10 frame gap), the board undergoes translation + rotation + scale change. The perspective change is negligible. Affine captures these motions with fewer parameters, giving better-constrained extrapolation at edges.

### 9.3 H3: Velocity-Field Cross-Check

Each FCG-recovered marker is cross-checked against the velocity field prediction:

1. Look up the marker's track in `tracker_state.tracks_`
2. If a track exists, compute `transport_predict(last_position)` -> predicted position
3. Compare predicted position to the blob's actual position
4. **Accept** if distance < half the image-space row spacing
5. **Reject** if distance exceeds this bound
6. If no track exists for this gid, accept unconditionally

**Structural row-spacing bound**: The image-space row spacing is computed from identified markers in adjacent rows. For a correctly assigned marker, velocity prediction error is ~5-10 px. For a row-slipped marker, the error is ~30-35 px (one full row off). Half the row spacing (~15-17 px) cleanly separates the two cases. This threshold adapts automatically to zoom level and perspective.

### 9.4 Iterative Board-to-Image Outlier Removal

After affine recovery adds markers, an iterative outlier removal pass checks each marker against a board-to-image RANSAC homography. The worst outlier is removed if it exceeds max(3×median, 2×P90) reprojection error. This converges naturally and adapts to the actual reproj distribution, unlike a fixed RANSAC threshold which can over-reject correct edge markers with fish-eye distortion.

### 9.5 Fallback to Local Homography

When no recent FCG reference exists (>20 frames since last FCG success, or no FCG success yet), the pipeline falls back to the v1 local homography re-identification.

## 10. Post-Identification Filters

### 10.1 Disappeared-Neighbor Swap Detection

*Unchanged from v1 (Section 10).*

### 10.2 Brute-Force Blob Matching Gate [NEW]

The brute-force blob matching step (triggered when identified count < total/3) uses a 33 px threshold &mdash; much looser than the FCG pixel recovery's 10 px. When a recent FCG reference exists, brute-force is skipped entirely because H2 handles this case more accurately.

### 10.3 H4: Post-Cleanup FCG Recovery [NEW]

After the board-to-image reproj check removes wrong markers (from any source), a second FCG recovery pass fills the gaps using clean survivors as seeds:

1. Compute affine transform from FCG reference -> current clean positions
2. Project all FCG positions -> match unidentified blobs (20% spacing threshold)
3. Velocity cross-check each match (H3, same as Section 9.3)
4. Board-to-image reproj validation against clean-only homography

This two-pass approach ensures that the first pass (H2) adds markers aggressively, the reproj check removes wrong ones, and the second pass (H4) fills remaining gaps using only validated seeds.

## 11-12. Remaining Sections

*Unchanged from v1 (Tracking State, Debug Output -- Sections 11-12 of v1).*

## Appendix A: Key Thresholds

| Threshold | Value | Rationale |
|-----------|-------|-----------|
| H1 board-to-image RANSAC | 8 px | 10x margin: 3 px correct vs 30 px row-slip |
| H2 FCG affine matching | 20% of mean spacing (~10 px) | Tighter than local homography (30%) because FCG mapping is more accurate |
| H3 velocity cross-check | half image row spacing | Structural: 5-10 px correct vs 30 px row-slip; adapts to zoom |
| H2 board-to-image outlier | iterative 3x median / 2x P90 | Adapts to actual reproj distribution; no fixed threshold |
| H4 reproj validation | max(5×clean_median, 15 px) | Generous for FCG reference from distant frames |
| 180° flip match radius | max(50 px, 1.5× speed) | Velocity-scaled to handle fast motion |
| 180° flip min count | max(5, compared/5) | Prevents single-match noise from triggering flip |
| Brute-force gate | ≤20 frames since FCG | Same recency as H2 |
| FCG reference recency | 20 frames | Balance between accuracy and coverage |
| *All v1 thresholds* | *Unchanged* | *See v1 Appendix A* |

## Appendix B: Design Decisions

### B.1-B.4 *Unchanged from v1.*

### B.5 Affine vs Homography for Pixel-to-Pixel Mapping

**Decision**: Use affine (6 DOF) for FCG pixel-to-pixel recovery instead of homography (8 DOF).

**Rationale**: The pixel-to-pixel mapping between fish-eye frames is non-projective. A homography fits the center well (where seeds cluster) but extrapolates wildly at corners due to the perspective terms that have no physical basis in this context. An affine transform has no perspective terms, giving stable linear extrapolation. For nearby frames (≤20 frame gap), the board motion is well-approximated by translation + rotation + scale, which affine captures exactly.

**Evidence**: On n2c F668, homography produced 34/38 wrong markers at edges. Affine produced 47/50 correct markers on the same frame.

### B.6 Early Board-to-Image Validation (H1)

**Decision**: Validate Hungarian matches against board-to-image immediately, before any recovery steps.

**Rationale**: Row-slipped Hungarian matches, if left in place, become seeds for local homography and FCG recovery. These downstream steps then add more wrong markers at the same wrong positions. Catching row-slips early breaks this cascade. The board-to-image RANSAC is the right tool because both correct and row-slipped markers map to different board coordinates at the same image position -- the RANSAC consensus always favors the correct gids.

### B.7 Two-Pass FCG Recovery (H2 + H4)

**Decision**: Run FCG recovery twice &mdash; before and after the reproj quality check.

**Rationale**: The first pass (H2) adds markers aggressively using the FCG affine. Some may be wrong at edges. The reproj check removes these. The second pass (H4) uses only the clean survivors as seeds for a more accurate affine, recovering additional markers that the first pass missed or got wrong. This "add -> validate -> re-add" pattern converges to a better solution than a single pass with conservative thresholds.

### B.8 Velocity-Aware 180° Flip Detection

**Decision**: Scale the spatial match radius with the velocity field magnitude, and require a minimum count of flipped matches.

**Rationale**: The v1 fixed 50 px radius failed at high motion speeds (>50 px/f) where no markers from the previous frame were within range. With zero matches in both orientations, a single accidental match triggered the flip. The velocity-aware radius ensures matches are found at any speed. The minimum-count guard is a safety net: even with perfect radius, noise can produce a few spurious matches. Requiring ≥5 (or 20% of compared) ensures the flip decision is statistically significant.

**Evidence**: On n2c F223 (v=63.7 px/f), v1 incorrectly flipped 30 markers. v2 correctly identifies all 47 markers without flipping.
