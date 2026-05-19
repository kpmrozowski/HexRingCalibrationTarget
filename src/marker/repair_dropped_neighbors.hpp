#pragma once

#include <cstdint>
#include <filesystem>
#include <map>
#include <string>
#include <vector>

#include "calibration.hpp"
#include "board.hpp"
#include "detection_parameters.hpp"
#include "identification/board_circle/identification_circle.hpp"

namespace marker::repair
{

// Why a given frame was dropped from calibration during the repair pass.
//   kGap     — the span had no reachable FCG before the next jump / EOF, so
//              the frame's pass-1 Hungarian IDs are untrusted and we bail
//              out without ever attempting an affine cure.
//   kPhantom — repair fit an affine and matched blobs, but the final
//              homography check found a "virtual-row" alias (see
//              has_virtual_row_alias in the .cpp), so we invalidate the
//              frame rather than emit wrong IDs into calibration.
enum class InvalidationReason
{
    kGap,
    kPhantom,
};

// Outcome of back-propagation across one recoverable span.
//   cured             — frames that were successfully relabelled
//                       via the per-span affine chain.
//   phantom_rejected  — frames where repair produced IDs but the virtual-row
//                       homography check flagged them as row/column aliased
//                       and therefore must be invalidated under kPhantom.
// Frames neither cured nor phantom_rejected are antialias-rejected and get
// invalidated by the caller's second sweep under reason kGap.
struct SpanRepairResult
{
    std::vector<int> cured;
    std::vector<int> phantom_rejected;
};

struct FrameCacheEntry
{
    uint64_t                         ts_ns            = 0;
    std::vector<base::MarkerCoding>  coding_markers;
    std::vector<cv::Point2f>         marker_positions;   // indexed by gid; (-1,-1) if unset
    std::string                      method;
    int                              identified_count = 0;
    std::filesystem::path            image_path;         // used when cached_image* are empty
    cv::Mat1b                        cached_image;       // raw in-memory image (legacy callers)
    std::vector<uchar>               cached_image_png;   // PNG-encoded image (preferred — ~5x less RAM than raw cv::Mat1b)
    bool                             fcg_succeeded    = false;
};

struct Span
{
    int      start_idx    = -1;
    int      anchor_idx   = -1;
    uint64_t dt_at_gap_ns = 0;
};

// Represents a gap-affected stretch of frames for which no FCG re-anchor
// exists before the next timestamp jump (or before end-of-sequence). These
// frames held bad pass-1 identifications (Hungarian / H2 after a jump) and
// must be invalidated before calibration, otherwise the wrong labels corrupt
// bundle adjustment.
//
// end_idx is bounded by the next resync point after start_idx:
//   - the frame right before the next timestamp gap, or
//   - the last frame in the cache when no further gap exists before EOF.
struct UnrecoverableSpan
{
    int      start_idx    = -1;   // first bad frame (immediately after the gap)
    int      end_idx      = -1;   // last  bad frame (inclusive)
    uint64_t dt_at_gap_ns = 0;
};

uint64_t median_dt_ns(const std::map<int, FrameCacheEntry>& frame_cache);

// Decode the per-frame backup image stored in a FrameCacheEntry. Returns
// the PNG-decoded buffer when cached_image_png is populated (the low-RAM
// path used by CircleGridCalibInterface since VN-3980); otherwise falls
// back to the raw cv::Mat1b cached_image; otherwise loads from image_path;
// otherwise an empty Mat. Used by callers that need pixel data outside of
// load_frame_image()'s file scope (e.g. debug rendering).
cv::Mat1b load_frame_image(const FrameCacheEntry& entry);

// Spans whose next-boundary scan (unbounded) reaches a FCG success before the
// next timestamp jump. The FCG becomes the back-propagation anchor for the
// whole stretch; the anti-alias guard inside repair_span() decides which
// frames in the span get cured and which remain bad.
std::vector<Span> detect_drop_affected_spans(
    const std::map<int, FrameCacheEntry>& frame_cache,
    float gap_factor);

// Spans with no FCG success before the next timestamp jump (or end-of-sequence).
// No anchor is reachable without crossing a second jump, so every frame in
// the span is invalidated outright.
std::vector<UnrecoverableSpan> detect_unrecoverable_spans(
    const std::map<int, FrameCacheEntry>& frame_cache,
    float gap_factor);

// Back-propagate IDs from `span.anchor_idx` (an FCG) through frames
// `[anchor_idx-1 .. start_idx]`, writing repaired `ImageDecoding` entries
// into `decoded`. Returns a `SpanRepairResult` with two sorted index lists:
//   - `cured`            — frames successfully relabelled by the affine chain
//   - `phantom_rejected` — frames where repair matched enough blobs but the
//                          virtual-row homography check flagged the fit as
//                          row/column aliased; the caller must invalidate
//                          them under `InvalidationReason::kPhantom`.
// Frames that appear in neither list are antialias-rejected; the caller's
// gap sweep invalidates them under `InvalidationReason::kGap`.
SpanRepairResult repair_span(
    const Span& span,
    const std::map<int, FrameCacheEntry>& frame_cache,
    const DetectionParameters& base_params,
    const BoardCircleGrid& board,
    std::map<int, base::ImageDecoding>& decoded,
    const std::filesystem::path& output_path);

// Mark the frames in the span as failed in `decoded` so they drop out of
// downstream calibration. The `reason` is reported verbatim in the log line
// so operators can tell gap-driven invalidations from phantom-driven ones.
// Frame indices listed in `skip_cured` are left untouched — the caller uses
// this to preserve frames already cured (or already phantom-invalidated)
// by a prior pass on an overlapping span.
void invalidate_span(
    const UnrecoverableSpan&            span,
    std::map<int, base::ImageDecoding>& decoded,
    InvalidationReason                  reason,
    const std::vector<int>&             skip_cured = {});

void run_repair_pass(
    const std::map<int, FrameCacheEntry>& frame_cache,
    const DetectionParameters& base_params,
    const BoardCircleGrid& board,
    std::map<int, base::ImageDecoding>& decoded,
    const std::filesystem::path& output_path);

}  // namespace marker::repair
