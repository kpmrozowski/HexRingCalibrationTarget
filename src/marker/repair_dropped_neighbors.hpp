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

struct FrameCacheEntry
{
    uint64_t                         ts_ns            = 0;
    std::vector<base::MarkerCoding>  coding_markers;
    std::vector<cv::Point2f>         marker_positions;   // indexed by gid; (-1,-1) if unset
    std::string                      method;
    int                              identified_count = 0;
    std::filesystem::path            image_path;         // used when cached_image is empty
    cv::Mat1b                        cached_image;       // in-memory image for pipelines without a path
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
// into `decoded`. Returns the sorted list of frame indices that were cured.
// Uncured frames keep their pre-repair decoded entries; the caller is
// expected to invalidate them (they carry untrusted pass-1 Hungarian IDs).
std::vector<int> repair_span(
    const Span& span,
    const std::map<int, FrameCacheEntry>& frame_cache,
    const DetectionParameters& base_params,
    const BoardCircleGrid& board,
    std::map<int, base::ImageDecoding>& decoded,
    const std::filesystem::path& output_path);

// Mark the frames in the span as failed in `decoded` so they drop out of
// downstream calibration. Frame indices listed in `skip_cured` are left
// untouched — the caller uses this to preserve frames already cured by a
// prior `repair_span` call on an overlapping recoverable span.
void invalidate_span(
    const UnrecoverableSpan&            span,
    std::map<int, base::ImageDecoding>& decoded,
    const std::vector<int>&             skip_cured = {});

void run_repair_pass(
    const std::map<int, FrameCacheEntry>& frame_cache,
    const DetectionParameters& base_params,
    const BoardCircleGrid& board,
    std::map<int, base::ImageDecoding>& decoded,
    const std::filesystem::path& output_path);

}  // namespace marker::repair
