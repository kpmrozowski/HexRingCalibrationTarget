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

// Represents a gap-affected stretch of frames for which no usable FCG anchor
// is reachable within repair_max_span_len_. These frames held bad pass-1
// identifications (Hungarian / H2 after a jump) and must be invalidated
// before calibration, otherwise the wrong labels corrupt bundle adjustment.
//
// end_idx is bounded by the earliest resync point after start_idx:
//   - the next FCG success, or
//   - the frame right after the next timestamp gap,
//   - repair_max_invalidation_span_ only as a trailing-tail safety fallback
//     when neither boundary is reachable (dataset ends with no further FCG
//     or gap).
// Without this, an inter-gap stretch with no FCG in between (eposN_4
// F251..F404) would only be partially invalidated up to start + 60 and
// leave a no-man's-land of bad pass-1 IDs behind.
struct UnrecoverableSpan
{
    int      start_idx    = -1;   // first bad frame (immediately after the gap)
    int      end_idx      = -1;   // last  bad frame (inclusive)
    uint64_t dt_at_gap_ns = 0;
};

uint64_t median_dt_ns(const std::map<int, FrameCacheEntry>& frame_cache);

std::vector<Span> detect_drop_affected_spans(
    const std::map<int, FrameCacheEntry>& frame_cache,
    float gap_factor,
    int   max_span_len);

std::vector<UnrecoverableSpan> detect_unrecoverable_spans(
    const std::map<int, FrameCacheEntry>& frame_cache,
    float gap_factor,
    int   max_span_len,
    int   max_invalidation_span);

void repair_span(
    const Span& span,
    const std::map<int, FrameCacheEntry>& frame_cache,
    const DetectionParameters& base_params,
    const BoardCircleGrid& board,
    std::map<int, base::ImageDecoding>& decoded,
    const std::filesystem::path& output_path);

// Mark the frames in the span as failed in `decoded` so they drop out of
// downstream calibration. Used for gaps whose next FCG anchor is beyond
// repair_max_span_len_.
void invalidate_span(
    const UnrecoverableSpan& span,
    std::map<int, base::ImageDecoding>& decoded);

void run_repair_pass(
    const std::map<int, FrameCacheEntry>& frame_cache,
    const DetectionParameters& base_params,
    const BoardCircleGrid& board,
    std::map<int, base::ImageDecoding>& decoded,
    const std::filesystem::path& output_path);

}  // namespace marker::repair
