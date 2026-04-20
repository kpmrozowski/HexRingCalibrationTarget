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

uint64_t median_dt_ns(const std::map<int, FrameCacheEntry>& frame_cache);

std::vector<Span> detect_drop_affected_spans(
    const std::map<int, FrameCacheEntry>& frame_cache,
    float gap_factor,
    int   max_span_len);

void repair_span(
    const Span& span,
    const std::map<int, FrameCacheEntry>& frame_cache,
    const DetectionParameters& base_params,
    const BoardCircleGrid& board,
    std::map<int, base::ImageDecoding>& decoded,
    const std::filesystem::path& output_path);

void run_repair_pass(
    const std::map<int, FrameCacheEntry>& frame_cache,
    const DetectionParameters& base_params,
    const BoardCircleGrid& board,
    std::map<int, base::ImageDecoding>& decoded,
    const std::filesystem::path& output_path);

}  // namespace marker::repair
