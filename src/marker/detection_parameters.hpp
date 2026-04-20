#pragma once

#include <array>

namespace marker
{
struct DetectionParameters
{
    const int row_tiles_count_ = 4 * 2;
    const int col_tiles_count_ = 7 * 2;

    const int minimal_threshold_ = 40;

    int minimal_pixel_count_core_ = 0;
    int maximal_edge_length_core_ = 12;
    int maximal_pixel_count_core_ = maximal_edge_length_core_ * maximal_edge_length_core_;

    int minimal_pixel_count_ring_ = maximal_edge_length_core_ * maximal_edge_length_core_;
    int maximal_edge_length_ring_ = maximal_edge_length_core_ * 3;
    int maximal_pixel_count_ring_ = maximal_edge_length_ring_ * maximal_edge_length_ring_;

    float reduction_in_edge_length_ring_ = 0.8f;

    float interia_ratio_ = 0.4f;

    float min_difference_scale_ = 1.5f;  // at least 20% brighterr

    float edge_average_difference_allowed = 0.5f;

    std::array<float, 5> brightness_scales_{1.0f, 1.5f, 0.67f, 0.44f, 0.30};

    // Dropped-frame repair (see docs/modules/marker/2026-04-20-dropped-frame-repair-design.md)
    bool  repair_dropped_neighbors_ = true;
    float repair_gap_factor_        = 2.5f;
    int   repair_max_span_len_      = 20;
    bool  repair_extend_backward_   = false;

    DetectionParameters(const float focal_in_pixel, const float core_dimension, const float ring_dimension,
                        const float min_z, const float max_z);
};
}  // namespace marker