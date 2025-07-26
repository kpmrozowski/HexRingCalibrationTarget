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

    float min_difference_scale_ = 1.2f;  // at least 20% brighterr

    float edge_average_difference_allowed = 0.5f;

    std::array<float, 9> brightness_scales_{1.0f, 1.2f, 0.83f, 1.5f, 0.67f, 1.8f, 0.56f, 2.2f, 0.46f};

    DetectionParameters(const float focal_in_pixel, const float core_dimension, const float ring_dimension,
                        const float min_z, const float max_z);
};
}  // namespace marker