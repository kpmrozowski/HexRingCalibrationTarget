#include "detection_parameters.hpp"

#include <cmath>
#include <tuple>

namespace
{
std::tuple<int, int, int> get_minimal_maximal_and_edge(const float focal_in_pixel, const float dimension,
                                                       const float min_z, const float max_z)
{
    const float dimension_at_angle = dimension * std::tan(3.14 / 4.0);  // allow 45 deg observation

    const float minimal_frontal_size = dimension_at_angle / max_z;
    const float minimal_pixel_size = minimal_frontal_size * focal_in_pixel;
    const float minimal_inscribed_circle_size = 3.14 * (minimal_pixel_size / 2) * (minimal_pixel_size / 2);

    const int minimal_pixel_count = minimal_inscribed_circle_size + 1;

    const float maximal_frontal_size = dimension / min_z;
    const float maximal_pixel_size = maximal_frontal_size * focal_in_pixel;
    const float maximal_inscribed_circle_size = 3.14 * (maximal_pixel_size / 2) * (maximal_pixel_size / 2);

    const int maximal_pixel_count = maximal_inscribed_circle_size + 1;

    const int max_edge_length = maximal_pixel_size + 1;

    return {minimal_pixel_count, maximal_pixel_count, max_edge_length};
}

}  // namespace

namespace marker
{

DetectionParameters::DetectionParameters(const float focal_in_pixel, const float core_dimension,
                                         const float ring_dimension, const float min_z, const float max_z)
{
    /*
        0_________dim_
        |        /
        |       /
     Z  |      /
        |     /
        |    /
     1  |---/ u_norm
        |--/
        |-/
        |/
        0

        u_norm * focal_in pixel = pixel size
    */

    std::tie(minimal_pixel_count_core_, maximal_pixel_count_core_, maximal_edge_length_core_) =
        get_minimal_maximal_and_edge(focal_in_pixel, core_dimension, min_z, max_z);
    std::tie(minimal_pixel_count_ring_, maximal_pixel_count_ring_, maximal_edge_length_ring_) =
        get_minimal_maximal_and_edge(focal_in_pixel, ring_dimension, min_z, max_z);

    // for very small markers we set size to zero as quantization have huge infuence
    if (minimal_pixel_count_core_ < 10)
    {
        minimal_pixel_count_core_ = 0;
    }
}
}  // namespace marker
