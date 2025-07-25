#pragma once

#include "calibration.hpp"

namespace marker::symetries
{
/**
 * @brief Test type of difference of given marker of it's interior and exterior. For cores (so white dot in black ring)
 * we are testing if inner region is sufficiently brighter than outer.
 *
 *    Looks for pattern of
 *  __________             ______
 *             \          /
 *              \        /
 *               \______/
 *      ^           ^
 *      |           |
 *    centrer     minloc
 *    --------------> direction search
 */
base::Difference test_intensities_values_and_symmetries_cores(const cv::Mat1b &image, const int row_center,
                                                              const int col_center, const int width, const int height,
                                                              const float min_difference_scale);

/**
 * @brief Test type of difference of given marker of it's interior and exterior. For ring marker (so white dot in black
 * ring) we are testing if inner region is sufficiently brighter than outer OR coding marker (so black dot in white
 * area) we are testing if it sufficiently brighter that outer
 *
 *     Looks for pattern of
 *  __________             ______
 *            \          /
 *             \        /
 *               \______/
 *      ^           ^            ^
 *      |           |            |
 *    center     minloc       maxloc
 *    --------------> direction search
 *
 *   (maximum AFTER minimum)
 */
base::Difference test_intensities_values_and_symmetries_rings(const cv::Mat1b &image, const int row_center,
                                                              const int col_center, const int width, const int height,
                                                              const float min_difference_scale);
}  // namespace marker::symetries
