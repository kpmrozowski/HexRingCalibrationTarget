#pragma once

#include <opencv2/core.hpp>

#include "board.hpp"

namespace board
{

/**
 * @brief
 * We define pixel row / col AS A CENTER OF PIXEL, so pixel starts at row-0.5 and ends at row+0.5
 *
 * X-------X
 * |       |
 * |   X   |
 * |       |
 * X-------X
 *
 */
std::pair<int, bool> accurate_intensity(const float row_center, const float col_center, const float circle_row_center,
                                        const float circle_col_center, const float radius_in_pixel,
                                        const float in_value, const float out_value);
/**
 * @brief Function allow to accurately draw image of calibration board.
 */
cv::Mat1b draw_canonical_board(const BoardRectGrid &board, const float cm_per_pixel);

/**
 * @brief Function allow to accurately draw image of calibration board.
 */
cv::Mat1b draw_canonical_board(const BoardHexGrid &board, const float cm_per_pixel,
                               const cv::Size_<float> board_dimention_mm);

}  // namespace board
