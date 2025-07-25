#include "board_drawing.hpp"

#include <opencv2/imgproc.hpp>

#include <spdlog/spdlog.h>

namespace
{
std::pair<float, float> on_image_location(const int marker_id, const std::vector<Eigen::Vector3d> &marker_centers,
                                          const float cm_per_pixel)
{
    const Eigen::Vector3d location = marker_centers[marker_id];

    const float row_center = location(1) / cm_per_pixel;
    const float col_center = location(0) / cm_per_pixel;

    return {row_center, col_center};
}

std::pair<float, float> on_image_location(const BoardRectGrid &board, const int row, const int col,
                                          const float cm_per_pixel)
{
    return on_image_location(board.row_and_col_to_id(row, col), board.marker_centers_, cm_per_pixel);
}

std::pair<float, float> on_image_location(const BoardHexGrid &board, const int row, const int col,
                                          const float cm_per_pixel)
{
    return on_image_location(board.row_and_col_to_id(row, col), board.marker_centers_, cm_per_pixel);
}

float percentage_outside(const float row_center, const float col_center, const float circle_row_center,
                         const float circle_col_center, const float radius_in_pixel_sq)
{
    // it require a lot of consideration, for now just sample in grid
    constexpr int kSplits = 10;
    constexpr float kStep = 1.0 / kSplits;
    constexpr float kOffset = kSplits / 2 * kStep;

    int outside = 0;
    for (int step_row = 0; step_row <= kSplits; ++step_row)
    {
        const float row_coord = row_center + step_row * kStep - kOffset;
        const float row_distance = (row_coord - circle_row_center) * (row_coord - circle_row_center);
        for (int step_col = 0; step_col <= kSplits; ++step_col)
        {
            const float col_coord = col_center + step_col * kStep - kOffset;
            const float col_distance = (col_coord - circle_col_center) * (col_coord - circle_col_center);

            outside += (row_distance + col_distance) < radius_in_pixel_sq;
        }
    }

    return float(outside) / ((kSplits + 1) * (kSplits + 1));
}

void draw_circle(cv::Mat1b &image, const float circle_row_center, const float circle_col_center,
                 const float radius_in_pixel, const float in_value, const float out_value)
{
    const int start_row = std::clamp(int(circle_row_center) - int(radius_in_pixel) - 2, 0, image.rows);
    const int end_row = std::clamp(int(circle_row_center) + 1 + int(radius_in_pixel) + 3, 0, image.rows);

    const int start_col = std::clamp(int(circle_col_center) - int(radius_in_pixel) - 2, 0, image.cols);
    const int end_col = std::clamp(int(circle_col_center) + 1 + int(radius_in_pixel) + 3, 0, image.cols);

    for (int row = start_row; row < end_row; ++row)
    {
        for (int col = start_col; col < end_col; ++col)
        {
            // NOTE THAT WE DO NOT ALLOW ALIASING OF CIRCLES!!!
            const auto [val, inside] = board::accurate_intensity(row, col, circle_row_center, circle_col_center,
                                                                 radius_in_pixel, in_value, out_value);
            if (inside)
            {
                image(row, col) = val;
            }
        }
    }
}

void draw_footer(cv::Mat1b &image, const BoardHexGrid &board, const float cm_per_pixel)
{
    const cv::HersheyFonts font_face = cv::FONT_HERSHEY_DUPLEX;
    const float footer_height_cm = .15f;
    const int footer_height_pixels = footer_height_cm / cm_per_pixel;
    const int footer_thickness = 10;
    const double footer_scale = getFontScaleFromHeight(font_face, footer_height_pixels, footer_thickness);
    const auto [row_center, col_center] = on_image_location(board, board.rows_ - 1, 1, cm_per_pixel);

    spdlog::info("footer_height_pixels={}, footer_scale={}, (c,r): ({}, {}), ({}, {})", footer_height_pixels,
                 footer_scale, col_center, row_center, board.top_left_(0) / cm_per_pixel,
                 board.bottom_right_(1) / cm_per_pixel);

    const std::string footer = std::format(
        "HexBoard by k.mrozowski | {}x{} | Coding: ({},{}), ({},{}) | Spacing: {:0.3f} | Outer: {:0.3f} | Inner: "
        "{:0.3f} | Is even: {}",
        board.rows_, board.cols_, board.row_left_, board.col_left_, board.row_right_, board.col_right_,
        board.spacing_cols_, board.outer_radius_, board.inner_radius_, board.is_even_);

    spdlog::info("Footer: '{}'", footer);
    cv::putText(image, footer,
                cv::Point(0.6f * (board.top_left_(0) - board.outer_radius_) / cm_per_pixel,
                          0.6f * (board.top_left_(1) - board.outer_radius_) / cm_per_pixel),
                cv::FONT_HERSHEY_COMPLEX, footer_scale, cv::Scalar(0, 0, 0), footer_thickness, cv::LINE_AA, false);
}

}  // namespace

std::pair<int, bool> board::accurate_intensity(const float row_center, const float col_center,
                                               const float circle_row_center, const float circle_col_center,
                                               const float radius_in_pixel, const float in_value, const float out_value)
{
    Eigen::Vector2d point_to_center(circle_row_center - row_center, circle_col_center - col_center);
    const float norm = point_to_center.norm();
    const float diff = (norm - radius_in_pixel);

    // we iterate in floating points, and with assumption of square pixels we can check if current pixel center does
    // overlap with circle edge. As we measure distance from center of pixel, if intersection with circle is further
    // than half pixel (one of the corner) expanded by some safety factor.
    if (std::abs(diff) > std::sqrt(0.5 * 0.5 * 2))
    {
        if (diff < 0)
        {
            return {in_value, true};
        }
        return {out_value, false};
    }
    const float outside_val = percentage_outside(row_center, col_center, circle_row_center, circle_col_center,
                                                 radius_in_pixel * radius_in_pixel);
    const float diff_val = in_value - out_value;

    return {std::round(out_value + outside_val * diff_val), true};
}

cv::Mat1b board::draw_canonical_board(const BoardRectGrid &board, const float cm_per_pixel)
{
    const Eigen::Vector3d dimension = board.bottom_right_;

    const int rows = std::ceil(dimension(1) / cm_per_pixel + 1);
    const int cols = std::ceil(dimension(0) / cm_per_pixel + 1);

    cv::Mat1b image = cv::Mat1b(rows, cols, 255);

    const float inner_radius_pix = board.inner_radius_ / cm_per_pixel;
    const float outer_radius_pix = board.outer_radius_ / cm_per_pixel;

#pragma omp parallel for
    for (int row = 0; row < board.rows_; ++row)
    {
        for (int col = 0; col < board.cols_; ++col)
        {
            const auto [row_center, col_center] = on_image_location(board, row, col, cm_per_pixel);

            if (row == board.row_top_ && col == board.col_top_)
            {
                draw_circle(image, row_center, col_center, outer_radius_pix, 0.0f, 255.0f);
            }
            else if (row == board.row_down_ && col == board.col_down_)
            {
                draw_circle(image, row_center, col_center, outer_radius_pix, 0.0f, 255.0f);
            }
            else if (row == board.row_right_ && col == board.col_right_)
            {
                draw_circle(image, row_center, col_center, outer_radius_pix, 0.0f, 255.0f);
            }
            else
            {
                draw_circle(image, row_center, col_center, outer_radius_pix, 0.0f, 255.0f);
                draw_circle(image, row_center, col_center, inner_radius_pix, 255.0f, 0.0f);
            }
        }
    }

    return image;
}

cv::Mat1b board::draw_canonical_board(const BoardHexGrid &board, const float cm_per_pixel,
                                      const cv::Size_<float> board_dimention_mm)
{
    const Eigen::Vector3f board_size_cm = board.bottom_right_.cast<float>();
    spdlog::debug("board_size_cm=({:0.2f}, {:0.2f}), cm_per_pixel={}", board_size_cm(0), board_size_cm(1),
                  cm_per_pixel);

    const int rows_needed = std::ceil(board_size_cm(1) / cm_per_pixel + 1);
    const int cols_needed = std::ceil(board_size_cm(0) / cm_per_pixel + 1);

    const int rows = board_dimention_mm.height / cm_per_pixel;
    const int cols = board_dimention_mm.width / cm_per_pixel;

    cv::Mat1b image = cv::Mat1b(rows, cols, 255);
    spdlog::debug("image.size=({}, {}), max_size=({}, {}), free_space=({}, {})", rows, cols, rows_needed, cols_needed,
                  rows - rows_needed, cols - cols_needed);
    if (rows < rows_needed || cols < cols_needed)
    {
        throw std::invalid_argument("Board is too big for this DPI.");
    }

    const float inner_radius_pix = board.inner_radius_ / cm_per_pixel;
    const float outer_radius_pix = board.outer_radius_ / cm_per_pixel;

    draw_footer(image, board, cm_per_pixel);

#pragma omp parallel for
    for (int row = 0; row < board.rows_; ++row)
    {
        for (int col = 0; col < board.cols_; ++col)
        {
            const auto [row_center, col_center] = on_image_location(board, row, col, cm_per_pixel);

            if (row == board.row_left_ && col == board.col_left_)
            {
                draw_circle(image, row_center, col_center, outer_radius_pix, 0.0f, 255.0f);
            }
            else if (row == board.row_right_ && col == board.col_right_)
            {
                draw_circle(image, row_center, col_center, outer_radius_pix, 0.0f, 255.0f);
            }
            else
            {
                draw_circle(image, row_center, col_center, outer_radius_pix, 0.0f, 255.0f);
                draw_circle(image, row_center, col_center, inner_radius_pix, 255.0f, 0.0f);
            }
        }
    }

    return image;
}
