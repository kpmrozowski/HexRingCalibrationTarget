#include "symmetries.hpp"

namespace
{
int mean_in_small_window(const cv::Mat1b &image, const int row, const int col)
{
    int average = 0;
    int elements = 0;
    for (int row_iter = std::max(row - 1, 0); row_iter < std::min(row + 2, image.rows - 1); ++row_iter)
    {
        for (int col_iter = std::max(col - 1, 0); col_iter < std::min(col + 2, image.cols - 1); ++col_iter)
        {
            average += image(row_iter, col_iter);
            elements++;
        }
    }
    if (elements == 0)
    {
        return -1;
    }
    return average / elements;
}

int max_in_small_window(const cv::Mat1b &image, const int row, const int col)
{
    int max = 0;
    for (int row_iter = std::max(row - 1, 0); row_iter < std::min(row + 2, image.rows - 1); ++row_iter)
    {
        for (int col_iter = std::max(col - 1, 0); col_iter < std::min(col + 2, image.cols - 1); ++col_iter)
        {
            max = std::max(max, int(image(row_iter, col_iter)));
        }
    }
    return max;
}

std::pair<int, int> minimum_over_line_col(const cv::Mat1b &image, const int row_center, const int col_center,
                                          const int start_value, const int width, int direction)
{
    int minimum = std::numeric_limits<int>::max();
    int location = start_value;
    for (int on_line = start_value; on_line <= width; ++on_line)
    {
        const int current_minimum = mean_in_small_window(image, row_center, col_center + on_line * direction);
        if (current_minimum == -1)
        {
            break;
        }

        if (current_minimum < minimum)
        {
            minimum = current_minimum;
            location = on_line;
        }
    }
    return {minimum, location};
}

std::pair<int, int> maximum_over_line_col(const cv::Mat1b &image, const int row_center, const int col_center,
                                          const int start_value, const int width, int direction)
{
    int maximum = std::numeric_limits<int>::min();
    int location = start_value;
    for (int on_line = start_value; on_line <= width; ++on_line)
    {
        const int current_maximum = mean_in_small_window(image, row_center, col_center + on_line * direction);
        if (current_maximum == -1)
        {
            break;
        }
        if (current_maximum > maximum)
        {
            maximum = current_maximum;
            location = on_line;
        }
    }
    return {maximum, location};
}

std::pair<int, int> minimum_over_line_row(const cv::Mat1b &image, const int row_center, const int col_center,
                                          const int start_value, const int height, int direction)
{
    int minimum = std::numeric_limits<int>::max();
    int location = start_value;

    for (int on_line = start_value; on_line <= height; ++on_line)
    {
        const int current_minimum = mean_in_small_window(image, row_center + on_line * direction, col_center);
        if (current_minimum == -1)
        {
            break;
        }

        if (current_minimum < minimum)
        {
            minimum = current_minimum;
            location = on_line;
        }
    }
    return {minimum, location};
}

std::pair<int, int> maximum_over_line_row(const cv::Mat1b &image, const int row_center, const int col_center,
                                          const int start_value, const int height, int direction)
{
    int maximum = std::numeric_limits<int>::min();
    int location = start_value;
    for (int on_line = start_value; on_line <= height; ++on_line)
    {
        const int current_maximum = mean_in_small_window(image, row_center + on_line * direction, col_center);
        if (current_maximum == -1)
        {
            break;
        }

        if (current_maximum > maximum)
        {
            maximum = current_maximum;
            location = on_line;
        }
    }
    return {maximum, location};
}

bool is_within_bounds(const int outer, const int center, const float scale)
{
    const int offset = outer * (scale - 1.0f);
    return (center < outer + offset && center > outer - offset);
}

}  // namespace

namespace marker::symetries
{

base::Difference test_intensities_values_and_symmetries_cores(const cv::Mat1b &image, const int row_center,
                                                              const int col_center, const int width, const int height,
                                                              const float min_difference_scale)
{
    // for derivation of cores, AND small markers said mean is biased to low values
    // (compare to white area outside marker [with is usually brighter due to blurring]).
    // But we later compare it with lowest value over given span with is picked as minimum, so it's also biased to low
    // values
    const int center_intensity = mean_in_small_window(image, row_center, col_center);

    const auto [left_intensity, left_loc] = minimum_over_line_col(image, row_center, col_center, 1, width + 1, 1);
    const auto [right_intensity, right_loc] = minimum_over_line_col(image, row_center, col_center, 1, width + 1, -1);

    const auto [up_intensity, up_loc] = minimum_over_line_row(image, row_center, col_center, 1, height, 1);
    const auto [down_intensity, down_loc] = minimum_over_line_row(image, row_center, col_center, 1, height, -1);

    // check if inner is brighter
    const bool left_brighter = left_intensity * min_difference_scale < center_intensity;
    const bool right_brighter = right_intensity * min_difference_scale < center_intensity;
    const bool up_brighter = up_intensity * min_difference_scale < center_intensity;
    const bool down_brighter = down_intensity * min_difference_scale < center_intensity;

    if (left_brighter && right_brighter && up_brighter && down_brighter)
    {
        return base::Difference::INNER_BRIGHTER;
    }
    else if (!left_brighter && !right_brighter && !up_brighter && !down_brighter)
    {
        return base::Difference::OUTER_BRIGHTER;
    }
    return base::Difference::NO_DIFFERENCE;
}

base::Difference test_intensities_values_and_symmetries_rings(const cv::Mat1b &image, const int row_center,
                                                              const int col_center, const int width, const int height,
                                                              const float min_difference_scale)
{
    // If we have small markers it's better to pick up max value in window as average is biased to low values
    // due to blurring. Outside values are biased to bigger values so it kind of offset offect
    // TODO: pick some fixed median
    const int center_intensity = max_in_small_window(image, row_center, col_center);

    const auto [left_intensity_min, left_loc_min] =
        minimum_over_line_col(image, row_center, col_center, 1, width + 1, 1);
    const auto [right_intensity_min, right_loc_min] =
        minimum_over_line_col(image, row_center, col_center, 1, width + 1, -1);

    const auto [up_intensity_min, up_loc_min] = minimum_over_line_row(image, row_center, col_center, 1, height, 1);
    const auto [down_intensity_min, down_loc_min] = minimum_over_line_row(image, row_center, col_center, 1, height, -1);

    // search for maximal after minumum
    const auto [left_intensity_max, left_loc_max] =
        maximum_over_line_col(image, row_center, col_center, left_loc_min, width + 1, 1);
    const auto [right_intensity_max, right_loc_max] =
        maximum_over_line_col(image, row_center, col_center, right_loc_min, width + 1, -1);

    const auto [up_intensity_max, up_loc_max] =
        maximum_over_line_row(image, row_center, col_center, up_loc_min, height, 1);
    const auto [down_intensity_max, down_loc_max] =
        maximum_over_line_row(image, row_center, col_center, down_loc_min, height, -1);

    // check it inner is brighter

    const bool left_simmilar = is_within_bounds(left_intensity_max, center_intensity, min_difference_scale);
    const bool right_simmilar = is_within_bounds(right_intensity_max, center_intensity, min_difference_scale);
    const bool up_simmilar = is_within_bounds(up_intensity_max, center_intensity, min_difference_scale);
    const bool down_simmilar = is_within_bounds(down_intensity_max, center_intensity, min_difference_scale);

    if (left_simmilar && right_simmilar && up_simmilar && down_simmilar)
    {
        // no difference so it is ring
        return base::Difference::NO_DIFFERENCE;
    }
    // check it inner is brighter
    const bool left_brighter = left_intensity_max * min_difference_scale > center_intensity;
    const bool right_brighter = right_intensity_max * min_difference_scale > center_intensity;
    const bool up_brighter = up_intensity_max * min_difference_scale > center_intensity;
    const bool down_brighter = down_intensity_max * min_difference_scale > center_intensity;

    if (left_brighter && right_brighter && up_brighter && down_brighter)
    {
        return base::Difference::OUTER_BRIGHTER;
    }
    return base::Difference::INNER_BRIGHTER;
}

}  // namespace marker::symetries
