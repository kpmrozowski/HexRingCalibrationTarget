#include "thresholds.hpp"

#include <opencv2/imgproc.hpp>

#include <io/debug.hpp>

#include "debugging.hpp"

namespace
{
static constexpr std::string_view kThresholdsSubdir = "thresholds";
}

namespace debug
{
void paint_tresholds(const cv::Mat1b &image, const std::vector<int> &tresholds, const int tiles_row,
                     const int tiles_col)
{
    cv::Mat3b painted;
    cv::cvtColor(image, painted, cv::COLOR_GRAY2BGR);

    const int row_step = image.rows / tiles_row;
    const int col_step = image.cols / tiles_col;

    for (int row_tile = 0; row_tile < tiles_row; ++row_tile)
    {
        const int start_row = row_tile * row_step;
        const int middle_row = row_tile * row_step + row_step / 2;

        for (int col_tile = 0; col_tile < tiles_col; ++col_tile)
        {
            const int start_col = col_tile * col_step;
            const int middle_col = col_tile * col_step + col_step / 2;

            const int tresh_to_use = tresholds[row_tile * tiles_col + col_tile];

            cv::rectangle(painted, cv::Rect2i(start_col, start_row, col_step, row_step), cv::Scalar(0, 255, 0), 1);

            cv::putText(painted, std::to_string(tresh_to_use), cv::Point(middle_col, middle_row),
                        cv::FONT_HERSHEY_COMPLEX, 0.3, cv::Scalar(0, 255, 0));
        }
    }

    io::debug::save_image(painted, "thresholds", kThresholdsSubdir);
}
}  // namespace debug

namespace
{
int thresholds_in_tile(const int start_row, const int end_row, const int start_col, const int end_col,
                       const cv::Mat1b &image)
{
    std::vector<int> histogram(256, 0);
    const int all_pixels = (end_row - start_row) * (end_col - start_col);

    for (int row = start_row; row < end_row; ++row)
    {
        for (int col = start_col; col < end_col; ++col)
        {
            const int value = image(row, col);
            histogram[value]++;
        }
    }

    // Compute threshold
    float sum = 0.0f;
    float sum_b = 0.0f;
    int q1 = 0;
    int q2 = 0;
    float var_max = 0;
    int threshold = 0;

    // Auxiliary value for computing m2
    for (int i = 0; i < histogram.size(); i++)
    {
        sum += i * histogram[i];
    }

    for (int i = 0; i < 256; i++)
    {
        // Update q1
        q1 += histogram[i];
        if (q1 == 0)
        {
            continue;
        }

        // Update q2
        q2 = all_pixels - q1;
        if (q2 == 0)
        {
            break;
        }

        // Update m1 and m2
        sum_b += (float)(i * histogram[i]);
        float m1 = sum_b / q1;
        float m2 = (sum - sum_b) / q2;

        // Update the between class variance
        float var_between = (float)q1 * (float)q2 * (m1 - m2) * (m1 - m2);

        // Update the threshold if necessary
        if (var_between > var_max)
        {
            var_max = var_between;
            threshold = i;
        }
    }

    return threshold;
}
}  // namespace

thresholds::TiledThresholds::TiledThresholds(const int tiles_row, const int tiles_col)
    : tiles_row_(tiles_row), tiles_col_(tiles_col)
{
    thresholds_.resize(tiles_row * tiles_col);
}

cv::Mat1b thresholds::binarize_image(const cv::Mat1b &input)
{
    cv::Mat1b out;
    cv::threshold(input, out, 0, 255, cv::ThresholdTypes::THRESH_BINARY | cv::ThresholdTypes::THRESH_OTSU);
    return out;
}

thresholds::TiledThresholds thresholds::thresholds(const cv::Mat1b &image, const int tiles_row, const int tiles_col)
{
    thresholds::TiledThresholds tiled_thresholds(tiles_row, tiles_col);

    const int row_step = image.rows / tiles_row;
    const int col_step = image.cols / tiles_col;

    for (int row_tile = 0; row_tile < tiles_row; ++row_tile)
    {
        const int start_row = row_tile * row_step;
        const int end_row = std::min(start_row + row_step, image.rows);
        for (int col_tile = 0; col_tile < tiles_col; ++col_tile)
        {
            const int start_col = col_tile * col_step;
            const int end_col = std::min(start_col + col_step, image.cols);

            tiled_thresholds.thresholds_[row_tile * tiles_col + col_tile] =
                thresholds_in_tile(start_row, end_row, start_col, end_col, image);
        }
    }
    return tiled_thresholds;
}

cv::Mat1b thresholds::binarize(const cv::Mat1b &image, const TiledThresholds &tresholds)
{
    cv::Mat1b binarized(image.rows, image.cols);

    const int row_step = image.rows / tresholds.tiles_row_;
    const int col_step = image.cols / tresholds.tiles_col_;

    for (int row_tile = 0; row_tile < tresholds.tiles_row_; ++row_tile)
    {
        const int start_row = row_tile * row_step;
        const int end_row = std::min(start_row + row_step, image.rows);
        for (int col_tile = 0; col_tile < tresholds.tiles_col_; ++col_tile)
        {
            const int start_col = col_tile * col_step;
            const int end_col = std::min(start_col + col_step, image.cols);

            const int tresh_to_use = tresholds.thresholds_[row_tile * tresholds.tiles_col_ + col_tile];

            for (int row = start_row; row < end_row; ++row)
            {
                for (int col = start_col; col < end_col; ++col)
                {
                    const int value = image(row, col);
                    binarized(row, col) = (value > tresh_to_use) * 255;
                }
            }
        }
    }

    if (kShowTresholds)
    {
        debug::paint_tresholds(image, tresholds.thresholds_, tresholds.tiles_row_, tresholds.tiles_col_);
        io::debug::save_image(binarized, "binarized", kThresholdsSubdir);
    }

    return binarized;
}
