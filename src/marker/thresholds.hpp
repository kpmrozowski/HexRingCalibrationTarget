#pragma once

#include <opencv2/core.hpp>

/**
 * @brief Algorithm used for binarization (Otsu method)
 * https://en.wikipedia.org/wiki/Otsu%27s_method
 */
namespace thresholds
{
struct TiledThresholds
{
    std::vector<int> thresholds_;
    const int tiles_row_;
    const int tiles_col_;

    TiledThresholds(const int tiles_row, const int tiles_col);
};

/**
 * @brief Binarize using Opencv backend
 *
 * @param input image to be binarized
 */
cv::Mat1b binarize_image(const cv::Mat1b &input);

/**
 * @brief Split image into ties, and compute thresholds in each of them using Otsu method
 *
 * @param input image to be binarized
 * @param tiles_row number of tiles in rows
 * @param tiles_col number of tiles in cols
 *
 * @return set of thresholds for each tiles
 */
TiledThresholds thresholds(const cv::Mat1b &image, const int tiles_row, const int tiles_col);

/**
 * @brief Perform binarization of image using computed thresholds in tiles.To get "tresholds" parameter we need to call
 * "tresholds" function.
 *
 * @param input image to be binarized
 * @param thresholds set of thresholds
 * @param tiles_row number of tiles in rows
 * @param tiles_col number of tiles in cols
 *
 * @return set of thresholds for each tiles
 */
cv::Mat1b binarize(const cv::Mat1b &image, const TiledThresholds &tresholds);

}  // namespace thresholds