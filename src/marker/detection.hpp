#pragma once

#include <filesystem>
#include <optional>
#include <vector>

#include <opencv2/core.hpp>

#include "board.hpp"
#include "calibration.hpp"
#include "detection_parameters.hpp"

class BoardCircleGrid;

namespace identification::circlegrid
{
struct TrackingState;
struct LightTrackingState;
}

namespace marker::detection
{
std::optional<base::ImageDecoding> detect_and_identify(cv::Mat1b &input, const DetectionParameters &parameters,
                                                       const std::unique_ptr<Board> &board, const int image_idx,
                                                       const std::filesystem::path &output_path = {});

base::ImageDecoding detect_and_identify_circlegrid(
    cv::Mat1b &input, const DetectionParameters &parameters, const BoardCircleGrid &board,
    identification::circlegrid::TrackingState &tracker_state, const int image_idx,
    const std::filesystem::path &output_path = {});

struct CirclegridExtractionResult
{
    bool found_any = false;
    std::vector<base::MarkerCoding> coding_markers;
    std::vector<int> grid_indices;
    bool find_circles_grid_succeeded = false;
    cv::Mat1b scaled_input;
    cv::Mat1b binarized;
    cv::Mat1b inverted_binarization;
    float brightness_scale = 1.0f;
};

CirclegridExtractionResult extract_coding_markers(const cv::Mat1b &input, const DetectionParameters &parameters,
                                                   const BoardCircleGrid &board, int image_idx);

base::ImageDecoding identify_circlegrid_from_markers(const CirclegridExtractionResult &extraction,
                                                      const DetectionParameters &parameters,
                                                      const BoardCircleGrid &board,
                                                      identification::circlegrid::TrackingState &tracker_state,
                                                      int image_idx, const std::filesystem::path &output_path = {},
                                                      bool use_ecc_validation = false);

struct IdentificationResult
{
    bool success = false;
    std::vector<base::MarkerRing> markers;
};

/// Lightweight identification: returns only markers with global IDs.
/// No image allocation, no ECC validation, no marker_area/calibrated_area creation.
IdentificationResult identify_markers_only(const CirclegridExtractionResult &extraction,
                                           const BoardCircleGrid &board,
                                           identification::circlegrid::LightTrackingState &tracker_state,
                                           int image_idx);

}  // namespace marker::detection
