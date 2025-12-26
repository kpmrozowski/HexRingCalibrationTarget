#pragma once

#include <filesystem>
#include <optional>

#include "board.hpp"
#include "calibration.hpp"
#include "detection_parameters.hpp"

class BoardCircleGrid;

namespace identification::circlegrid
{
struct TrackingState;
}

namespace marker::detection
{
std::optional<base::ImageDecoding> detect_and_identify(cv::Mat1b &input, const DetectionParameters &parameters,
                                                       const std::unique_ptr<Board> &board, const int image_idx);

std::optional<base::ImageDecoding> detect_and_identify_circlegrid(
    cv::Mat1b &input, const DetectionParameters &parameters, const BoardCircleGrid &board,
    identification::circlegrid::TrackingState &tracker_state, const int image_idx,
    const std::filesystem::path &output_path = {});

}  // namespace marker::detection
