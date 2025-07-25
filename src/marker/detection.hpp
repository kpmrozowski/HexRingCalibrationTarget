#pragma once

#include <optional>

#include "board.hpp"
#include "calibration.hpp"
#include "detection_parameters.hpp"

namespace marker::detection
{
std::optional<base::ImageDecoding> detect_and_identify(cv::Mat1b &input, const DetectionParameters &parameters,
                                                       const std::unique_ptr<Board> &board, const int image_idx);
}  // namespace marker::detection
