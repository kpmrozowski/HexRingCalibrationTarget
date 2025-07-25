#pragma once

#include "calibration.hpp"

namespace precalibration
{
struct PinholeCalibration
{
    Eigen::Matrix3d camera_matrix_;
    Eigen::Matrix<double, 5, 1> distortions_;

    int image_rows_, image_cols_;
    double reproj_rmse_;
};

PinholeCalibration initial_calibration(const std::map<int, base::ImageDecoding> &decoding,
                                       const std::vector<Eigen::Vector3d> &board_points);

}  // namespace precalibration