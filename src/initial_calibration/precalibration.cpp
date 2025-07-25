#include "precalibration.hpp"

#include <spdlog/spdlog.h>
#include <Eigen/Geometry>
#include <opencv2/calib3d.hpp>

#include "../transform.hpp"

namespace
{
struct OpenCVWrapper
{
    std::vector<std::vector<cv::Point3f>> board_points_;
    std::vector<std::vector<cv::Point2f>> image_points_;
    std::vector<int> vector_to_image_id_;
    cv::Size sensor_size_;
    cv::Size sensor_size_for_calibration_init_;
};

// 3d location / 2d observation / vector to global idx
OpenCVWrapper flat_data_and_zero_z(const std::map<int, base::ImageDecoding> &data,
                                   const std::vector<Eigen::Vector3d> &board_points)
{
    OpenCVWrapper wrapper;

    for (const auto &[id, decoding] : data)
    {
        wrapper.sensor_size_ = cv::Size(decoding.linear_input_.cols, decoding.linear_input_.rows);

        const auto &markers = decoding.coding_markers_.markers_;

        wrapper.vector_to_image_id_.emplace_back(id);

        auto &pts = wrapper.board_points_.emplace_back();
        auto &obs = wrapper.image_points_.emplace_back();

        for (size_t idx = 0; idx < markers.size(); ++idx)
        {
            const int marker_id = markers[idx].global_id_;
            pts.emplace_back(board_points[marker_id](0), board_points[marker_id](1), 0.0f);
            obs.emplace_back(markers[idx].col_, markers[idx].row_);
        }
    }

    // for camera center is usually at center
    wrapper.sensor_size_for_calibration_init_ = wrapper.sensor_size_;

    return wrapper;
}

Eigen::Quaterniond cv_rot_to_eigen(const cv::Mat1d &rotation)
{
    cv::Mat rotation_matrix;
    if (rotation.rows == 3 && rotation.cols == 3)
    {
        rotation_matrix = rotation;
    }
    else
    {
        cv::Rodrigues(rotation, rotation_matrix);
    }

    Eigen::Matrix3d rotation_eig;

    for (int row = 0; row < 3; ++row)
    {
        for (int col = 0; col < 3; ++col)
        {
            rotation_eig(row, col) = rotation_matrix.at<double>(row, col);
        }
    }
    return Eigen::Quaterniond(rotation_eig);
}

Eigen::Vector3d cv_translation_to_eigen(const cv::Mat1d &translation)
{
    Eigen::Vector3d translation_eig;

    for (int row = 0; row < 3; ++row)
    {
        translation_eig(row) = translation(row);
    }
    return translation_eig;
}

std::map<int, Transformd> concentrate_transformations(const std::vector<cv::Mat> &rotations,
                                                      const std::vector<cv::Mat> &translations,
                                                      const std::vector<int> &mapping)
{
    std::map<int, Transformd> transformations;
    for (size_t idx_poz = 0; idx_poz < rotations.size(); ++idx_poz)
    {
        const int position_idx = mapping[idx_poz];
        auto &position = transformations[position_idx];
        position.rotation_ = cv_rot_to_eigen(rotations[idx_poz]);
        position.translation_ = cv_translation_to_eigen(translations[idx_poz]);
    }
    return transformations;
}

precalibration::PinholeCalibration get_calibration(const cv::Mat1d &camera, const cv::Mat1d &distortions)
{
    precalibration::PinholeCalibration calibration;

    for (int row = 0; row < 3; ++row)
    {
        for (int col = 0; col < 3; ++col)
        {
            calibration.camera_matrix_(row, col) = camera(row, col);
        }
    }

    for (int row = 0; row < 5; ++row)
    {
        calibration.distortions_[row] = distortions(row);
    }

    return calibration;
}

precalibration::PinholeCalibration calculate(const OpenCVWrapper &wrapper)
{
    cv::Mat1d camera, distortions;
    std::vector<cv::Mat> rotations, translations;

    spdlog::info("Running initial calibration.");
    const double error = cv::calibrateCamera(wrapper.board_points_, wrapper.image_points_,
                                             wrapper.sensor_size_for_calibration_init_, camera, distortions, rotations,
                                             translations, 0, cv::TermCriteria(cv::TermCriteria::COUNT, 100, 0.0));


    const auto board_positions = concentrate_transformations(rotations, translations, wrapper.vector_to_image_id_);

    auto calibration_pinhole = get_calibration(camera, distortions);
    calibration_pinhole.image_rows_ = wrapper.sensor_size_.height;
    calibration_pinhole.image_cols_ = wrapper.sensor_size_.width;
    calibration_pinhole.reproj_rmse_ = error;

    spdlog::info("The overall RMS re-projection error: {}", calibration_pinhole.reproj_rmse_);

    return calibration_pinhole;
}

}  // namespace

precalibration::PinholeCalibration precalibration::initial_calibration(
    const std::map<int, base::ImageDecoding> &decoding, const std::vector<Eigen::Vector3d> &board_points)
{
    const auto wrapper = flat_data_and_zero_z(decoding, board_points);
    auto calibration = calculate(wrapper);
    return calibration;
}
