#include "calibrate_mono.hpp"

#include <cstdlib>
#include <map>

#include <spdlog/spdlog.h>

#include "images_set.hpp"
#include "initial_calibration/debug.hpp"
#include "initial_calibration/precalibration.hpp"
#include "marker/detection.hpp"

void CalibrateMono::execute()
{
    std::map<int, base::ImageDecoding> decoded;

    ImageFilesDataset images_set(dataset_folder_, camera_id_);
    const auto data_container = images_set();

    const std::unique_ptr<Board> calibration_board = !board_params_vec_.empty()
                                                         ? board::get_board(board_type_, board_params_vec_)
                                                         : board::get_board(board_params_path_);

    for (const ImageFileDescriptor& descriptor : data_container)
    {
        const auto [image, image_id] = descriptor.read_image();
        cv::Mat1b mat = image;
        const auto decoded_image = marker::detection::detect_and_identify(
            mat,
            marker::DetectionParameters(650.0, calibration_board->inner_radius_ * 2,
                                        calibration_board->outer_radius_ * 2, 100.0, 1000.0),
            calibration_board, image_id);

        if (decoded_image.has_value())
        {
            decoded.emplace(std::make_pair(image_id, decoded_image.value()));
        }
        spdlog::info("processed cam {} pos {}", camera_id_, image_id);
    }

    auto pre_calibration = precalibration::initial_calibration(decoded, calibration_board->marker_centers_);
    std::cout << "\ncamera_matrix:\n" << pre_calibration.camera_matrix_ << "\n";
    std::cout << "distortions: [ " << pre_calibration.distortions_.transpose() << " ]\n";
    std::cout << "image_cols_rows: ( " << pre_calibration.image_cols_ << ", " << pre_calibration.image_rows_ << " )\n";
    std::cout << "reprojection_RMSE: " << pre_calibration.reproj_rmse_ << "\n\n";

    precalibration::debug::save_calibration(pre_calibration, camera_id_, output_folder_);
}
