#include "calibrate_mono.hpp"

#include <cstdlib>
#include <map>

#include <spdlog/spdlog.h>

#include "images_set.hpp"
#include "initial_calibration/debug.hpp"
#include "initial_calibration/precalibration.hpp"
#include "marker/detection.hpp"

namespace
{
BoardHexGrid::Params get_hex_params(const std::vector<int>& board_params)
{
    BoardHexGrid::Params params;
    switch (board_params.size())
    {
        case 6:
            params.col_right = board_params.at(5);
        case 5:
            params.row_right = board_params.at(4);
        case 4:
            params.col_left = board_params.at(3);
        case 3:
            params.row_left = board_params.at(2);
        case 2:
            params.cols = board_params.at(1);
        case 1:
            params.rows = board_params.at(0);
    }

    return params;
}

std::unique_ptr<Board> get_board(const int board_type, const std::vector<int>& board_params)
{
    std::unique_ptr<Board> calibration_board;
    switch ((BoardType)board_type)
    {
        case BoardType::RECT:
            return std::make_unique<BoardRectGrid>();
        case BoardType::HEX:
            return std::make_unique<BoardHexGrid>(get_hex_params(board_params));
    }
}

}  // namespace

void CalibrateMono::execute()
{
    std::map<int, base::ImageDecoding> decoded;

    const std::unique_ptr<Board> calibration_board = get_board(board_type_, board_params_);

    ImageFilesDataset dataset(dataset_folder_, camera_id_);
    const auto data_container = dataset();

    for (const ImageFileDescriptor& descriptor : data_container)
    {
        const auto [image, image_id] = descriptor.read_image();
        cv::Mat1b mat = image;
        const auto decoded_image = marker::detection::detect_and_identify(
            mat,
            marker::DetectionParameters(650.0, calibration_board->inner_radius_ * 2,
                                        calibration_board->outer_radius_ * 2, 10.0, 100.0),
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
