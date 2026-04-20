#include "calibrate_mono.hpp"

#include <cstdlib>
#include <filesystem>
#include <map>

#include <spdlog/spdlog.h>

#include "identification/board_circle/identification_circle.hpp"
#include "images_set.hpp"
#include "initial_calibration/debug.hpp"
#include "initial_calibration/precalibration.hpp"
#include "marker/detection.hpp"
#include "marker/repair_dropped_neighbors.hpp"

namespace
{
uint64_t parse_timestamp_ns_from_stem(const std::filesystem::path& path)
{
    const std::string stem = path.stem().string();
    uint64_t ts = 0;
    for (const char character : stem)
    {
        if (character < '0' || character > '9')
        {
            return 0;
        }
        ts = ts * 10ULL + static_cast<uint64_t>(character - '0');
    }
    return ts;
}
}  // namespace

void CalibrateMono::execute()
{
    std::map<int, base::ImageDecoding>            decoded;
    std::map<int, marker::repair::FrameCacheEntry> frame_cache;

    ImageFilesDataset images_set(dataset_folder_, camera_id_, start_idx_);
    const auto data_container = images_set();

    const std::unique_ptr<Board> calibration_board = !board_params_vec_.empty()
                                                         ? board::get_board(board_type_, board_params_vec_)
                                                         : board::get_board(board_params_path_);

    identification::circlegrid::TrackingState tracker_state;

    for (const ImageFileDescriptor& descriptor : data_container)
    {
        const auto [image, image_id] = descriptor.read_image();
        cv::Mat1b mat = image;
        const auto decoded_image = [&]() -> std::optional<base::ImageDecoding>
        {
            if (calibration_board->type_ == BoardType::CIRCLE)
            {
                const BoardCircleGrid* circle_board = dynamic_cast<const BoardCircleGrid*>(calibration_board.get());

                return marker::detection::detect_and_identify_circlegrid(
                    mat,
                    marker::DetectionParameters(650.0, circle_board->outer_radius_ * 2, circle_board->outer_radius_ * 2,
                                                100.0, 1000.0),
                    *circle_board, tracker_state, image_id, output_folder_);
            }
            else
            {
                return marker::detection::detect_and_identify(
                    mat,
                    marker::DetectionParameters(650.0, calibration_board->inner_radius_ * 2,
                                                calibration_board->outer_radius_ * 2, 100.0, 1000.0),
                    calibration_board, image_id, output_folder_);
            }
        }();

        if (decoded_image.has_value())
        {
            decoded.emplace(std::make_pair(image_id, decoded_image.value()));

            if (calibration_board->type_ == BoardType::CIRCLE)
            {
                marker::repair::FrameCacheEntry entry;
                entry.ts_ns          = parse_timestamp_ns_from_stem(descriptor.path());
                entry.image_path     = std::filesystem::path(descriptor.path());
                entry.coding_markers = decoded_image->all_detected_markers_;
                entry.method         = tracker_state.last_method_;
                entry.fcg_succeeded  = (entry.method == "findCirclesGrid");

                const int total_markers =
                    static_cast<int>(calibration_board->rows_ * calibration_board->cols_);
                entry.marker_positions.assign(total_markers, cv::Point2f(-1.f, -1.f));

                const Eigen::Matrix<std::optional<int>, -1, -1>& ordering =
                    decoded_image->coding_markers_.ordering_;
                const std::vector<base::MarkerRing>& rings = decoded_image->coding_markers_.markers_;
                for (int row_idx = 0; row_idx < ordering.rows(); ++row_idx)
                {
                    for (int col_idx = 0; col_idx < ordering.cols(); ++col_idx)
                    {
                        const std::optional<int>& ring_index = ordering(row_idx, col_idx);
                        if (!ring_index.has_value())
                        {
                            continue;
                        }
                        const int gid = row_idx * static_cast<int>(calibration_board->cols_) + col_idx;
                        if (gid < 0 || gid >= total_markers)
                        {
                            continue;
                        }
                        if (ring_index.value() < 0
                            || ring_index.value() >= static_cast<int>(rings.size()))
                        {
                            continue;
                        }
                        const base::MarkerRing& ring = rings[ring_index.value()];
                        entry.marker_positions[gid] = cv::Point2f(ring.col_, ring.row_);
                        ++entry.identified_count;
                    }
                }

                frame_cache.emplace(image_id, std::move(entry));
            }
        }
        spdlog::info("processed cam {} pos {}", camera_id_, image_id);
    }

    if (calibration_board->type_ == BoardType::CIRCLE)
    {
        const BoardCircleGrid* circle_board = dynamic_cast<const BoardCircleGrid*>(calibration_board.get());
        marker::DetectionParameters params_for_repair(
            650.0, circle_board->outer_radius_ * 2, circle_board->outer_radius_ * 2, 100.0, 1000.0);
        params_for_repair.repair_dropped_neighbors_ = !repair_disabled_;
        marker::repair::run_repair_pass(frame_cache, params_for_repair, *circle_board, decoded, output_folder_);
    }

    auto pre_calibration = precalibration::initial_calibration(decoded, calibration_board->marker_centers_);
    std::cout << "\ncamera_matrix:\n" << pre_calibration.camera_matrix_ << "\n";
    std::cout << "distortions: [ " << pre_calibration.distortions_.transpose() << " ]\n";
    std::cout << "image_cols_rows: ( " << pre_calibration.image_cols_ << ", " << pre_calibration.image_rows_ << " )\n";
    std::cout << "reprojection_RMSE: " << pre_calibration.reproj_rmse_ << "\n\n";

    precalibration::debug::save_calibration(pre_calibration, camera_id_, output_folder_);
}
