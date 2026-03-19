#include "detection.hpp"

#include <algorithm>
#include <filesystem>
#include <fstream>

#include <spdlog/spdlog.h>
#include <nlohmann/json.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>

#include <io/debug.hpp>

#include "debugging.hpp"
#include "thresholds.hpp"

#include "debug.hpp"
#include "identification/board_circle/identification_circle.hpp"
#include "identification/identification.hpp"
#include "symmetries.hpp"

namespace
{

// Default debug path, overridden by output_path parameter in detect_and_identify_circlegrid()
std::string pth = "out/debug";

void append_tracking_stats_csv(int frame_id, int detected_count, int identified_count,
                               const std::string& method,
                               float homography_reproj_mean = -1.f, float homography_reproj_max = -1.f)
{
    const std::string dump_dir = []() -> std::string
    {
        const char* env = std::getenv("HEXRING_DEBUG_DIR");
        return env ? env : "/tmp";
    }();
    const std::string filepath = dump_dir + "/tracking_stats.csv";

    const bool file_exists = std::filesystem::exists(filepath);
    std::ofstream ofs(filepath, std::ios::app);
    if (!ofs.is_open())
    {
        return;
    }
    if (!file_exists)
    {
        ofs << "frame_id,detected_count,identified_count,method,homography_reproj_mean,homography_reproj_max\n";
    }
    ofs << frame_id << "," << detected_count << "," << identified_count << "," << method
        << "," << homography_reproj_mean << "," << homography_reproj_max << "\n";
}

cv::Mat1b binarize(cv::Mat1b &input, const marker::DetectionParameters &parameters)
{
    auto thresholds = thresholds::thresholds(input, parameters.row_tiles_count_, parameters.col_tiles_count_);
    for (auto &thresh : thresholds.thresholds_)
    {
        thresh = std::max(thresh, parameters.minimal_threshold_);
    }

    return thresholds::binarize(input, thresholds);
}

std::pair<float, float> min_max_in_rect(const cv::Mat1b &input, const int row_center, const int col_center,
                                        const int width, const int height)
{
    float min = std::numeric_limits<float>::max();
    float max = std::numeric_limits<float>::min();

    const int row_start = std::clamp(row_center - height, 0, input.rows);
    const int row_end = std::clamp(row_center + height, 0, input.rows);
    const int col_start = std::clamp(col_center - width, 0, input.cols);
    const int col_end = std::clamp(col_center - width, 0, input.cols);

    for (int row = row_start; row < row_end; ++row)
    {
        for (int col = col_start; col < col_end; ++col)
        {
            min = std::min(min, float(input(row, col)));
            max = std::max(max, float(input(row, col)));
        }
    }
    return {min, max};
}

std::vector<base::MarkerUnidentified> filter_as_objects_simple_descriptors_cores(
    cv::Mat1b &input, cv::Mat1b &binarized, const marker::DetectionParameters &parameters)
{
    cv::Mat1i labels, stats;
    cv::Mat1d centroid;
    cv::connectedComponentsWithStats(binarized, labels, stats, centroid);

    std::vector<base::MarkerUnidentified> markers;

    for (int object_idx = 0; object_idx < stats.rows; ++object_idx)
    {
        const int pixel_area = stats(object_idx, cv::CC_STAT_AREA);

        if (pixel_area > parameters.maximal_pixel_count_core_ || pixel_area < parameters.minimal_pixel_count_core_)
        {
            continue;
        }

        const int longest = std::max(stats(object_idx, cv::CC_STAT_HEIGHT), stats(object_idx, cv::CC_STAT_WIDTH));

        if (longest > parameters.maximal_edge_length_core_)
        {
            continue;
        }
        const int shorter = std::min(stats(object_idx, cv::CC_STAT_HEIGHT), stats(object_idx, cv::CC_STAT_WIDTH));

        // we compute biased estimate as for object of size 3-5 pixels single pixel of error can invalidly reject sample
        const float biased_obj_interia_ratio = float(shorter + 1) / float(longest + 1);
        if (parameters.interia_ratio_ > biased_obj_interia_ratio)
        {
            continue;
        }

        const auto type = marker::symetries::test_intensities_values_and_symmetries_cores(
            input, centroid(object_idx, 1), centroid(object_idx, 0), stats(object_idx, cv::CC_STAT_WIDTH) + 1,
            stats(object_idx, cv::CC_STAT_HEIGHT) + 1, parameters.min_difference_scale_);

        if (type == base::Difference::INNER_BRIGHTER)
        {
            const auto [min, max] =
                min_max_in_rect(input, centroid(object_idx, 1), centroid(object_idx, 0),
                                stats(object_idx, cv::CC_STAT_WIDTH), stats(object_idx, cv::CC_STAT_HEIGHT));
            markers.emplace_back(object_idx + 1, centroid(object_idx, 1), centroid(object_idx, 0),
                                 base::Type::INNER_CORE, stats(object_idx, cv::CC_STAT_WIDTH),
                                 stats(object_idx, cv::CC_STAT_HEIGHT), min, max);
        }
    }
    return markers;
}

std::vector<base::MarkerUnidentified> filter_as_objects_simple_descriptors_rings(
    cv::Mat1b &input, cv::Mat1b &binarized, const marker::DetectionParameters &parameters)
{
    cv::Mat1i labels, stats;
    cv::Mat1d centroid;
    cv::connectedComponentsWithStats(binarized, labels, stats, centroid);
    std::vector<base::MarkerUnidentified> markers;

    constexpr int kTooCloseToBoundary = 4;

    for (int object_idx = 0; object_idx < stats.rows; ++object_idx)
    {
        // check if our bounding box is too close to image boundary
        const int col_min = stats(object_idx, cv::CC_STAT_LEFT);
        if (col_min < kTooCloseToBoundary)
        {
            continue;
        }
        const int col_max = col_min + stats(object_idx, cv::CC_STAT_WIDTH);
        if (col_max > input.cols - kTooCloseToBoundary - 1)
        {
            continue;
        }
        const int row_min = stats(object_idx, cv::CC_STAT_TOP);
        if (row_min < kTooCloseToBoundary)
        {
            continue;
        }
        const int row_max = row_min + stats(object_idx, cv::CC_STAT_HEIGHT);
        if (row_max > input.rows - kTooCloseToBoundary - 1)
        {
            continue;
        }

        const int pixel_area = stats(object_idx, cv::CC_STAT_AREA);

        if (pixel_area > parameters.maximal_pixel_count_ring_ || pixel_area < parameters.minimal_pixel_count_ring_)
        {
            continue;
        }

        const int longest = std::max(stats(object_idx, cv::CC_STAT_HEIGHT), stats(object_idx, cv::CC_STAT_WIDTH));

        if (longest > parameters.maximal_edge_length_ring_)
        {
            continue;
        }
        const int shorter = std::min(stats(object_idx, cv::CC_STAT_HEIGHT), stats(object_idx, cv::CC_STAT_WIDTH));

        // we compute biased estimate as for object of size 3-5 pixels single pixel of error can invalidly reject sample
        const float biased_obj_interia_ratio = float(shorter + 1) / float(longest + 1);
        if (parameters.interia_ratio_ > biased_obj_interia_ratio)
        {
            continue;
        }

        const auto type = marker::symetries::test_intensities_values_and_symmetries_rings(
            input, centroid(object_idx, 1), centroid(object_idx, 0),
            stats(object_idx, cv::CC_STAT_WIDTH) * parameters.reduction_in_edge_length_ring_,
            stats(object_idx, cv::CC_STAT_HEIGHT) * parameters.reduction_in_edge_length_ring_,
            parameters.min_difference_scale_);

        switch (type)
        {
            case base::Difference::OUTER_BRIGHTER:
            {
                const auto [min, max] =
                    min_max_in_rect(input, centroid(object_idx, 1), centroid(object_idx, 0),
                                    stats(object_idx, cv::CC_STAT_WIDTH), stats(object_idx, cv::CC_STAT_HEIGHT));
                // possible coding marker
                markers.emplace_back(object_idx + 1, centroid(object_idx, 1), centroid(object_idx, 0),
                                     base::Type::CODING, stats(object_idx, cv::CC_STAT_WIDTH),
                                     stats(object_idx, cv::CC_STAT_HEIGHT), min, max);
                break;
            }
            case base::Difference::NO_DIFFERENCE:
            {
                const auto [min, max] =
                    min_max_in_rect(input, centroid(object_idx, 1), centroid(object_idx, 0),
                                    stats(object_idx, cv::CC_STAT_WIDTH), stats(object_idx, cv::CC_STAT_HEIGHT));
                markers.emplace_back(object_idx + 1, centroid(object_idx, 1), centroid(object_idx, 0), base::Type::RING,
                                     stats(object_idx, cv::CC_STAT_WIDTH), stats(object_idx, cv::CC_STAT_HEIGHT), min,
                                     max);
                break;
            }
            case base::Difference::INNER_BRIGHTER:
            default:
                break;
        }
    }
    return markers;
}

std::pair<std::vector<base::MarkerCoding>, std::vector<base::MarkerRing>> symetric_prune(
    std::vector<base::MarkerUnidentified> &inner, std::vector<base::MarkerUnidentified> &rings_and_unique)
{
    constexpr int kFreeConnection = -1;
    constexpr int kInvalidMultiConnected = -2;

    std::vector<int> inner_connected_to(inner.size(), kFreeConnection);
    std::vector<int> rings_connected_to(rings_and_unique.size(), kFreeConnection);

    std::vector<base::MarkerCoding> coding_markers;

    for (int ring_idx = 0; ring_idx < rings_and_unique.size(); ++ring_idx)
    {
        const auto &ring = rings_and_unique[ring_idx];
        if (rings_and_unique[ring_idx].type_ == base::Type::CODING)
        {
            coding_markers.emplace_back(ring.label_, ring.row_, ring.col_, ring.width_, ring.height_, ring.black_value_,
                                        ring.white_value_);
            continue;
        }
        const Eigen::Vector2f center_ring(ring.col_, ring.row_);
        const float trust_radius = std::min(0.3f * (ring.height_ + ring.width_) / 2.0f, 5.0f);

        bool was_connected = false;
        // TODO: provide some speed up of that (simple grid scatter would suffice)
        for (int inner_idx = 0; inner_idx < inner.size(); ++inner_idx)
        {
            const float distance = (Eigen::Vector2f(inner[inner_idx].col_, inner[inner_idx].row_) - center_ring).norm();
            if (distance < trust_radius)
            {
                if (was_connected)
                {
                    // prune my connection and myself
                    const int inner_connected = rings_connected_to[ring_idx];
                    inner_connected_to[inner_connected] = kFreeConnection;
                    rings_connected_to[ring_idx] = kInvalidMultiConnected;
                }
                else
                {
                    was_connected = true;
                    const int connection_type = inner_connected_to[inner_idx];
                    if (connection_type == kFreeConnection)
                    {
                        // first connection, all ok
                        inner_connected_to[inner_idx] = ring_idx;
                        rings_connected_to[ring_idx] = inner_idx;
                    }
                    else if (connection_type == -2)
                    {
                        // multiconnected, but already prunned
                        rings_connected_to[ring_idx] = kInvalidMultiConnected;
                    }
                    else
                    {
                        const int ring_connected_to = inner_connected_to[inner_idx];
                        // prune it
                        rings_connected_to[ring_connected_to] = kInvalidMultiConnected;
                        inner_connected_to[inner_idx] = kInvalidMultiConnected;
                        // prune myself
                        rings_connected_to[ring_idx] = kInvalidMultiConnected;
                    }
                }
            }
        }
    }

    std::vector<base::MarkerRing> ring_with_centers;

    for (int inner_idx = 0; inner_idx < inner_connected_to.size(); ++inner_idx)
    {
        const int ring_idx = inner_connected_to[inner_idx];
        if (ring_idx < 0)
        {
            continue;
        }

        ring_with_centers.emplace_back(inner[inner_idx], rings_and_unique[ring_idx]);
    }

    // TODO: prune coding to be in convex hull of rings

    return {coding_markers, ring_with_centers};
}

void remove_unused_markers(Eigen::Matrix<std::optional<int>, -1, -1> &ordering, std::vector<base::MarkerRing> &ring)
{
    std::vector<bool> used(ring.size(), false);

    for (int row = 0; row < ordering.rows(); ++row)
    {
        for (int col = 0; col < ordering.cols(); ++col)
        {
            if (ordering(row, col).has_value())
            {
                used[ordering(row, col).value()] = true;
            }
        }
    }

    // position is old idx, value is new idx
    std::vector<int> to_new_idx(ring.size(), -1);
    std::vector<base::MarkerRing> rings_shrink;

    for (size_t idx = 0; idx < used.size(); ++idx)
    {
        if (used[idx])
        {
            to_new_idx[idx] = rings_shrink.size();
            rings_shrink.emplace_back(ring[idx]);
        }
    }

    for (int row = 0; row < ordering.rows(); ++row)
    {
        for (int col = 0; col < ordering.cols(); ++col)
        {
            if (ordering(row, col).has_value())
            {
                ordering(row, col) = to_new_idx[ordering(row, col).value()];
            }
        }
    }
    ring = rings_shrink;
}

void assign_global_ids(const Eigen::Matrix<std::optional<int>, -1, -1> &ordering, std::vector<base::MarkerRing> &marker,
                       const std::unique_ptr<Board> &board)
{
    for (int row = 0; row < ordering.rows(); ++row)
    {
        for (int col = 0; col < ordering.cols(); ++col)
        {
            if (ordering(row, col).has_value())
            {
                marker[ordering(row, col).value()].global_id_ = board->row_and_col_to_id(row, col);
            }
        }
    }
}

cv::Mat1b create_marker_area(const std::vector<base::MarkerRing> &rings, const int rows, const int cols)
{
    cv::Mat1b marker_area = cv::Mat1b::zeros(rows, cols);

    for (const auto &marker : rings)
    {
        // TODO:
        // draw as elipses, cicrle is too crude. But without estimation of how elipse is fit, only crude approximation
        // could be get, as we do not currently use skewness of axis
        //
        // cv::circle(validity, cv::Point2f(marker.col_, marker.row_),
        //            std::max(marker.width_ring_, makrer.height_ring_) * 0.8, 255, -1);
        cv::ellipse(marker_area, cv::Point2f(marker.col_, marker.row_),
                    cv::Size(marker.width_ring_ / 2.0f * 1.1, marker.height_ring_ / 2.0f * 1.1), 0, 0, 360, 255, -1);
    }

    return marker_area;
}

cv::Mat1b create_calibrated_area(const std::vector<base::MarkerRing> &rings, const std::unique_ptr<Board> &board,
                                 const int rows, const int cols)
{
    // TODO:
    // find contres of spaned by markers, we should create it ourself as it's non convex in general, but for now we
    // stick to polyfinding
    cv::Mat1b calibrated_area = cv::Mat1b::zeros(rows, cols);
    cv::Mat1b board_markers = cv::Mat1b::zeros(board->rows_, board->cols_);

    std::map<int, int> id_to_marker;
    for (size_t marker = 0; marker < rings.size(); ++marker)
    {
        const int marker_id = rings[marker].global_id_;
        if (marker_id < 0)
        {
            // Skip unidentified markers
            continue;
        }
        const auto row_col = board->id_to_row_and_col(marker_id);
        board_markers(row_col(0), row_col(1)) = 255;

        id_to_marker[rings[marker].global_id_] = marker;
    }

    std::vector<std::vector<cv::Point2i>> contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(board_markers, contours, hierarchy, cv::RetrievalModes::RETR_TREE,
                     cv::ContourApproximationModes::CHAIN_APPROX_NONE);

    for (auto &contour : contours)
    {
        for (auto &point : contour)
        {
            const int marker_id = board->row_and_col_to_id(point.y, point.x);
            const auto &marker = rings[id_to_marker.at(marker_id)];

            point.x = marker.col_;
            point.y = marker.row_;
        }
    }

    int idx = 0;
    for (; idx >= 0; idx = hierarchy[idx][0])
    {
        cv::Scalar color(255, 255, 255);
        cv::drawContours(calibrated_area, contours, idx, color, cv::FILLED, 8, hierarchy);
    }
    return calibrated_area;
}

void save_markers(const std::filesystem::path &output_path, const int image_idx, const size_t total_expected_markers,
                  const int identified_markers, const std::vector<base::MarkerRing> &rings,
                  const BoardCircleGrid &board, const cv::Mat1b &input,
                  const Eigen::Matrix<std::optional<int>, -1, -1> &ordering)
{
    // Save JSON markers to identified-markers subdirectory
    const std::filesystem::path json_dir = output_path / "identified-markers";
    std::filesystem::create_directories(json_dir);
    nlohmann::json markers_json;
    markers_json["image_id"] = image_idx;
    markers_json["total_expected"] = total_expected_markers;
    markers_json["identified_count"] = identified_markers;
    markers_json["markers"] = nlohmann::json::array();

    for (size_t i = 0; i < rings.size(); ++i)
    {
        if (rings[i].global_id_ >= 0)
        {
            const auto rc = board.id_to_row_and_col(rings[i].global_id_);
            nlohmann::json marker_entry;
            marker_entry["global_id"] = rings[i].global_id_;
            marker_entry["row"] = rc(0);
            marker_entry["col"] = rc(1);
            marker_entry["pixel_row"] = rings[i].row_;
            marker_entry["pixel_col"] = rings[i].col_;
            markers_json["markers"].push_back(marker_entry);
        }
    }

    const std::filesystem::path json_path = json_dir / std::format("markers_{:06d}.json", image_idx);
    std::ofstream ofs(json_path);
    if (ofs.is_open())
    {
        ofs << markers_json.dump(2);
        spdlog::info("image {}: Saved identified markers to {}", image_idx, json_path.string());
    }
    else
    {
        spdlog::warn("image {}: Failed to save markers to {}", image_idx, json_path.string());
    }

    marker::debug::save_marker_identification(input, ordering, rings, image_idx, output_path);
}

}  // namespace

namespace marker
{
std::optional<base::ImageDecoding> detection::detect_and_identify(cv::Mat1b &input,
                                                                  const DetectionParameters &parameters,
                                                                  const std::unique_ptr<Board> &board,
                                                                  const int image_idx,
                                                                  const std::filesystem::path &output_path)
{
    spdlog::info("detecting markers in image {}", image_idx);

    std::vector<base::MarkerCoding> best_coding;
    std::vector<base::MarkerRing> best_rings;

    cv::Mat1b best_input = input.clone();
    cv::Mat1b best_binarized;
    cv::Mat1b best_inverted_binarization;
    std::vector<base::MarkerUnidentified> best_inner;
    std::vector<base::MarkerUnidentified> best_ring_and_coding;
    size_t best_num_rings = 0;
    size_t stagnation_count = 0;
    size_t best_brightness_scale_idx = 0;
    for (size_t brightness_scale_idx = 0; brightness_scale_idx < parameters.brightness_scales_.size();
         ++brightness_scale_idx)
    {
        if (stagnation_count > 2)
        {
            break;
        }
        cv::Mat1b input_copy;
        const float brightness_scale = parameters.brightness_scales_[brightness_scale_idx];
        if (brightness_scale != 1.0f)
        {
            input.convertTo(input_copy, input_copy.type(), brightness_scale);
        }
        else
        {
            input_copy = input.clone();
        }

        cv::Mat1b binarized = binarize(input_copy, parameters);

        auto inner = filter_as_objects_simple_descriptors_cores(input_copy, binarized, parameters);

        cv::Mat1b inverted_binarization;
        cv::bitwise_not(binarized, inverted_binarization);

        auto ring_and_coding =
            filter_as_objects_simple_descriptors_rings(input_copy, inverted_binarization, parameters);

        auto [coding, rings] = symetric_prune(inner, ring_and_coding);

        spdlog::info("image {}: brightness scale {}: found {} coding markers and {} rings", image_idx, brightness_scale,
                     coding.size(), rings.size());
        if (best_num_rings + 5 < rings.size())
        {
            best_coding = coding;
            best_rings = rings;
            best_num_rings = rings.size();
            best_input = input_copy.clone();
            best_binarized = binarized.clone();
            best_inner = inner;
            best_ring_and_coding = ring_and_coding;
            best_inverted_binarization = inverted_binarization.clone();
            best_brightness_scale_idx = brightness_scale_idx;
            stagnation_count = 0;
        }
        else
        {
            ++stagnation_count;
        }
    }
    input = best_input.clone();
    cv::Mat1b binarized = best_binarized.clone();
    cv::Mat1b inverted_binarization = best_inverted_binarization.clone();
    std::vector<base::MarkerCoding> coding = best_coding;
    std::vector<base::MarkerRing> rings = best_rings;

    if constexpr (kShowMarkers)
    {
        marker::debug::save_inner_markers_and_unique(input, best_inner, best_ring_and_coding, image_idx,
                                                     best_brightness_scale_idx);
    }
    if constexpr (kShowMarkers)
    {
        marker::debug::save_inner_markers_and_unique(input, coding, rings, image_idx, best_brightness_scale_idx);
    }

    if (rings.empty() || coding.empty())
    {
        spdlog::warn("image {}: no markers found", image_idx);
        return std::nullopt;
    }

    auto [decoding, neighbors] = identification::assign_global_IDs(coding, rings, board);

    if constexpr (kShowMarkers)
    {
        marker::debug::save_neighbors_edges(input, neighbors, coding, rings, image_idx);
    }

    if (!decoding)
    {
        return std::nullopt;
    }

    remove_unused_markers(decoding->ordering(), decoding->markers);
    assign_global_ids(decoding->ordering(), decoding->markers, board);

    if constexpr (kShowMarkers)
    {
        marker::debug::save_marker_identification(input, decoding->ordering(), decoding->markers, image_idx,
                                                  output_path);
    }

    const cv::Mat1b marker_area = create_marker_area(decoding->markers, input.rows, input.cols);
    const cv::Mat1b calibrated_area = create_calibrated_area(decoding->markers, board, input.rows, input.cols);
    if constexpr (kShowMarkers)
    {
        io::debug::save_image(marker_area, std::format("marker_area_{}", image_idx), debug::kMarkersSubdir);
        io::debug::save_image(calibrated_area, std::format("calibrated_area_{}", image_idx), debug::kMarkersSubdir);
    }

    return base::ImageDecoding(true, input, binarized, inverted_binarization, decoding->ordering(), decoding->markers,
                               marker_area, calibrated_area);
}

base::ImageDecoding detection::detect_and_identify_circlegrid(cv::Mat1b &input, const DetectionParameters &parameters,
                                                              const BoardCircleGrid &board,
                                                              identification::circlegrid::TrackingState &tracker_state,
                                                              const int image_idx,
                                                              const std::filesystem::path &output_path)
{
    spdlog::info("Detecting circle grid markers in image {}", image_idx);

    // Use output_path if provided, otherwise keep default
    if (!output_path.empty())
        pth = output_path.string();

    // Declared before lambda so it can be captured by reference
    std::vector<base::MarkerCoding> best_coding_markers;

    // Helper to create a failed result
    auto make_failed_result = [&](const cv::Mat1b &img, const cv::Mat1b &bin = cv::Mat1b(),
                                  const cv::Mat1b &inv_bin = cv::Mat1b(),
                                  const std::vector<base::MarkerRing> &markers = {})
    {
        Eigen::Matrix<std::optional<int>, -1, -1> empty_ordering =
            Eigen::Matrix<std::optional<int>, -1, -1>::Constant(board.rows_, board.cols_, std::nullopt);
        cv::Mat1b empty_area = cv::Mat1b::zeros(img.rows, img.cols);
        cv::Mat1b bin_to_use = bin.empty() ? empty_area : bin;
        cv::Mat1b inv_bin_to_use = inv_bin.empty() ? empty_area : inv_bin;
        return base::ImageDecoding(false, img, bin_to_use, inv_bin_to_use, empty_ordering, markers, empty_area,
                                   empty_area, best_coding_markers);
    };

    const size_t total_expected_markers = board.rows_ * board.cols_;

    // Best result tracking - we keep only the best one to avoid memory issues with cv::Mat
    std::vector<int> best_indices;
    cv::Mat1b best_input;
    cv::Mat1b best_binarized;
    cv::Mat1b best_inverted_binarization;
    float best_brightness_scale = 1.0f;
    bool best_find_circles_grid_succeeded = false;
    size_t best_marker_count = 0;

    for (size_t brightness_scale_idx = 0; brightness_scale_idx < parameters.brightness_scales_.size();
         ++brightness_scale_idx)
    {
        const float brightness_scale = parameters.brightness_scales_[brightness_scale_idx];

        cv::Mat1b input_scaled;
        if (brightness_scale != 1.0f)
        {
            input.convertTo(input_scaled, input_scaled.type(), brightness_scale);
        }
        else
        {
            input_scaled = input.clone();
        }

        const cv::Mat1b binarized_temp = binarize(input_scaled, parameters);

        cv::Mat1b inverted_binarization_temp;
        cv::bitwise_not(binarized_temp, inverted_binarization_temp);

        const auto ring_and_coding =
            filter_as_objects_simple_descriptors_rings(input_scaled, inverted_binarization_temp, parameters);

        std::vector<base::MarkerCoding> coding_markers_temp;
        for (const auto &m : ring_and_coding)
        {
            if (m.type_ == base::Type::CODING)
            {
                coding_markers_temp.emplace_back(m.label_, m.row_, m.col_, m.width_, m.height_, m.black_value_,
                                                 m.white_value_);
            }
        }

        // Test if this set of markers passes findCirclesGrid
        std::vector<int> indices_temp;
        const bool find_circles_grid_succeeded =
            coding_markers_temp.size() >= total_expected_markers &&
            identification::circlegrid::test_find_circles_grid(indices_temp, coding_markers_temp, board, tracker_state);

        spdlog::info("image {}: brightness scale {}: found {} coding markers, findCirclesGrid: {}", image_idx,
                      brightness_scale, coding_markers_temp.size(), find_circles_grid_succeeded ? "PASS" : "FAIL");

        // Decide if this result is better than the current best
        // Priority:
        // 1. Prefer results that pass findCirclesGrid (have >= expected markers)
        // 2. Among passing results, prefer exactly the expected count (no extra outliers)
        // 3. If same distance from expected, prefer more markers
        bool is_better = false;
        if (find_circles_grid_succeeded && !best_find_circles_grid_succeeded)
        {
            // This one passes findCirclesGrid, previous best didn't
            is_better = true;
        }
        else if (find_circles_grid_succeeded && best_find_circles_grid_succeeded)
        {
            // Both pass - prefer exactly the expected count, or closer to it
            const size_t expected = total_expected_markers;
            const size_t curr_diff = coding_markers_temp.size() >= expected ? coding_markers_temp.size() - expected
                                                                            : expected - coding_markers_temp.size();
            const size_t best_diff =
                best_marker_count >= expected ? best_marker_count - expected : expected - best_marker_count;

            if (curr_diff < best_diff)
            {
                is_better = true;
            }
            else if (curr_diff == best_diff && coding_markers_temp.size() > best_marker_count)
            {
                is_better = true;
            }
        }
        else if (!find_circles_grid_succeeded && !best_find_circles_grid_succeeded)
        {
            // Both fail - prefer more markers
            if (coding_markers_temp.size() > best_marker_count)
            {
                is_better = true;
            }
        }

        if (is_better)
        {
            best_coding_markers = std::move(coding_markers_temp);
            best_indices = indices_temp;
            best_input = input_scaled.clone();
            best_binarized = binarized_temp.clone();
            best_inverted_binarization = inverted_binarization_temp.clone();
            best_brightness_scale = brightness_scale;
            best_find_circles_grid_succeeded = find_circles_grid_succeeded;
            best_marker_count = best_coding_markers.size();
        }

        if (find_circles_grid_succeeded)
        {
            break;
        }
    }

    if (best_coding_markers.empty())
    {
        spdlog::warn("image {}: no coding markers found at any brightness scale", image_idx);
        return make_failed_result(input);
    }

    spdlog::info("image {}: selected brightness scale {} with {} markers (findCirclesGrid: {})", image_idx,
                  best_brightness_scale, best_coding_markers.size(),
                  best_find_circles_grid_succeeded ? "PASS" : "FAIL");

    input = best_input.clone();
    const cv::Mat1b binarized = best_binarized.clone();
    const cv::Mat1b inverted_binarization = best_inverted_binarization.clone();
    const std::vector<base::MarkerCoding> coding_markers = best_coding_markers;

    if (coding_markers.empty())
    {
        spdlog::warn("image {}: no coding markers (full black circles) found", image_idx);
        return make_failed_result(input, binarized, inverted_binarization);
    }

    const bool primary_succeeded = best_find_circles_grid_succeeded;
    std::string identification_method = primary_succeeded ? "findCirclesGrid" : "none";

    std::vector<int> global_ids = best_indices;

    // Orientation ambiguity is now resolved inside test_find_circles_grid via discrete optimization
    // (trying all 4 orientations and picking the one with lowest homography reprojection error).

    // Store findCirclesGrid positions as trusted reference for swap detection
    if (primary_succeeded)
    {
        const int total = board.rows_ * board.cols_;
        tracker_state.last_fcg_positions_.assign(total, cv::Point2f(-1, -1));
        for (size_t i = 0; i < coding_markers.size() && i < global_ids.size(); ++i)
        {
            if (global_ids[i] >= 0 && global_ids[i] < total)
                tracker_state.last_fcg_positions_[global_ids[i]] =
                    cv::Point2f(coding_markers[i].col_, coding_markers[i].row_);
        }
        tracker_state.last_fcg_frame_ = image_idx;
        spdlog::debug("image {}: stored findCirclesGrid reference positions", image_idx);
    }

    // Count how many markers findCirclesGrid actually identified
    const int primary_identified = primary_succeeded
                                       ? static_cast<int>(std::count_if(global_ids.begin(), global_ids.end(),
                                                                         [](int id) { return id >= 0; }))
                                       : 0;

    // Use Hungarian tracking when findCirclesGrid failed or identified less than 50% of markers.
    const bool primary_poor = !primary_succeeded || primary_identified < static_cast<int>(coding_markers.size()) / 2;
    const bool use_tracking = primary_poor && tracker_state.has_previous_;

    if (use_tracking)
    {
        spdlog::info("image {}: Using Hungarian tracking (primary_id={}, markers={}/{})", image_idx, primary_identified,
                      coding_markers.size(), total_expected_markers);

        auto tracking_result = identification::circlegrid::identify_with_hungarian_tracking(
            tracker_state, coding_markers, board, 80.0f, 5.0f, nullptr);

        // Detect tracker divergence: if Hungarian identifies very few markers
        // relative to detected blobs AND with high cost, the tracking state is
        // corrupted. Clear it so subsequent frames start fresh.
        const float identified_ratio = static_cast<float>(tracking_result.matched_count)
            / std::max(1.f, static_cast<float>(coding_markers.size()));
        if (tracking_result.matched_count <= primary_identified && identified_ratio < 0.25f)
        {
            spdlog::warn("image {}: tracker diverged (matched={}, ratio={:.2f}, avg_cost={:.1f}), resetting state",
                          image_idx, tracking_result.matched_count, identified_ratio, tracking_result.avg_cost);
            tracker_state.clear();
            // Use whatever primary identification we have (even if poor)
        }
        else if (tracking_result.matched_count > primary_identified)
        {
            // Resolve 180-degree ambiguity
            tracking_result.global_ids = identification::circlegrid::resolve_180_ambiguity(
                tracking_result.global_ids, coding_markers, tracker_state, board);

            global_ids = tracking_result.global_ids;
            const int identified_count =
                static_cast<int>(std::count_if(global_ids.begin(), global_ids.end(), [](int id) { return id >= 0; }));
            spdlog::info("image {}: Hungarian tracking identified {} markers (avg_cost={:.1f})", image_idx,
                          identified_count, tracking_result.avg_cost);
            identification_method = "hungarian";
        }
        else
        {
            // Hungarian didn't improve; fallback to KNN tracking
            spdlog::debug("image {}: Hungarian tracking didn't improve ({}<={}), trying KNN fallback", image_idx,
                          tracking_result.matched_count, primary_identified);
            const auto knn_result = identification::circlegrid::identify_with_tracking(
                tracker_state.prev_markers_, coding_markers, tracker_state.prev_global_ids_, 50.0f, 5.0f);
            if (knn_result.has_value())
            {
                const int knn_count = static_cast<int>(
                    std::count_if(knn_result->begin(), knn_result->end(), [](int id) { return id >= 0; }));
                if (knn_count > primary_identified)
                {
                    global_ids = *knn_result;
                    identification_method = "knn_fallback";
                    spdlog::info("image {}: KNN fallback identified {} markers", image_idx, knn_count);
                }
            }
        }
    }
    else if (primary_poor && !tracker_state.has_previous_)
    {
        spdlog::warn("image {}: No previous frame for tracking (primary_id={}, markers={}/{})", image_idx,
                     primary_identified, coding_markers.size(), total_expected_markers);
    }

    // Ensure global_ids is initialized before brute-force fallbacks
    if (global_ids.empty())
        global_ids.assign(coding_markers.size(), -1);

    // Last-resort: brute-force blob-to-board matching.
    // When FCG fails AND Hungarian/KNN fail, try two approaches:
    // A) If FCG reference exists: homography from reference to current blobs
    // B) If no reference (cold start): RANSAC with board coordinates directly
    {
        const int current_id_count = static_cast<int>(
            std::count_if(global_ids.begin(), global_ids.end(), [](int id) { return id >= 0; }));
        const int total_markers = board.rows_ * board.cols_;
        const bool has_fcg_ref = !tracker_state.last_fcg_positions_.empty()
            && static_cast<int>(tracker_state.last_fcg_positions_.size()) == total_markers;
        const bool need_brute_force = current_id_count < std::max(8, total_markers / 3)
            && static_cast<int>(coding_markers.size()) >= total_markers / 2;

        if (need_brute_force && has_fcg_ref)
        {
            // Collect current blob positions
            std::vector<cv::Point2f> blob_pts;
            blob_pts.reserve(coding_markers.size());
            for (const auto& m : coding_markers)
                blob_pts.emplace_back(m.col_, m.row_);

            // Collect valid FCG reference positions
            std::vector<cv::Point2f> ref_pts;
            std::vector<int> ref_gids;
            for (int gid = 0; gid < total_markers; ++gid)
            {
                const auto& p = tracker_state.last_fcg_positions_[gid];
                if (p.x != 0.f || p.y != 0.f)
                {
                    ref_pts.push_back(p);
                    ref_gids.push_back(gid);
                }
            }

            if (ref_pts.size() >= 8 && blob_pts.size() >= 8)
            {
                // Compute centroids
                cv::Point2f blob_centroid(0, 0), ref_centroid(0, 0);
                for (const auto& p : blob_pts) blob_centroid += p;
                for (const auto& p : ref_pts) ref_centroid += p;
                blob_centroid /= static_cast<float>(blob_pts.size());
                ref_centroid /= static_cast<float>(ref_pts.size());

                // Try to match via RANSAC homography from ref → blob positions.
                // Use any already-identified markers as seeds; if none, use nearest-neighbor
                // matching between centroids of ref and blob point clouds.
                std::vector<cv::Point2f> src_seed, dst_seed;

                // Use existing identifications as seeds
                for (size_t i = 0; i < coding_markers.size(); ++i)
                {
                    if (global_ids[i] >= 0 && global_ids[i] < total_markers)
                    {
                        const auto& ref_p = tracker_state.last_fcg_positions_[global_ids[i]];
                        if (ref_p.x != 0.f || ref_p.y != 0.f)
                        {
                            src_seed.push_back(ref_p);
                            dst_seed.emplace_back(coding_markers[i].col_, coding_markers[i].row_);
                        }
                    }
                }

                // If no seeds, use translation-based initialization:
                // shift = blob_centroid - ref_centroid, then match nearest neighbors
                if (src_seed.size() < 4)
                {
                    const cv::Point2f shift = blob_centroid - ref_centroid;
                    for (const auto& rp : ref_pts)
                    {
                        const cv::Point2f predicted = rp + shift;
                        float min_dist = 30.f;
                        int best_blob = -1;
                        for (size_t bi = 0; bi < blob_pts.size(); ++bi)
                        {
                            const float d = static_cast<float>(cv::norm(predicted - blob_pts[bi]));
                            if (d < min_dist) { min_dist = d; best_blob = static_cast<int>(bi); }
                        }
                        if (best_blob >= 0)
                        {
                            src_seed.push_back(rp);
                            dst_seed.push_back(blob_pts[best_blob]);
                        }
                    }
                }

                if (src_seed.size() >= 4)
                {
                    const cv::Mat H_bf = cv::findHomography(src_seed, dst_seed, cv::RANSAC, 10.0);
                    if (!H_bf.empty())
                    {
                        // Compute mean spacing for threshold
                        float mean_sp = 0.f;
                        int sp_count = 0;
                        for (size_t a = 1; a < dst_seed.size() && a < 20; ++a)
                            for (size_t b = 0; b < a; ++b)
                            {
                                const float d = static_cast<float>(cv::norm(dst_seed[a] - dst_seed[b]));
                                if (d < 200.f) { mean_sp += d; ++sp_count; }
                            }
                        mean_sp = sp_count > 0 ? mean_sp / static_cast<float>(sp_count) : 50.f;
                        const float match_threshold = mean_sp * 0.25f;

                        std::set<int> used_gids, used_blobs;
                        for (size_t i = 0; i < global_ids.size(); ++i)
                            if (global_ids[i] >= 0) { used_gids.insert(global_ids[i]); used_blobs.insert(static_cast<int>(i)); }

                        int bf_identified = 0;
                        for (size_t ri = 0; ri < ref_pts.size(); ++ri)
                        {
                            if (used_gids.count(ref_gids[ri])) continue;
                            const cv::Mat pt = (cv::Mat_<double>(3,1) << ref_pts[ri].x, ref_pts[ri].y, 1.0);
                            const cv::Mat proj = H_bf * pt;
                            const cv::Point2f projected(
                                static_cast<float>(proj.at<double>(0) / proj.at<double>(2)),
                                static_cast<float>(proj.at<double>(1) / proj.at<double>(2)));

                            float best_dist = match_threshold;
                            int best_blob = -1;
                            for (size_t bi = 0; bi < blob_pts.size(); ++bi)
                            {
                                if (used_blobs.count(static_cast<int>(bi))) continue;
                                const float d = static_cast<float>(cv::norm(projected - blob_pts[bi]));
                                if (d < best_dist) { best_dist = d; best_blob = static_cast<int>(bi); }
                            }
                            if (best_blob >= 0)
                            {
                                global_ids[best_blob] = ref_gids[ri];
                                used_gids.insert(ref_gids[ri]);
                                used_blobs.insert(best_blob);
                                ++bf_identified;
                            }
                        }
                        if (bf_identified > 0)
                        {
                            spdlog::info("image {}: brute-force blob matching identified {} markers "
                                         "(from {} ref pts, {} blobs, threshold={:.1f}px)",
                                         image_idx, bf_identified, ref_pts.size(), blob_pts.size(), match_threshold);
                            if (identification_method == "none" || identification_method.empty())
                                identification_method = "brute_force";
                            else
                                identification_method += "+brute_force";
                        }
                    }
                }
            }
        }
    }

    // Cold-start fallback: when no FCG reference exists yet and we have enough blobs,
    // try RANSAC matching from board coordinates to blob positions.
    // Use blob centroid + board centroid for translation init, then RANSAC.
    {
        const int current_id_count2 = static_cast<int>(
            std::count_if(global_ids.begin(), global_ids.end(), [](int id) { return id >= 0; }));
        const int total_markers = board.rows_ * board.cols_;
        if (current_id_count2 < std::max(8, total_markers / 3)
            && static_cast<int>(coding_markers.size()) >= total_markers * 3 / 4
            && tracker_state.last_fcg_positions_.empty())
        {
            // Compute board coordinates
            std::vector<cv::Point2f> board_coords;
            for (int r = 0; r < board.rows_; ++r)
                for (int c = 0; c < board.cols_; ++c)
                {
                    const float bx = board.is_asymetric_
                        ? static_cast<float>((2*c + r%2) * board.spacing_)
                        : static_cast<float>(c * board.spacing_);
                    const float by = static_cast<float>(r * board.spacing_);
                    board_coords.emplace_back(bx, by);
                }

            // Collect blob positions
            std::vector<cv::Point2f> blob_pts;
            for (const auto& m : coding_markers)
                blob_pts.emplace_back(m.col_, m.row_);

            // Try to find homography from board coords to blob positions
            // using identified markers as seeds, or nearest-neighbor matching if none
            std::vector<cv::Point2f> src_pts, dst_pts;
            for (size_t i = 0; i < global_ids.size(); ++i)
            {
                if (global_ids[i] >= 0 && global_ids[i] < total_markers)
                {
                    src_pts.push_back(board_coords[global_ids[i]]);
                    dst_pts.emplace_back(coding_markers[i].col_, coding_markers[i].row_);
                }
            }

            // If not enough seeds, try centroid-based matching
            if (src_pts.size() < 4)
            {
                cv::Point2f board_centroid(0, 0), blob_centroid(0, 0);
                for (const auto& p : board_coords) board_centroid += p;
                for (const auto& p : blob_pts) blob_centroid += p;
                board_centroid /= static_cast<float>(board_coords.size());
                blob_centroid /= static_cast<float>(blob_pts.size());

                // Estimate scale: ratio of point cloud spreads
                float board_spread = 0, blob_spread = 0;
                for (const auto& p : board_coords) board_spread += static_cast<float>(cv::norm(p - board_centroid));
                for (const auto& p : blob_pts) blob_spread += static_cast<float>(cv::norm(p - blob_centroid));
                const float scale = blob_spread / std::max(board_spread, 1.f);

                // Scale + translate board coords to blob space, then nearest-neighbor match
                src_pts.clear(); dst_pts.clear();
                for (const auto& bc : board_coords)
                {
                    const cv::Point2f predicted = blob_centroid + scale * (bc - board_centroid);
                    float min_dist = 25.f;
                    int best_blob = -1;
                    for (size_t bi = 0; bi < blob_pts.size(); ++bi)
                    {
                        const float d = static_cast<float>(cv::norm(predicted - blob_pts[bi]));
                        if (d < min_dist) { min_dist = d; best_blob = static_cast<int>(bi); }
                    }
                    if (best_blob >= 0)
                    {
                        src_pts.push_back(bc);
                        dst_pts.push_back(blob_pts[best_blob]);
                    }
                }
            }

            if (src_pts.size() >= 8)
            {
                const cv::Mat H_cs = cv::findHomography(src_pts, dst_pts, cv::RANSAC, 8.0);
                if (!H_cs.empty())
                {
                    float mean_sp = 0.f;
                    int sp_count = 0;
                    for (size_t a = 1; a < dst_pts.size() && a < 20; ++a)
                        for (size_t b = 0; b < a; ++b)
                        {
                            const float d = static_cast<float>(cv::norm(dst_pts[a] - dst_pts[b]));
                            if (d < 200.f) { mean_sp += d; ++sp_count; }
                        }
                    mean_sp = sp_count > 0 ? mean_sp / static_cast<float>(sp_count) : 50.f;

                    std::set<int> used_gids, used_blobs;
                    for (size_t i = 0; i < global_ids.size(); ++i)
                        if (global_ids[i] >= 0) { used_gids.insert(global_ids[i]); used_blobs.insert(static_cast<int>(i)); }

                    int cs_identified = 0;
                    for (int gid = 0; gid < total_markers; ++gid)
                    {
                        if (used_gids.count(gid)) continue;
                        const cv::Mat pt = (cv::Mat_<double>(3,1) << board_coords[gid].x, board_coords[gid].y, 1.0);
                        const cv::Mat proj = H_cs * pt;
                        const cv::Point2f projected(
                            static_cast<float>(proj.at<double>(0) / proj.at<double>(2)),
                            static_cast<float>(proj.at<double>(1) / proj.at<double>(2)));

                        float best_dist = mean_sp * 0.2f;
                        int best_blob = -1;
                        for (size_t bi = 0; bi < blob_pts.size(); ++bi)
                        {
                            if (used_blobs.count(static_cast<int>(bi))) continue;
                            const float d = static_cast<float>(cv::norm(projected - blob_pts[bi]));
                            if (d < best_dist) { best_dist = d; best_blob = static_cast<int>(bi); }
                        }
                        if (best_blob >= 0)
                        {
                            global_ids[best_blob] = gid;
                            used_gids.insert(gid);
                            used_blobs.insert(best_blob);
                            ++cs_identified;
                        }
                    }
                    if (cs_identified > 0)
                    {
                        spdlog::info("image {}: cold-start board matching identified {} markers",
                                      image_idx, cs_identified);
                        if (identification_method == "none" || identification_method.empty())
                            identification_method = "cold_start";
                        else
                            identification_method += "+cold_start";
                    }
                }
            }
        }
    }

    // Desperate last resort: if we still have <8 identified markers but ≥4,
    // try homography from board coords using just those 4+ markers as seeds.
    // Lower min_seeds requirement since we have no other option.
    {
        const int final_id_count = static_cast<int>(
            std::count_if(global_ids.begin(), global_ids.end(), [](int id) { return id >= 0; }));
        const int total_markers = board.rows_ * board.cols_;
        if (final_id_count >= 4 && final_id_count < 8)
        {
            // Build board→image correspondence from current identifications
            std::vector<cv::Point2f> board_seed, image_seed;
            std::set<int> used_gids;
            for (size_t i = 0; i < global_ids.size(); ++i)
            {
                if (global_ids[i] >= 0 && global_ids[i] < total_markers)
                {
                    const int r = global_ids[i] / board.cols_;
                    const int c = global_ids[i] % board.cols_;
                    const float bx = board.is_asymetric_
                        ? static_cast<float>((2*c + r%2) * board.spacing_)
                        : static_cast<float>(c * board.spacing_);
                    const float by = static_cast<float>(r * board.spacing_);
                    board_seed.emplace_back(bx, by);
                    image_seed.emplace_back(coding_markers[i].col_, coding_markers[i].row_);
                    used_gids.insert(global_ids[i]);
                }
            }

            if (board_seed.size() >= 4)
            {
                const cv::Mat H_last = cv::findHomography(board_seed, image_seed, 0);  // no RANSAC with few points
                if (!H_last.empty())
                {
                    float mean_sp = 0.f;
                    int sp_count = 0;
                    for (size_t a = 1; a < image_seed.size(); ++a)
                        for (size_t b = 0; b < a; ++b)
                        {
                            const float d = static_cast<float>(cv::norm(image_seed[a] - image_seed[b]));
                            if (d < 200.f) { mean_sp += d; ++sp_count; }
                        }
                    mean_sp = sp_count > 0 ? mean_sp / static_cast<float>(sp_count) : 50.f;

                    std::set<int> used_blobs;
                    for (size_t i = 0; i < global_ids.size(); ++i)
                        if (global_ids[i] >= 0) used_blobs.insert(static_cast<int>(i));

                    int last_resort = 0;
                    for (int gid = 0; gid < total_markers; ++gid)
                    {
                        if (used_gids.count(gid)) continue;
                        const int r = gid / board.cols_;
                        const int c = gid % board.cols_;
                        const float bx = board.is_asymetric_
                            ? static_cast<float>((2*c + r%2) * board.spacing_)
                            : static_cast<float>(c * board.spacing_);
                        const float by = static_cast<float>(r * board.spacing_);
                        const cv::Mat pt = (cv::Mat_<double>(3,1) << bx, by, 1.0);
                        const cv::Mat proj = H_last * pt;
                        const cv::Point2f projected(
                            static_cast<float>(proj.at<double>(0) / proj.at<double>(2)),
                            static_cast<float>(proj.at<double>(1) / proj.at<double>(2)));

                        float best_dist = mean_sp * 0.25f;
                        int best_blob = -1;
                        for (size_t bi = 0; bi < coding_markers.size(); ++bi)
                        {
                            if (used_blobs.count(static_cast<int>(bi))) continue;
                            const float d = static_cast<float>(cv::norm(
                                projected - cv::Point2f(coding_markers[bi].col_, coding_markers[bi].row_)));
                            if (d < best_dist) { best_dist = d; best_blob = static_cast<int>(bi); }
                        }
                        if (best_blob >= 0)
                        {
                            global_ids[best_blob] = gid;
                            used_gids.insert(gid);
                            used_blobs.insert(best_blob);
                            ++last_resort;
                        }
                    }
                    if (last_resort > 0)
                    {
                        spdlog::info("image {}: last-resort homography identified {} additional markers (from {} seeds)",
                                      image_idx, last_resort, board_seed.size());
                        identification_method += "+last_resort";
                    }
                }
            }
        }
    }

    if (global_ids.empty())
    {
        global_ids.assign(coding_markers.size(), -1);
    }

    if (global_ids.size() != coding_markers.size())
    {
        throw std::runtime_error(std::format("Not all markers has asigned indices! , global_ids {}, coding_markers {}",
                                             global_ids.size(), coding_markers.size()));
    }

    // For Hungarian/KNN frames: compare marker IDs against previous frame's IDs
    // using spatial matching. For consecutive frames, markers move <50px. If the
    // current assignment is 180° flipped, the matched prev IDs will be complementary.
    if (tracker_state.has_previous_ &&
        (identification_method.find("hungarian") != std::string::npos ||
         identification_method.find("knn") != std::string::npos))
    {
        const int total = board.rows_ * board.cols_;
        int matches_original = 0;
        int matches_flipped = 0;
        int compared = 0;

        for (size_t i = 0; i < coding_markers.size(); ++i)
        {
            if (global_ids[i] < 0) continue;
            const int flipped_id = total - 1 - global_ids[i];

            // Find nearest prev-frame marker by position
            float best_dist = 50.f;
            int best_prev_id = -1;
            for (size_t j = 0; j < tracker_state.prev_markers_.size(); ++j)
            {
                if (tracker_state.prev_global_ids_[j] < 0) continue;
                const float dx = coding_markers[i].col_ - tracker_state.prev_markers_[j].col_;
                const float dy = coding_markers[i].row_ - tracker_state.prev_markers_[j].row_;
                const float d = std::sqrt(dx * dx + dy * dy);
                if (d < best_dist)
                {
                    best_dist = d;
                    best_prev_id = tracker_state.prev_global_ids_[j];
                }
            }

            if (best_prev_id >= 0)
            {
                ++compared;
                if (global_ids[i] == best_prev_id) ++matches_original;
                if (flipped_id == best_prev_id) ++matches_flipped;
            }
        }

        if (compared >= 10 && matches_flipped > matches_original * 3)
        {
            spdlog::info("image {}: Hungarian 180° flip (orig={}, flip={}, cmp={}), correcting",
                         image_idx, matches_original, matches_flipped, compared);
            for (auto& gid : global_ids)
            {
                if (gid >= 0) gid = total - 1 - gid;
            }
        }
    }

    std::vector<base::MarkerRing> rings;
    rings.reserve(coding_markers.size());
    for (size_t i = 0; i < coding_markers.size(); ++i)
    {
        rings.emplace_back(coding_markers[i]);
        rings.back().global_id_ = global_ids[i];
    }

    // Populate debug prediction vectors for visualization.
    // For each identified marker that has a track, show where the blob velocity field
    // predicts it should be: query forward field at the PREVIOUS position (where the field
    // is anchored), then draw from the current position to the predicted position.
    tracker_state.debug_prediction_vectors_.clear();
    const bool field_valid = tracker_state.forward_blob_field_.valid;
    int vec_count = 0;
    if (field_valid)
    {
        for (const auto& ring : rings)
        {
            if (ring.global_id_ < 0) continue;
            const cv::Point2f curr_pos(ring.col_, ring.row_);

            // Find this marker's previous position from the track
            auto track_it = tracker_state.tracks_.find(ring.global_id_);
            if (track_it == tracker_state.tracks_.end()) continue;
            const cv::Point2f prev_pos = track_it->second.last_position;

            // Forward prediction: from previous position, where should the marker go?
            const cv::Point2f v = tracker_state.forward_blob_field_.transport_predict(prev_pos);
            const cv::Point2f predicted = prev_pos + v;

            // Draw line from current actual position to where it was predicted to be
            tracker_state.debug_prediction_vectors_.emplace_back(curr_pos, predicted);
            ++vec_count;
        }
    }
    spdlog::info("image {}: prediction vectors: field_valid={}, vectors={}", image_idx, field_valid, vec_count);

    // Per-marker filter decision tracking for CSV debug output
    struct MarkerFilterDecision {
        int disappeared_nbr_gid = -1;
        float disappeared_dist = -1.f;
        float disappeared_threshold = -1.f;
        bool disappeared_rejected = false;
        bool homography_added = false;
    };
    std::unordered_map<size_t, MarkerFilterDecision> filter_decisions;

    // Snapshot gids before homography for tracking which markers were added
    std::set<size_t> pre_homography_identified;
    for (size_t i = 0; i < rings.size(); ++i)
        if (rings[i].global_id_ >= 0) pre_homography_identified.insert(i);

    // Always try to identify unmatched markers using local homography
    const int unidentified_count =
        static_cast<int>(std::count_if(rings.begin(), rings.end(), [](const auto& r) { return r.global_id_ < 0; }));
    if (unidentified_count > 0)
    {
        const int pre_count = static_cast<int>(rings.size()) - unidentified_count;
        identification::circlegrid::identify_unmatched_by_local_homography(rings, board);
        const int post_count = static_cast<int>(
            std::count_if(rings.begin(), rings.end(), [](const auto& r) { return r.global_id_ >= 0; }));
        if (post_count > pre_count)
        {
            spdlog::info("image {}: Local homography identified {} additional markers", image_idx,
                          post_count - pre_count);
            if (identification_method == "none")
            {
                identification_method = "homography";
            }
            else
            {
                identification_method += "+homography";
            }
        }
    }

    // FCG-reference fallback: if we still have many unidentified blobs and have a
    // previous findCirclesGrid reference, compute homography from FCG positions to
    // current frame and match unidentified blobs directly.
    {
        const int current_identified = static_cast<int>(
            std::count_if(rings.begin(), rings.end(), [](const auto& r) { return r.global_id_ >= 0; }));
        const int current_unidentified = static_cast<int>(rings.size()) - current_identified;
        const int total_markers = board.rows_ * board.cols_;
        const bool need_fcg_fallback = current_identified < total_markers / 2
            && current_unidentified > 0
            && !tracker_state.last_fcg_positions_.empty()
            && static_cast<int>(tracker_state.last_fcg_positions_.size()) == total_markers;

        if (need_fcg_fallback)
        {
            // Use currently identified markers to compute homography from FCG reference to current
            std::vector<cv::Point2f> src_pts, dst_pts;
            for (const auto& ring : rings)
            {
                if (ring.global_id_ < 0 || ring.global_id_ >= total_markers) continue;
                const auto& ref_pos = tracker_state.last_fcg_positions_[ring.global_id_];
                if (ref_pos.x == 0.f && ref_pos.y == 0.f) continue;
                src_pts.push_back(ref_pos);
                dst_pts.emplace_back(ring.col_, ring.row_);
            }

            if (static_cast<int>(src_pts.size()) >= 4)
            {
                const cv::Mat H_fcg = cv::findHomography(src_pts, dst_pts, cv::RANSAC, 10.0);
                if (!H_fcg.empty())
                {
                    std::set<int> used_gids;
                    for (const auto& r : rings)
                        if (r.global_id_ >= 0) used_gids.insert(r.global_id_);

                    // Compute mean spacing for threshold
                    float mean_sp = 0.f;
                    int sp_count = 0;
                    for (size_t a = 1; a < dst_pts.size(); ++a)
                        for (size_t b = 0; b < a; ++b)
                        {
                            const float d = static_cast<float>(cv::norm(dst_pts[a] - dst_pts[b]));
                            if (d < 200.f) { mean_sp += d; ++sp_count; }
                        }
                    mean_sp = sp_count > 0 ? mean_sp / static_cast<float>(sp_count) : 50.f;

                    int fcg_identified = 0;
                    for (auto& ring : rings)
                    {
                        if (ring.global_id_ >= 0) continue;
                        const cv::Point2f img_pos(ring.col_, ring.row_);

                        // Tighter threshold than local homography (20% vs 30%) since
                        // the FCG reference may be from a distant frame
                        float best_dist = mean_sp * 0.2f;
                        int best_gid = -1;
                        for (int gid = 0; gid < total_markers; ++gid)
                        {
                            if (used_gids.count(gid)) continue;
                            const auto& ref_pos = tracker_state.last_fcg_positions_[gid];
                            if (ref_pos.x == 0.f && ref_pos.y == 0.f) continue;

                            const cv::Mat pt = (cv::Mat_<double>(3,1) << ref_pos.x, ref_pos.y, 1.0);
                            const cv::Mat proj = H_fcg * pt;
                            const cv::Point2f projected(
                                static_cast<float>(proj.at<double>(0) / proj.at<double>(2)),
                                static_cast<float>(proj.at<double>(1) / proj.at<double>(2)));
                            const float dist = static_cast<float>(cv::norm(img_pos - projected));
                            if (dist < best_dist)
                            {
                                best_dist = dist;
                                best_gid = gid;
                            }
                        }
                        if (best_gid >= 0)
                        {
                            ring.global_id_ = best_gid;
                            used_gids.insert(best_gid);
                            ++fcg_identified;
                        }
                    }
                    if (fcg_identified > 0)
                    {
                        spdlog::info("image {}: FCG-reference fallback identified {} additional markers",
                                      image_idx, fcg_identified);
                        if (identification_method == "none")
                            identification_method = "fcg_reference";
                        else
                            identification_method += "+fcg_reference";
                    }
                }
            }
        }
    }

    // Compute board aspect ratio for near-square guard (used by multiple filters below)
    const float board_width = board.is_asymetric_
        ? static_cast<float>((2 * (board.cols_ - 1) + 1) * board.spacing_)
        : static_cast<float>((board.cols_ - 1) * board.spacing_);
    const float board_height = static_cast<float>((board.rows_ - 1) * board.spacing_);
    const float aspect = std::max(board_width, board_height) / std::max(1.f, std::min(board_width, board_height));
    const bool is_near_square = aspect < 1.3f;

    // Mark homography-added markers and verify them against velocity field.
    // The homography step can re-introduce swapped markers that were correctly
    // rejected by the velocity acceptance check. Re-check newly-added markers.
    // Post-homography velocity check DISABLED: causes cascading tracker divergence
    // on fast-moving boards. The RANSAC outlier removal (below) provides equivalent protection.
    if (false && is_near_square && tracker_state.forward_blob_field_.valid && tracker_state.backward_blob_field_.valid)
    {
        const float img_short = static_cast<float>(std::min(input.cols, input.rows));
        const float abs_cap = 0.05f * img_short;
        int homography_vel_rejects = 0;
        for (size_t i = 0; i < rings.size(); ++i)
        {
            if (rings[i].global_id_ < 0) continue;
            if (pre_homography_identified.count(i) > 0) continue;  // not homography-added
            filter_decisions[i].homography_added = true;

            // Velocity check on homography-added marker
            auto track_it = tracker_state.tracks_.find(rings[i].global_id_);
            if (track_it == tracker_state.tracks_.end() || track_it->second.history_count < 1)
                continue;

            const cv::Point2f prev_pos = track_it->second.last_position;
            const cv::Point2f curr_pos(rings[i].col_, rings[i].row_);
            const cv::Point2f v_fwd = tracker_state.forward_blob_field_.transport_predict(prev_pos);
            const cv::Point2f predicted_curr = prev_pos + v_fwd;
            const float fwd_err = static_cast<float>(cv::norm(predicted_curr - curr_pos));
            const float disp = static_cast<float>(cv::norm(v_fwd));
            const float tol = std::min(std::max(0.5f * disp, 25.f), abs_cap);

            if (fwd_err > tol)
            {
                spdlog::info("image {}: post-homography velocity reject: gid {} fwd_err={:.1f} "
                             "(tol={:.1f} disp={:.1f}), unsetting",
                             image_idx, rings[i].global_id_, fwd_err, tol, disp);
                rings[i].global_id_ = -1;
                ++homography_vel_rejects;
            }
        }
        if (homography_vel_rejects > 0)
            spdlog::info("image {}: rejected {} homography-added markers via velocity check",
                         image_idx, homography_vel_rejects);
    }
    else
    {
        // Just mark homography-added markers without velocity check
        for (size_t i = 0; i < rings.size(); ++i)
            if (rings[i].global_id_ >= 0 && pre_homography_identified.count(i) == 0)
                filter_decisions[i].homography_added = true;
    }

    // Fix individual row swaps using the last findCirclesGrid frame as trusted reference.
    // Compute a homography from findCirclesGrid reference positions to current frame positions.
    // For each adjacent-row pair, check if the current assignment or swapped assignment
    // better matches the projected reference positions.
    if (!tracker_state.last_fcg_positions_.empty() && identification_method != "findCirclesGrid")
    {
        const int total = board.rows_ * board.cols_;
        const auto& ref = tracker_state.last_fcg_positions_;

        // Use the BOARD MODEL as the reference coordinate system.
        // Compute homography from board coordinates to current image positions.
        // The board coordinates are absolute (gid-independent), so even if some
        // markers have wrong gids, the RANSAC homography will fit the majority correctly.
        std::vector<cv::Point2f> board_pts_h, image_pts_h;
        for (const auto& ring : rings)
        {
            if (ring.global_id_ < 0 || ring.global_id_ >= total) continue;
            const int r = ring.global_id_ / board.cols_;
            const int c = ring.global_id_ % board.cols_;
            float bx = board.is_asymetric_ ? static_cast<float>((2*c + r%2) * board.spacing_)
                                            : static_cast<float>(c * board.spacing_);
            float by = static_cast<float>(r * board.spacing_);
            board_pts_h.emplace_back(bx, by);
            image_pts_h.emplace_back(ring.col_, ring.row_);
        }

        if (static_cast<int>(board_pts_h.size()) >= 8)
        {
            // RANSAC homography — robust to outliers (swapped markers)
            const cv::Mat H = cv::findHomography(board_pts_h, image_pts_h, cv::RANSAC, 5.0);
            if (!H.empty())
            {
                std::unordered_map<int, size_t> gid_to_ring;
                for (size_t i = 0; i < rings.size(); ++i)
                    if (rings[i].global_id_ >= 0)
                        gid_to_ring[rings[i].global_id_] = i;

                // Project a board position through the RANSAC homography
                auto project_board = [&](int r, int c) -> cv::Point2f {
                    float bx = board.is_asymetric_ ? static_cast<float>((2*c + r%2) * board.spacing_)
                                                    : static_cast<float>(c * board.spacing_);
                    float by = static_cast<float>(r * board.spacing_);
                    const cv::Mat pt = (cv::Mat_<double>(3,1) << bx, by, 1.0);
                    const cv::Mat proj = H * pt;
                    return {static_cast<float>(proj.at<double>(0) / proj.at<double>(2)),
                            static_cast<float>(proj.at<double>(1) / proj.at<double>(2))};
                };

                int swaps_fixed = 0;
                for (int r = 0; r < board.rows_ - 1; ++r)
                {
                    for (int c = 0; c < board.cols_; ++c)
                    {
                        const int gid_a = board.row_and_col_to_id(r, c);
                        const int gid_b = board.row_and_col_to_id(r + 1, c);
                        auto it_a = gid_to_ring.find(gid_a);
                        auto it_b = gid_to_ring.find(gid_b);
                        if (it_a == gid_to_ring.end() || it_b == gid_to_ring.end()) continue;

                        const cv::Point2f pos_a(rings[it_a->second].col_, rings[it_a->second].row_);
                        const cv::Point2f pos_b(rings[it_b->second].col_, rings[it_b->second].row_);

                        // Where should (r,c) and (r+1,c) be according to the RANSAC homography?
                        const cv::Point2f exp_a = project_board(r, c);
                        const cv::Point2f exp_b = project_board(r + 1, c);

                        const float cost_current = static_cast<float>(
                            cv::norm(pos_a - exp_a) + cv::norm(pos_b - exp_b));
                        const float cost_swapped = static_cast<float>(
                            cv::norm(pos_a - exp_b) + cv::norm(pos_b - exp_a));

                        if (cost_swapped < cost_current * 0.5f)
                        {
                            std::swap(rings[it_a->second].global_id_, rings[it_b->second].global_id_);
                            std::swap(gid_to_ring[gid_a], gid_to_ring[gid_b]);
                            ++swaps_fixed;
                            spdlog::debug("image {}: swap row {} col {} (gid {}<->{}), cost {:.1f}→{:.1f}",
                                          image_idx, r, c, gid_a, gid_b, cost_current, cost_swapped);
                        }
                    }
                }
                if (swaps_fixed > 0)
                {
                    spdlog::info("image {}: fixed {} adjacent-row swaps (fcg-reference)", image_idx, swaps_fixed);
                }
            }
        }
    }

    // Disappeared-neighbor swap detection: if gid X is assigned to a marker at
    // approximately the PREVIOUS position of gid Y (a hex neighbor),
    // and gid Y is NOT in the current frame, then gid X likely stole gid Y's marker.
    // Only run on near-square grids where row swaps actually occur.
    if (tracker_state.has_previous_ && identification_method != "findCirclesGrid" && is_near_square)
    {
        const int total = board.rows_ * board.cols_;

        // Build set of currently assigned gids
        std::set<int> assigned_gids;
        for (const auto& r : rings)
            if (r.global_id_ >= 0) assigned_gids.insert(r.global_id_);

        // Build map: gid → last known position (from tracks, not just prev frame)
        // This catches swaps even when the neighbor disappeared several frames ago
        std::unordered_map<int, cv::Point2f> prev_gid_pos;
        for (const auto& [gid, track] : tracker_state.tracks_)
        {
            // Use tracks within last 5 frames
            if (tracker_state.frame_counter_ - track.last_seen_frame <= 5)
                prev_gid_pos[gid] = track.last_position;
        }

        // Account for board motion: compute average displacement between prev and curr frames
        cv::Point2f avg_motion(0.f, 0.f);
        int motion_count = 0;
        for (const auto& ring : rings)
        {
            if (ring.global_id_ < 0) continue;
            auto pit = prev_gid_pos.find(ring.global_id_);
            if (pit != prev_gid_pos.end())
            {
                avg_motion += cv::Point2f(ring.col_, ring.row_) - pit->second;
                ++motion_count;
            }
        }
        if (motion_count > 0)
            avg_motion /= static_cast<float>(motion_count);

        // Threshold for motion-compensated distance. Scale with average velocity
        // to handle fast-moving boards (prediction error ∝ speed).
        const float avg_speed = tracker_state.forward_blob_field_.valid
            ? std::sqrt(tracker_state.forward_blob_field_.vCx * tracker_state.forward_blob_field_.vCx
                      + tracker_state.forward_blob_field_.vCy * tracker_state.forward_blob_field_.vCy)
            : static_cast<float>(cv::norm(avg_motion));
        const float swap_dist_threshold = std::max(20.f, 0.5f * avg_speed);  // min 20px, or 50% of speed
        int disappeared_swaps = 0;

        // Iterate until no more cascading swaps found.
        // Each pass may unset a marker, making its neighbor "missing" for the next pass.
        // E.g.: M0 missing → M1 at M0's spot (unset) → M6 at M1's spot (unset next pass).
        for (int pass = 0; pass < 5; ++pass)  // max 5 cascade levels
        {
            int pass_swaps = 0;
            for (auto& ring : rings)
            {
                if (ring.global_id_ < 0) continue;
                const int gid = ring.global_id_;
                const int row = gid / board.cols_;
                const int col = gid % board.cols_;
                const cv::Point2f pos(ring.col_, ring.row_);

                // Check all physically adjacent neighbors on the asymmetric hex grid.
                // In an asymmetric grid, physical layout is:
                //   x = (2*col + row%2) * spacing,  y = row * spacing
                // So each marker has 6 hex neighbors:
                //   same row: col±1
                //   row-1: col+0 and col-1+row%2  (depends on row parity)
                //   row+1: col+0 and col-1+row%2
                // For non-asymmetric grids, just use 4-connected.
                struct Neighbor { int row; int col; };
                std::vector<Neighbor> neighbors;
                neighbors.reserve(6);
                // Same-row neighbors
                neighbors.push_back({row, col - 1});
                neighbors.push_back({row, col + 1});
                // Adjacent-row neighbors (hex connectivity)
                if (board.is_asymetric_)
                {
                    const int off = (row % 2 == 0) ? -1 : 0;  // hex offset
                    neighbors.push_back({row - 1, col + off});
                    neighbors.push_back({row - 1, col + off + 1});
                    neighbors.push_back({row + 1, col + off});
                    neighbors.push_back({row + 1, col + off + 1});
                }
                else
                {
                    neighbors.push_back({row - 1, col});
                    neighbors.push_back({row + 1, col});
                }

                for (const auto& [nbr_row, nbr_col] : neighbors)
                {
                    if (nbr_row < 0 || nbr_row >= board.rows_) continue;
                    if (nbr_col < 0 || nbr_col >= board.cols_) continue;
                    const int nbr_gid = board.row_and_col_to_id(nbr_row, nbr_col);

                    // Is the neighbor MISSING in current frame but WAS seen recently?
                    if (assigned_gids.count(nbr_gid) == 0 && prev_gid_pos.count(nbr_gid) > 0)
                    {
                        // Is the current marker at the MOTION-COMPENSATED position
                        // of the missing neighbor? Use velocity field if available.
                        const auto& nbr_track = tracker_state.tracks_.at(nbr_gid);
                        const int frames_since = tracker_state.frame_counter_ - nbr_track.last_seen_frame;
                        cv::Point2f expected_nbr_pos;
                        if (tracker_state.forward_blob_field_.valid)
                        {
                            const cv::Point2f pred = tracker_state.forward_blob_field_.transport_predict(
                                prev_gid_pos[nbr_gid], static_cast<float>(std::max(1, frames_since)));
                            expected_nbr_pos = prev_gid_pos[nbr_gid] + pred;
                        }
                        else
                        {
                            const cv::Point2f scaled_motion = avg_motion * static_cast<float>(std::max(1, frames_since));
                            expected_nbr_pos = prev_gid_pos[nbr_gid] + scaled_motion;
                        }
                        const float dist_to_expected = static_cast<float>(cv::norm(pos - expected_nbr_pos));
                        // Record decision for CSV
                        const size_t ring_idx = static_cast<size_t>(&ring - &rings[0]);
                        if (dist_to_expected < swap_dist_threshold)
                        {
                            // Don't remove if it would drop below 8 identified markers
                            if (static_cast<int>(assigned_gids.size()) <= 8)
                            {
                                spdlog::debug("image {}: skipping disappeared-neighbor swap gid {} "
                                              "(would drop below 8 markers)", image_idx, gid);
                                continue;
                            }
                            filter_decisions[ring_idx] = {nbr_gid, dist_to_expected,
                                                           swap_dist_threshold, true, false};
                            spdlog::info("image {}: disappeared-neighbor swap (pass {}): gid {} at prev gid {} pos "
                                         "(dist={:.1f}px, motion-comp), unsetting",
                                         image_idx, pass, gid, nbr_gid, dist_to_expected);
                            ring.global_id_ = -1;
                            assigned_gids.erase(gid);
                            ++pass_swaps;
                            break;
                        }
                    }
                }
            }
            disappeared_swaps += pass_swaps;
            if (pass_swaps == 0) break;  // no more cascading swaps
        }
        if (disappeared_swaps > 0)
        {
            spdlog::info("image {}: fixed {} disappeared-neighbor swaps", image_idx, disappeared_swaps);
        }
    }

    // Per-frame homography reprojection quality: compute RANSAC homography from
    // board coordinates to final identified image positions, then measure per-marker error.
    std::unordered_map<size_t, float> homography_reproj_error;
    float frame_reproj_mean = -1.f, frame_reproj_max = -1.f;
    {
        std::vector<cv::Point2f> board_pts_q, image_pts_q;
        std::vector<size_t> ring_indices_q;
        const int total = board.rows_ * board.cols_;
        for (size_t i = 0; i < rings.size(); ++i)
        {
            if (rings[i].global_id_ < 0 || rings[i].global_id_ >= total) continue;
            const int r = rings[i].global_id_ / board.cols_;
            const int c = rings[i].global_id_ % board.cols_;
            const float bx = board.is_asymetric_ ? static_cast<float>((2*c + r%2) * board.spacing_)
                                                   : static_cast<float>(c * board.spacing_);
            const float by = static_cast<float>(r * board.spacing_);
            board_pts_q.emplace_back(bx, by);
            image_pts_q.emplace_back(rings[i].col_, rings[i].row_);
            ring_indices_q.push_back(i);
        }

        if (board_pts_q.size() >= 8)
        {
            cv::Mat H_q = cv::findHomography(board_pts_q, image_pts_q, cv::RANSAC, 5.0);
            if (!H_q.empty())
            {
                // Iterative self-correcting outlier removal:
                // 1. Compute per-marker reproj against RANSAC H
                // 2. Find the worst outlier (> 3×median and > 2×P90)
                // 3. Remove it and recompute H
                // 4. Repeat until stable or all markers within threshold
                // This converges because removing one bad marker improves H for all others.
                int outliers_removed = 0;
                const bool allow_removal = (identification_method != "findCirclesGrid");

                for (int iter = 0; iter < 10 && allow_removal; ++iter)
                {
                    // Count current identified markers
                    int current_count = 0;
                    for (size_t j = 0; j < ring_indices_q.size(); ++j)
                        if (rings[ring_indices_q[j]].global_id_ >= 0) ++current_count;
                    if (current_count <= 8) break;

                    // Recompute H from current inliers
                    std::vector<cv::Point2f> curr_board, curr_image;
                    std::vector<size_t> curr_idx;
                    for (size_t j = 0; j < ring_indices_q.size(); ++j)
                    {
                        if (rings[ring_indices_q[j]].global_id_ < 0) continue;
                        curr_board.push_back(board_pts_q[j]);
                        curr_image.push_back(image_pts_q[j]);
                        curr_idx.push_back(j);
                    }
                    if (curr_board.size() < 8) break;

                    H_q = cv::findHomography(curr_board, curr_image, cv::RANSAC, 5.0);
                    if (H_q.empty()) break;

                    // Compute reproj errors
                    float worst_err = 0.f;
                    size_t worst_idx = 0;
                    std::vector<float> errors(curr_board.size());
                    for (size_t k = 0; k < curr_board.size(); ++k)
                    {
                        const cv::Mat pt = (cv::Mat_<double>(3,1) << curr_board[k].x, curr_board[k].y, 1.0);
                        const cv::Mat proj = H_q * pt;
                        const cv::Point2f projected(static_cast<float>(proj.at<double>(0) / proj.at<double>(2)),
                                                     static_cast<float>(proj.at<double>(1) / proj.at<double>(2)));
                        errors[k] = static_cast<float>(cv::norm(projected - curr_image[k]));
                        homography_reproj_error[ring_indices_q[curr_idx[k]]] = errors[k];
                        if (errors[k] > worst_err) { worst_err = errors[k]; worst_idx = k; }
                    }

                    // Compute median
                    std::vector<float> sorted_e = errors;
                    std::sort(sorted_e.begin(), sorted_e.end());
                    const float median_e = sorted_e[sorted_e.size() / 2];
                    const float p90_e = sorted_e[static_cast<size_t>(sorted_e.size() * 0.9)];
                    const float thresh = std::max(3.f * median_e, 2.f * p90_e);

                    // Remove worst outlier if it exceeds threshold
                    if (worst_err > thresh && current_count > 8)
                    {
                        const size_t orig_j = curr_idx[worst_idx];
                        spdlog::debug("image {}: iter {} reproj outlier: gid {} reproj={:.1f}px > {:.1f}px (median={:.1f})",
                                       image_idx, iter, rings[ring_indices_q[orig_j]].global_id_,
                                       worst_err, thresh, median_e);
                        rings[ring_indices_q[orig_j]].global_id_ = -1;
                        ++outliers_removed;
                    }
                    else
                    {
                        break;  // All within threshold — converged
                    }
                }

                // Compute final frame stats
                float sum_err = 0.f, max_err = 0.f;
                int valid_count = 0;
                for (size_t j = 0; j < ring_indices_q.size(); ++j)
                {
                    if (rings[ring_indices_q[j]].global_id_ < 0) continue;
                    const auto it = homography_reproj_error.find(ring_indices_q[j]);
                    const float e = it != homography_reproj_error.end() ? it->second : 0.f;
                    sum_err += e;
                    max_err = std::max(max_err, e);
                    ++valid_count;
                }
                frame_reproj_mean = valid_count > 0 ? sum_err / static_cast<float>(valid_count) : -1.f;
                frame_reproj_max = max_err;

                if (outliers_removed > 0)
                {
                    spdlog::info("image {}: removed {} reproj outliers (iterative), "
                                 "remaining: mean={:.2f}px max={:.2f}px ({} markers)",
                                 image_idx, outliers_removed,
                                 frame_reproj_mean, frame_reproj_max, valid_count);
                }

                // Aggressive cleanup: if reproj is still very high after iterative removal,
                // keep only RANSAC inlier markers (those consistent with the homography).
                // This prevents misidentified markers from corrupting the IMU-cam calibration.
                if (frame_reproj_mean > 20.f && valid_count > 8 && allow_removal)
                {
                    // Recompute H one more time and keep only tight inliers
                    std::vector<cv::Point2f> final_board, final_image;
                    std::vector<size_t> final_idx;
                    for (size_t j = 0; j < ring_indices_q.size(); ++j)
                    {
                        if (rings[ring_indices_q[j]].global_id_ < 0) continue;
                        final_board.push_back(board_pts_q[j]);
                        final_image.push_back(image_pts_q[j]);
                        final_idx.push_back(j);
                    }
                    if (final_board.size() >= 8)
                    {
                        std::vector<uchar> inlier_mask;
                        cv::findHomography(final_board, final_image, cv::RANSAC, 8.0, inlier_mask);
                        int aggressive_removed = 0;
                        int remaining = static_cast<int>(final_board.size());
                        for (size_t k = 0; k < final_idx.size(); ++k)
                        {
                            if (!inlier_mask[k] && remaining > 8)
                            {
                                rings[ring_indices_q[final_idx[k]]].global_id_ = -1;
                                ++aggressive_removed;
                                --remaining;
                            }
                        }
                        if (aggressive_removed > 0)
                            spdlog::info("image {}: aggressive RANSAC cleanup removed {} more markers ({} remaining)",
                                          image_idx, aggressive_removed, remaining);
                    }
                }
                else if (frame_reproj_mean > 5.0f)
                {
                    spdlog::warn("image {}: HIGH homography reprojection error: mean={:.2f}px max={:.2f}px "
                                 "({} markers)", image_idx, frame_reproj_mean, frame_reproj_max, valid_count);
                }
            }
        }
    }

    std::vector<int> current_ids;
    current_ids.reserve(rings.size());
    for (const auto &r : rings)
    {
        current_ids.push_back(r.global_id_);
    }
    tracker_state.last_method_ = identification_method;

    // Write per-frame debug CSV with per-marker filter decisions.
    // One CSV per frame: <debug_dir>/filter-csv/frame_NNNNNN.csv
    {
        static const std::string csv_dir = [&]() {
            const std::string d = pth + "/filter-csv";
            std::filesystem::create_directories(d);
            return d;
        }();
        const std::string csv_path = std::format("{}/frame_{:06d}.csv", csv_dir, image_idx);
        std::ofstream csv(csv_path);
        if (csv.is_open())
        {
            // Header
            csv << "marker_idx,pixel_x,pixel_y,final_gid,method,"
                   "vel_field_valid,vel_vCx,vel_vCy,vel_omega,vel_sigma,vel_rms,vel_inliers,vel_total,"
                   "vel_fwd_err,vel_bwd_err,vel_tolerance,vel_rejected,"
                   "disappeared_nbr_gid,disappeared_dist,disappeared_threshold,disappeared_rejected,"
                   "homography_added,fcg_swap_fixed,"
                   "track_prev_x,track_prev_y,predicted_x,predicted_y,"
                   "homography_reproj,frame_reproj_mean,frame_reproj_max\n";

            // Velocity field info (same for all markers in this frame)
            const auto& fwd = tracker_state.forward_blob_field_;
            const bool fv = fwd.valid;

            for (size_t i = 0; i < rings.size(); ++i)
            {
                const auto& ring = rings[i];
                csv << i << ','
                    << ring.col_ << ',' << ring.row_ << ','
                    << ring.global_id_ << ','
                    << identification_method << ',';

                // Velocity field model parameters
                csv << (fv ? 1 : 0) << ','
                    << (fv ? fwd.vCx : 0.f) << ','
                    << (fv ? fwd.vCy : 0.f) << ','
                    << (fv ? fwd.omega : 0.f) << ','
                    << (fv ? fwd.sigma : 0.f) << ','
                    << (fv ? fwd.velocity_residual_rms : 0.f) << ','
                    << (fv ? fwd.velocity_inlier_count : 0) << ','
                    << (fv ? static_cast<int>(fwd.positions.size()) : 0) << ',';

                // Per-marker velocity check (compute live — same logic as acceptance check)
                float fwd_err = -1.f, bwd_err = -1.f, tol = -1.f;
                bool vel_rejected = false;
                float track_prev_x = -1.f, track_prev_y = -1.f;
                float predicted_x = -1.f, predicted_y = -1.f;
                if (ring.global_id_ >= 0 && fv && tracker_state.backward_blob_field_.valid)
                {
                    auto track_it = tracker_state.tracks_.find(ring.global_id_);
                    if (track_it != tracker_state.tracks_.end() && track_it->second.history_count >= 1)
                    {
                        const cv::Point2f prev_pos = track_it->second.last_position;
                        const cv::Point2f curr_pos(ring.col_, ring.row_);
                        const cv::Point2f v_fwd = fwd.transport_predict(prev_pos);
                        const cv::Point2f pred_curr = prev_pos + v_fwd;
                        const cv::Point2f v_bwd = tracker_state.backward_blob_field_.transport_predict(curr_pos);
                        const cv::Point2f pred_prev = curr_pos - v_bwd;
                        fwd_err = static_cast<float>(cv::norm(pred_curr - curr_pos));
                        bwd_err = static_cast<float>(cv::norm(pred_prev - prev_pos));
                        const float disp = static_cast<float>(cv::norm(v_fwd));
                        const float img_short = static_cast<float>(
                            std::min(input.cols, input.rows));
                        tol = std::min(std::max(0.5f * disp, 25.f), 0.05f * img_short);
                        vel_rejected = (fwd_err > tol || bwd_err > tol);
                        track_prev_x = prev_pos.x;
                        track_prev_y = prev_pos.y;
                        predicted_x = pred_curr.x;
                        predicted_y = pred_curr.y;
                    }
                }
                csv << fwd_err << ',' << bwd_err << ',' << tol << ','
                    << (vel_rejected ? 1 : 0) << ',';

                // Disappeared-neighbor decision
                const auto fd_it = filter_decisions.find(i);
                if (fd_it != filter_decisions.end())
                {
                    csv << fd_it->second.disappeared_nbr_gid << ','
                        << fd_it->second.disappeared_dist << ','
                        << fd_it->second.disappeared_threshold << ','
                        << (fd_it->second.disappeared_rejected ? 1 : 0) << ','
                        << (fd_it->second.homography_added ? 1 : 0) << ',';
                }
                else
                {
                    csv << -1 << ',' << -1.f << ',' << -1.f << ',' << 0 << ','
                        << 0 << ',';
                }
                // fcg swap (not individually tracked yet)
                csv << 0 << ',';

                // Track history
                csv << track_prev_x << ',' << track_prev_y << ','
                    << predicted_x << ',' << predicted_y << ',';

                // Homography reprojection quality
                auto reproj_it = homography_reproj_error.find(i);
                csv << (reproj_it != homography_reproj_error.end() ? reproj_it->second : -1.f) << ','
                    << frame_reproj_mean << ',' << frame_reproj_max << '\n';
            }
        }
    }

    tracker_state.update(coding_markers, current_ids, input);

    Eigen::Matrix<std::optional<int>, -1, -1> ordering =
        Eigen::Matrix<std::optional<int>, -1, -1>::Constant(board.rows_, board.cols_, std::nullopt);

    int identified_markers = 0;
    for (size_t i = 0; i < rings.size(); ++i)
    {
        if (rings[i].global_id_ >= 0)
        {
            const auto rc = board.id_to_row_and_col(rings[i].global_id_);
            ordering(rc(0), rc(1)) = static_cast<int>(i);
            ++identified_markers;
        }
    }

    if (identified_markers == 0)
    {
        spdlog::warn("image {}: No circle grid markers identified", image_idx);
        append_tracking_stats_csv(image_idx, static_cast<int>(coding_markers.size()), 0, "failed");
        return make_failed_result(input, binarized, inverted_binarization, rings);
    }

    // Quality monitoring: log frames with high homography reprojection error
    // but do NOT reject — the sliding-window re-identification will fix these.
    if (frame_reproj_mean > 5.0f && frame_reproj_mean > 0.f)
    {
        spdlog::warn("image {}: HIGH homography reproj mean={:.2f}px (max={:.2f}px, {} markers)",
                     image_idx, frame_reproj_mean, frame_reproj_max, identified_markers);
    }

    spdlog::info("image {}: Final identification: {} / {} markers (method: {})", image_idx, identified_markers,
                  total_expected_markers, identification_method);

    append_tracking_stats_csv(image_idx, static_cast<int>(coding_markers.size()), identified_markers,
                              identification_method, frame_reproj_mean, frame_reproj_max);

    const cv::Mat1b marker_area = create_marker_area(rings, input.rows, input.cols);
    const cv::Mat1b calibrated_area =
        create_calibrated_area(rings, std::make_unique<BoardCircleGrid>(board), input.rows, input.cols);

    if constexpr (kShowMarkers)
    {
        // if (!output_path.empty())
        // {
        save_markers(pth + "/markers-json/", 99999, total_expected_markers, identified_markers, rings, board, input,
                     ordering);
        save_markers(pth + "/markers-json/", image_idx, total_expected_markers, identified_markers, rings, board, input,
                     ordering);

        io::debug::save_image(marker_area, std::format("marker_area_circle_{}", image_idx), "markers-png", pth);
        io::debug::save_image(calibrated_area, std::format("calibrated_area_circle_{}", image_idx), "markers-png", pth);
        //   static_assert(false);
        // }
    }

    return base::ImageDecoding(true, input, binarized, inverted_binarization, ordering, rings, marker_area,
                               calibrated_area, coding_markers);
}

}  // namespace marker
