#include "detection.hpp"

#include <algorithm>
#include <filesystem>
#include <fstream>

#include <spdlog/spdlog.h>
#include <nlohmann/json.hpp>
#include <opencv2/imgproc.hpp>

#include <io/debug.hpp>

#include "debugging.hpp"
#include "thresholds.hpp"

#include "debug.hpp"
#include "identification/board_circle/identification_circle.hpp"
#include "identification/identification.hpp"
#include "symmetries.hpp"

namespace
{

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
        spdlog::debug("image {}: Saved identified markers to {}", image_idx, json_path.string());
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
                                                                  const int image_idx)
{
    spdlog::debug("detecting markers in image {}", image_idx);

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
        marker::debug::save_marker_identification(input, decoding->ordering(), decoding->markers, image_idx);
    }

    const cv::Mat1b marker_area = create_marker_area(decoding->markers, input.rows, input.cols);
    const cv::Mat1b calibrated_area = create_calibrated_area(decoding->markers, board, input.rows, input.cols);
    if constexpr (kShowMarkers)
    {
        io::debug::save_image(marker_area, std::format("marker_area_{}", image_idx), debug::kMarkersSubdir);
        io::debug::save_image(calibrated_area, std::format("calibrated_area_{}", image_idx), debug::kMarkersSubdir);
    }

    return base::ImageDecoding(input, binarized, inverted_binarization, decoding->ordering(), decoding->markers,
                               marker_area, calibrated_area);
}

std::optional<base::ImageDecoding> detection::detect_and_identify_circlegrid(
    cv::Mat1b &input, const DetectionParameters &parameters, const BoardCircleGrid &board,
    identification::circlegrid::TrackingState &tracker_state, const int image_idx,
    const std::filesystem::path &output_path)
{
    spdlog::debug("Detecting circle grid markers in image {}", image_idx);

    const size_t total_expected_markers = board.rows_ * board.cols_;

    // Best result tracking - we keep only the best one to avoid memory issues with cv::Mat
    std::vector<base::MarkerCoding> best_coding_markers;
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
            identification::circlegrid::test_find_circles_grid(indices_temp, coding_markers_temp, board);

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
    }

    if (best_coding_markers.empty())
    {
        spdlog::warn("image {}: no coding markers found at any brightness scale", image_idx);
        return std::nullopt;
    }

    spdlog::info("image {}: selected brightness scale {} with {} markers (findCirclesGrid: {})", image_idx,
                 best_brightness_scale, best_coding_markers.size(), best_find_circles_grid_succeeded ? "PASS" : "FAIL");

    input = best_input.clone();
    const cv::Mat1b binarized = best_binarized.clone();
    const cv::Mat1b inverted_binarization = best_inverted_binarization.clone();
    const std::vector<base::MarkerCoding> coding_markers = best_coding_markers;

    if (coding_markers.empty())
    {
        spdlog::warn("image {}: no coding markers (full black circles) found", image_idx);
        return std::nullopt;
    }

    const bool primary_succeeded = best_find_circles_grid_succeeded;

    std::vector<int> global_ids = best_indices;

    // Tracking is mandatory ONLY when less than total_expected_markers were detected
    // (board has 35 markers, so partial visibility requires tracking)
    // When we have >= 35 markers, the geometric identification should succeed
    const bool tracking_mandatory = (coding_markers.size() < total_expected_markers);

    // Use tracking only if mandatory (fewer markers than expected) OR if primary identification failed
    // but we have a previous frame and enough markers
    const bool use_tracking =
        tracking_mandatory || (!primary_succeeded && tracker_state.has_previous_ &&
                               coding_markers.size() > size_t(float(total_expected_markers) * 0.3f));

    if (use_tracking && tracker_state.has_previous_)
    {
        spdlog::info("image {}: Using tracking (reason: {}, markers={}/{})", image_idx,
                     tracking_mandatory ? "fewer markers" : "identification failed", coding_markers.size(),
                     total_expected_markers);

        const auto tracking_result = identification::circlegrid::identify_with_tracking(
            tracker_state.prev_markers_, coding_markers, tracker_state.prev_global_ids_, 50.0f, 5.0f);

        if (tracking_result.has_value())
        {
            global_ids = *tracking_result;
            const int identified_count =
                static_cast<int>(std::count_if(global_ids.begin(), global_ids.end(), [](int id) { return id >= 0; }));
            spdlog::info("image {}: Tracking identified {} markers", image_idx, identified_count);
        }
    }
    else if (tracking_mandatory && !tracker_state.has_previous_)
    {
        spdlog::warn("image {}: Tracking is mandatory but no previous frame available (markers={}/{})", image_idx,
                     coding_markers.size(), total_expected_markers);
    }

    std::vector<base::MarkerRing> rings;
    rings.reserve(coding_markers.size());
    for (size_t i = 0; i < coding_markers.size(); ++i)
    {
        rings.emplace_back(coding_markers[i]);
        if (i < global_ids.size())
        {
            rings.back().global_id_ = global_ids[i];
        }
    }

    const size_t prev_count = tracker_state.has_previous_ ? tracker_state.prev_markers_.size() : 0;
    if (coding_markers.size() > prev_count && prev_count != 0)
    {
        if (coding_markers.size() != total_expected_markers)
        {
            identification::circlegrid::identify_new_markers_by_row_lines(rings, board);
        }
    }

    std::vector<int> current_ids;
    current_ids.reserve(rings.size());
    for (const auto &r : rings)
    {
        current_ids.push_back(r.global_id_);
    }
    tracker_state.update(coding_markers, current_ids);

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
        return std::nullopt;
    }

    spdlog::info("image {}: Final identification: {} / {} markers", image_idx, identified_markers,
                 total_expected_markers);

    const cv::Mat1b marker_area = create_marker_area(rings, input.rows, input.cols);
    const cv::Mat1b calibrated_area =
        create_calibrated_area(rings, std::make_unique<BoardCircleGrid>(board), input.rows, input.cols);

    if constexpr (kShowMarkers)
    {
        if (!output_path.empty())
        {
            save_markers(output_path, image_idx, total_expected_markers, identified_markers, rings, board, input,
                         ordering);

            io::debug::save_image(marker_area, std::format("marker_area_circle_{}", image_idx),
                                  output_path / debug::kMarkersSubdir);
            io::debug::save_image(calibrated_area, std::format("calibrated_area_circle_{}", image_idx),
                                  output_path / debug::kMarkersSubdir);
        }
    }

    return base::ImageDecoding(input, binarized, inverted_binarization, ordering, rings, marker_area, calibrated_area);
}

}  // namespace marker
