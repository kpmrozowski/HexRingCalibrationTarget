#include "detection.hpp"

#include <filesystem>

#include <spdlog/spdlog.h>
#include <opencv2/imgproc.hpp>

#include <io/debug.hpp>

#include "debugging.hpp"
#include "thresholds.hpp"

#include "debug.hpp"
#include "identification/identification.hpp"
#include "symmetries.hpp"

namespace
{
cv::Mat1b binarize(cv::Mat1b &input, const marker::DetectionParameters &parameters)
{
    auto thresholds = thresholds::thresholds(input, parameters.row_tiles_count_, parameters.col_tiles_count_);
    for (auto &tresh : thresholds.thresholds_)
    {
        tresh = std::max(tresh, parameters.minimal_treshold_);
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

        if (longest > parameters.maximal_edge_lenght_core_)
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

        if (longest > parameters.maximal_edge_lenght_ring_)
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
            stats(object_idx, cv::CC_STAT_WIDTH) * parameters.reduction_in_edge_lenght_ring_,
            stats(object_idx, cv::CC_STAT_HEIGHT) * parameters.reduction_in_edge_lenght_ring_,
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
    cv::Mat1b board_markers = cv::Mat1b::zeros(board->cols_, board->cols_);

    std::map<int, int> id_to_marker;
    for (size_t marker = 0; marker < rings.size(); ++marker)
    {
        const int marker_id = rings[marker].global_id_;
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

}  // namespace

namespace marker::detection
{
std::optional<base::ImageDecoding> detect_and_identify(cv::Mat1b &input, const DetectionParameters &parameters,
                                                       const std::unique_ptr<Board> &board, const int image_idx)
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
}  // namespace marker::detection
