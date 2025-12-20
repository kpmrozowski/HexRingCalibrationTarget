#include "identification_circle.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>

#include <Eigen/Core>
#include <opencv2/calib3d.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgproc.hpp>
#include <spdlog/spdlog.h>

namespace identification::circlegrid::detail
{
/**
 * @brief Custom blob detector that returns pre-detected keypoints.
 *        This allows us to use findCirclesGrid with our own marker detection.
 */
class PredetectedBlobDetector : public cv::Feature2D
{
   public:
    explicit PredetectedBlobDetector(const std::vector<cv::KeyPoint> &keypoints) : keypoints_(keypoints) {}
    ~PredetectedBlobDetector() override = default;

    void detect(cv::InputArray, std::vector<cv::KeyPoint> &keypoints, cv::InputArray = cv::noArray()) override
    {
        keypoints = keypoints_;
    }

   private:
    std::vector<cv::KeyPoint> keypoints_;
};
}  // namespace identification::circlegrid::detail

namespace identification::circlegrid
{

void TrackingState::update(const std::vector<base::MarkerCoding>& markers, const std::vector<int>& global_ids)
{
    prev_markers_ = markers;
    prev_global_ids_ = global_ids;
    has_previous_ = true;
}

void TrackingState::clear()
{
    prev_markers_.clear();
    prev_global_ids_.clear();
    has_previous_ = false;
}

std::optional<std::vector<int>> identify_with_findCirclesGrid(const cv::Mat1b& image,
                                                               const std::vector<base::MarkerCoding>& coding_markers,
                                                               const BoardCircleGrid& board)
{
    // Simple geometric identification:
    // 1. Sort markers by row (y-coordinate)
    // 2. Group markers into rows
    // 3. Within each row, sort by column (x-coordinate)
    // 4. Assign IDs based on grid position
    // 5. Handle extra markers by keeping only the expected number per row

    if (coding_markers.size() < static_cast<size_t>(board.rows_ * board.cols_))
    {
        spdlog::debug("Not enough markers for geometric identification: {} < {}", coding_markers.size(),
                      board.rows_ * board.cols_);
        return std::nullopt;
    }

    // Create index pairs for sorting (index, marker)
    std::vector<std::pair<int, const base::MarkerCoding*>> indexed_markers;
    indexed_markers.reserve(coding_markers.size());
    for (size_t i = 0; i < coding_markers.size(); ++i)
    {
        indexed_markers.emplace_back(static_cast<int>(i), &coding_markers[i]);
    }

    // Sort by row (y-coordinate)
    std::sort(indexed_markers.begin(), indexed_markers.end(),
              [](const auto& a, const auto& b) { return a.second->row_ < b.second->row_; });

    // Calculate gaps between consecutive y-coordinates
    std::vector<std::pair<float, size_t>> gaps;  // (gap_size, index_after_gap)
    for (size_t i = 1; i < indexed_markers.size(); ++i)
    {
        const float gap = indexed_markers[i].second->row_ - indexed_markers[i - 1].second->row_;
        gaps.emplace_back(gap, i);
    }

    // Sort gaps by size (descending) and take the (board.rows_ - 1) largest gaps as row separators
    std::sort(gaps.begin(), gaps.end(),
              [](const auto& a, const auto& b) { return a.first > b.first; });

    // Extract the indices where we should split into new rows
    std::vector<size_t> split_indices;
    const size_t num_row_gaps = static_cast<size_t>(board.rows_ - 1);
    for (size_t i = 0; i < std::min(num_row_gaps, gaps.size()); ++i)
    {
        split_indices.push_back(gaps[i].second);
    }
    std::sort(split_indices.begin(), split_indices.end());

    // Group into rows using the split indices
    std::vector<std::vector<std::pair<int, const base::MarkerCoding*>>> rows;
    rows.emplace_back();
    size_t split_idx = 0;

    for (size_t i = 0; i < indexed_markers.size(); ++i)
    {
        if (split_idx < split_indices.size() && i == split_indices[split_idx])
        {
            rows.emplace_back();
            ++split_idx;
        }
        rows.back().emplace_back(indexed_markers[i]);
    }

    // Check if we have the expected number of rows
    if (static_cast<int>(rows.size()) != board.rows_)
    {
        spdlog::debug("Geometric identification: wrong number of rows {} vs expected {}", rows.size(), board.rows_);
        // Log row sizes for debugging
        for (size_t i = 0; i < rows.size(); ++i)
        {
            spdlog::debug("  Row {}: {} markers", i, rows[i].size());
        }
        return std::nullopt;
    }

    // Sort each row by x-coordinate
    for (auto& row : rows)
    {
        std::sort(row.begin(), row.end(),
                  [](const auto& a, const auto& b) { return a.second->col_ < b.second->col_; });
    }

    // Validate and trim rows - each row should have at least board.cols_ markers
    // If a row has more markers, remove outliers based on spacing
    for (auto& row : rows)
    {
        if (static_cast<int>(row.size()) < board.cols_)
        {
            spdlog::debug("Geometric identification: row has too few markers {} < {}", row.size(), board.cols_);
            return std::nullopt;
        }

        // If row has extra markers, find and remove outliers
        while (static_cast<int>(row.size()) > board.cols_)
        {
            // Calculate spacing between adjacent markers
            std::vector<float> spacings;
            for (size_t i = 1; i < row.size(); ++i)
            {
                spacings.push_back(row[i].second->col_ - row[i - 1].second->col_);
            }

            // Find the median spacing
            std::vector<float> sorted_spacings = spacings;
            std::sort(sorted_spacings.begin(), sorted_spacings.end());
            const float median_spacing = sorted_spacings[sorted_spacings.size() / 2];

            // Find the marker pair with the smallest spacing (likely the outlier is one of them)
            size_t min_spacing_idx = 0;
            float min_spacing = spacings[0];
            for (size_t i = 1; i < spacings.size(); ++i)
            {
                if (spacings[i] < min_spacing)
                {
                    min_spacing = spacings[i];
                    min_spacing_idx = i;
                }
            }

            // Remove the marker that makes the spacing more consistent
            // If it's at the start or end, remove the boundary marker
            // Otherwise, remove the one that creates a larger gap
            if (min_spacing_idx == 0)
            {
                row.erase(row.begin());
            }
            else if (min_spacing_idx == spacings.size() - 1)
            {
                row.pop_back();
            }
            else
            {
                // Remove the marker that is closer to its neighbor
                const float left_gap = spacings[min_spacing_idx - 1];
                const float right_gap = spacings[min_spacing_idx + 1];
                if (left_gap < right_gap)
                {
                    row.erase(row.begin() + static_cast<long>(min_spacing_idx));
                }
                else
                {
                    row.erase(row.begin() + static_cast<long>(min_spacing_idx) + 1);
                }
            }
        }
    }

    // Assign global IDs
    std::vector<int> global_ids(coding_markers.size(), -1);

    for (int row_idx = 0; row_idx < static_cast<int>(rows.size()); ++row_idx)
    {
        for (int col_idx = 0; col_idx < static_cast<int>(rows[row_idx].size()); ++col_idx)
        {
            const int marker_idx = rows[row_idx][col_idx].first;
            const int global_id = board.row_and_col_to_id(row_idx, col_idx);
            global_ids[marker_idx] = global_id;
        }
    }

    const int identified_count =
        static_cast<int>(std::count_if(global_ids.begin(), global_ids.end(), [](int id) { return id >= 0; }));
    spdlog::debug("Geometric identification: assigned {} IDs", identified_count);

    return global_ids;
}

std::optional<std::vector<int>> identify_with_tracking(const std::vector<base::MarkerCoding>& prev_markers,
                                                        const std::vector<base::MarkerCoding>& curr_markers,
                                                        const std::vector<int>& prev_ids, float distance_threshold,
                                                        float ransac_threshold)
{
    if (prev_markers.empty() || curr_markers.empty())
    {
        return std::nullopt;
    }

    cv::Mat prev_pts(static_cast<int>(prev_markers.size()), 2, CV_32F);
    cv::Mat curr_pts(static_cast<int>(curr_markers.size()), 2, CV_32F);

    for (size_t i = 0; i < prev_markers.size(); ++i)
    {
        prev_pts.at<float>(static_cast<int>(i), 0) = prev_markers[i].col_;
        prev_pts.at<float>(static_cast<int>(i), 1) = prev_markers[i].row_;
    }
    for (size_t i = 0; i < curr_markers.size(); ++i)
    {
        curr_pts.at<float>(static_cast<int>(i), 0) = curr_markers[i].col_;
        curr_pts.at<float>(static_cast<int>(i), 1) = curr_markers[i].row_;
    }

    cv::BFMatcher matcher(cv::NORM_L2);
    std::vector<std::vector<cv::DMatch>> knn_matches;
    matcher.knnMatch(curr_pts, prev_pts, knn_matches, 2);

    std::vector<cv::Point2f> src_pts, dst_pts;
    std::vector<std::pair<int, int>> correspondences;

    constexpr float kRatioThreshold = 0.75f;

    for (size_t i = 0; i < knn_matches.size(); ++i)
    {
        if (knn_matches[i].size() < 2)
        {
            continue;
        }

        const auto& best = knn_matches[i][0];
        const auto& second = knn_matches[i][1];

        if (best.distance > kRatioThreshold * second.distance)
        {
            continue;
        }

        if (best.distance > distance_threshold)
        {
            continue;
        }

        src_pts.emplace_back(curr_markers[i].col_, curr_markers[i].row_);
        dst_pts.emplace_back(prev_markers[best.trainIdx].col_, prev_markers[best.trainIdx].row_);
        correspondences.emplace_back(best.trainIdx, static_cast<int>(i));
    }

    constexpr size_t kMinCorrespondences = 4;
    if (correspondences.size() < kMinCorrespondences)
    {
        spdlog::debug("Tracking: insufficient correspondences ({} < {})", correspondences.size(), kMinCorrespondences);
        return std::nullopt;
    }

    std::vector<uchar> inlier_mask;
    const cv::Mat H = cv::findHomography(src_pts, dst_pts, cv::RANSAC, ransac_threshold, inlier_mask);

    if (H.empty())
    {
        spdlog::debug("Tracking: RANSAC failed to find homography");
        return std::nullopt;
    }

    std::vector<int> global_ids(curr_markers.size(), -1);
    int inlier_count = 0;

    for (size_t i = 0; i < inlier_mask.size(); ++i)
    {
        if (inlier_mask[i])
        {
            const int prev_idx = correspondences[i].first;
            const int curr_idx = correspondences[i].second;

            if (prev_idx < static_cast<int>(prev_ids.size()) && prev_ids[prev_idx] >= 0)
            {
                global_ids[curr_idx] = prev_ids[prev_idx];
                ++inlier_count;
            }
        }
    }

    spdlog::debug("Tracking: {} inliers identified", inlier_count);
    return global_ids;
}

void identify_new_markers_by_row_lines(std::vector<base::MarkerRing>& markers, const BoardCircleGrid& board)
{
    std::map<int, std::vector<int>> row_to_indices;
    for (size_t i = 0; i < markers.size(); ++i)
    {
        if (markers[i].global_id_ < 0)
        {
            continue;
        }
        const int row = board.id_to_row_and_col(markers[i].global_id_)(0);
        row_to_indices[row].push_back(static_cast<int>(i));
    }

    struct RowInfo
    {
        int row_id;
        Eigen::Vector2f direction;
        Eigen::Vector2f point_on_line;
        float mean_spacing;
        std::vector<int> marker_indices;
    };
    std::vector<RowInfo> row_infos;

    for (const auto& [row_id, indices] : row_to_indices)
    {
        if (indices.size() < 2)
        {
            continue;
        }

        std::vector<std::pair<float, int>> sorted;
        for (const int idx : indices)
        {
            sorted.emplace_back(markers[idx].col_, idx);
        }
        std::sort(sorted.begin(), sorted.end());

        const int first = sorted.front().second;
        const int last = sorted.back().second;
        const Eigen::Vector2f p1(markers[first].col_, markers[first].row_);
        const Eigen::Vector2f p2(markers[last].col_, markers[last].row_);

        const float length = (p2 - p1).norm();
        if (length < 1e-6f)
        {
            continue;
        }

        const Eigen::Vector2f dir = (p2 - p1) / length;

        float total_dist = 0;
        for (size_t i = 1; i < sorted.size(); ++i)
        {
            const int prev = sorted[i - 1].second;
            const int curr = sorted[i].second;
            total_dist +=
                std::hypot(markers[curr].col_ - markers[prev].col_, markers[curr].row_ - markers[prev].row_);
        }
        const float mean_spacing = total_dist / static_cast<float>(sorted.size() - 1);

        RowInfo ri;
        ri.row_id = row_id;
        ri.direction = dir;
        ri.point_on_line = p1;
        ri.mean_spacing = mean_spacing;
        for (const auto& [col, idx] : sorted)
        {
            ri.marker_indices.push_back(idx);
        }
        row_infos.push_back(ri);
    }

    if (row_infos.empty())
    {
        return;
    }

    constexpr float kLineDistanceThreshold = 10.0f;

    for (auto& marker : markers)
    {
        if (marker.global_id_ >= 0)
        {
            continue;
        }

        const Eigen::Vector2f pos(marker.col_, marker.row_);
        float min_dist = kLineDistanceThreshold;
        const RowInfo* closest_row = nullptr;

        for (const auto& ri : row_infos)
        {
            const Eigen::Vector2f v = pos - ri.point_on_line;
            const float dist = std::abs(v.x() * (-ri.direction.y()) + v.y() * ri.direction.x());
            if (dist < min_dist)
            {
                min_dist = dist;
                closest_row = &ri;
            }
        }

        if (closest_row == nullptr)
        {
            continue;
        }

        float min_marker_dist = std::numeric_limits<float>::max();
        int closest_marker_idx = -1;

        for (const int idx : closest_row->marker_indices)
        {
            const float dist = std::hypot(marker.col_ - markers[idx].col_, marker.row_ - markers[idx].row_);
            if (dist < min_marker_dist)
            {
                min_marker_dist = dist;
                closest_marker_idx = idx;
            }
        }

        if (closest_marker_idx < 0 || closest_row->mean_spacing < 1e-6f)
        {
            continue;
        }

        const int col_offset = static_cast<int>(std::round(min_marker_dist / closest_row->mean_spacing));
        if (col_offset == 0)
        {
            continue;
        }

        const float direction = (marker.col_ > markers[closest_marker_idx].col_) ? 1.0f : -1.0f;
        const int ref_col = board.id_to_row_and_col(markers[closest_marker_idx].global_id_)(1);
        const int new_col = ref_col + static_cast<int>(direction * static_cast<float>(col_offset));

        if (new_col >= 0 && new_col < board.cols_)
        {
            marker.global_id_ = board.row_and_col_to_id(closest_row->row_id, new_col);
            spdlog::debug("Row-line identification: assigned marker at ({}, {}) to row={}, col={}", marker.row_,
                          marker.col_, closest_row->row_id, new_col);
        }
    }
}

}  // namespace identification::circlegrid
