#include "identification_circle.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>

#include <spdlog/spdlog.h>
#include <Eigen/Core>
#include <opencv2/calib3d.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgproc.hpp>

namespace
{



/**
 * @brief Custom blob detector that returns pre-detected keypoints.
 *        This allows us to use findCirclesGrid with our own marker detection.
 */
class PredetectedBlobDetector : public cv::Feature2D
{
   public:
    explicit PredetectedBlobDetector(const std::vector<cv::KeyPoint>& keypoints) : keypoints_(keypoints) {}
    ~PredetectedBlobDetector() override = default;

    void detect(cv::InputArray, std::vector<cv::KeyPoint>& keypoints, cv::InputArray = cv::noArray()) override
    {
        keypoints = keypoints_;
    }

   private:
    std::vector<cv::KeyPoint> keypoints_;
};

using RowIdx = int;
using MarkerIdx = int;

struct RowInfo
{
    bool is_complete;
    Eigen::Vector2f direction;
    Eigen::Vector2f point_on_line;
    std::vector<MarkerIdx> marker_indices;
    std::optional<float> mean_spacing;
};

/**
 * @brief The function try to find missing in row_infos rows. It does that by searching the correspondance between
 *        the markers from all_markers vector and their rows. The rows which markers were found on previous frame
 *        in number at least 2 per row are detected, so has their global_id set correctly and are listed in
 *        RowInfo::marker_indices and RowInfo::is_complete is true. But markers which were not found on previous frame
 *        have their global_id==-1 and are not present in RowInfo::marker_indices, so they are gonna be found here.
 *        In the other hand markers which were found on previous frame but are in less number then two markers per
 *        one line are present on RowInfo::marker_indices but their RowInfo::is_complete is false, so they will be tried
 *        to be found first.
 * 
 * @param row_infos partial data about detected markers and their current rows correspondance
 * @param all_markers markers that were detected on current frame. Those which have been detected on previous frame has
 *        their global_id different then -1.
 */
void try_fill_missing_rows(std::map<RowIdx, RowInfo>& row_infos, const std::vector<base::MarkerRing>& all_markers) {
}

}  // namespace identification::circlegrid::detail

namespace identification
{

using circlegrid::TrackingState;

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

std::optional<std::vector<int>> circlegrid::identify_with_tracking(const std::vector<base::MarkerCoding>& prev_markers,
                                                                   const std::vector<base::MarkerCoding>& curr_markers,
                                                                   const std::vector<int>& prev_ids,
                                                                   float distance_threshold, float ransac_threshold)
{
    spdlog::debug("identify_with_tracking: prev={}, curr={}", prev_markers.size(), curr_markers.size());

    if (prev_markers.empty() || curr_markers.empty())
    {
        return std::nullopt;
    }

    cv::Mat prev_pts(int(prev_markers.size()), 2, CV_32F);
    cv::Mat curr_pts(int(curr_markers.size()), 2, CV_32F);

    for (size_t idx = 0; idx < prev_markers.size(); ++idx)
    {
        prev_pts.at<float>(int(idx), 0) = prev_markers[idx].col_;
        prev_pts.at<float>(int(idx), 1) = prev_markers[idx].row_;
    }
    for (size_t idx = 0; idx < curr_markers.size(); ++idx)
    {
        curr_pts.at<float>(int(idx), 0) = curr_markers[idx].col_;
        curr_pts.at<float>(int(idx), 1) = curr_markers[idx].row_;
    }

    spdlog::debug("identify_with_tracking: starting KNN match");
    cv::BFMatcher matcher(cv::NORM_L2);
    std::vector<std::vector<cv::DMatch>> knn_matches;
    matcher.knnMatch(curr_pts, prev_pts, knn_matches, 2);
    spdlog::debug("identify_with_tracking: KNN match done, {} matches", knn_matches.size());

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

    spdlog::debug("identify_with_tracking: starting RANSAC with {} correspondences", correspondences.size());
    std::vector<uchar> inlier_mask;
    const cv::Mat H = cv::findHomography(src_pts, dst_pts, cv::RANSAC, ransac_threshold, inlier_mask);
    spdlog::debug("identify_with_tracking: RANSAC done");

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

void circlegrid::identify_new_markers_by_row_lines(std::vector<base::MarkerRing>& markers, const BoardCircleGrid& board)
{
    std::map<RowIdx, RowInfo> row_infos;
    for (size_t marker_idx = 0; marker_idx < markers.size(); ++marker_idx)
    {
        if (markers[marker_idx].global_id_ < 0)
        {
            continue;
        }
        const RowIdx row = board.id_to_row_and_col(markers[marker_idx].global_id_)(0);
        row_infos[row].marker_indices.push_back(static_cast<MarkerIdx>(marker_idx));
    }

    for (auto& [row_idx, row_info] : row_infos)
    {
        if (row_info.marker_indices.size() < 2)
        {
            continue;
        }

        std::vector<std::pair<float, int>> sorted;

        for (const MarkerIdx idx : row_info.marker_indices)
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
        row_info.point_on_line = p1;
        row_info.direction = (p2 - p1) / length;

        float total_dist = 0.f;
        for (size_t i = 1; i < sorted.size(); ++i)
        {
            const base::MarkerRing& marker_a = markers[sorted[i - 1].second];
            const base::MarkerRing& marker_b = markers[sorted[i].second];
            if (std::abs(marker_a.global_id_ - marker_b.global_id_) != 1)
            {
                continue;
            }
            total_dist += std::hypot(marker_b.col_ - marker_a.col_, marker_b.row_ - marker_a.row_);
        }
        if (total_dist > 1e-6f)
        {
            row_info.mean_spacing.emplace(total_dist / static_cast<float>(sorted.size() - 1));
        }

        row_info.marker_indices.clear();
        for (const auto& [col, idx] : sorted)
        {
            row_info.marker_indices.push_back(idx);
        }
        row_info.is_complete = true;
    }

    if (row_infos.empty())
    {
        return;
    }

    try_fill_missing_rows(row_infos, markers);

    constexpr float kLineDistanceThreshold = 10.0f;

    for (auto& marker : markers)
    {
        if (marker.global_id_ >= 0)
        {
            continue;
        }

        const Eigen::Vector2f pos(marker.col_, marker.row_);
        float min_dist = kLineDistanceThreshold;
        RowIdx closest_row = -1;

        for (const auto& [row, ri] : row_infos)
        {
            const Eigen::Vector2f v = pos - ri.point_on_line;
            const float dist = std::abs(v.x() * (-ri.direction.y()) + v.y() * ri.direction.x());
            if (dist < min_dist)
            {
                min_dist = dist;
                closest_row = row;
            }
        }

        if (closest_row == -1)
        {
            continue;
        }

        float min_marker_dist = std::numeric_limits<float>::max();
        int closest_marker_idx = -1;

        for (const int idx : row_infos[closest_row].marker_indices)
        {
            const float dist = std::hypot(marker.col_ - markers[idx].col_, marker.row_ - markers[idx].row_);
            if (dist < min_marker_dist)
            {
                min_marker_dist = dist;
                closest_marker_idx = idx;
            }
        }

        if (closest_marker_idx < 0 || row_infos[closest_row].mean_spacing < 1e-6f)
        {
            continue;
        }

        const int col_offset = static_cast<int>(std::round(min_marker_dist / row_infos[closest_row].mean_spacing.value()));
        if (col_offset == 0)
        {
            continue;
        }

        const float direction = (marker.col_ > markers[closest_marker_idx].col_) ? 1.0f : -1.0f;
        const int ref_col = board.id_to_row_and_col(markers[closest_marker_idx].global_id_)(1);
        const int new_col = ref_col + static_cast<int>(direction * static_cast<float>(col_offset));

        if (new_col >= 0 && new_col < board.cols_)
        {
            marker.global_id_ = board.row_and_col_to_id(closest_row, new_col);
            spdlog::debug("Row-line identification: assigned marker at ({}, {}) to row={}, col={}", marker.row_,
                          marker.col_, closest_row, new_col);
        }
    }
}

bool circlegrid::test_find_circles_grid(std::vector<int>& indices,
                                        const std::vector<base::MarkerCoding>& coding_markers,
                                        const BoardCircleGrid& board)
{
    const size_t total_expected_markers = board.rows_ * board.cols_;
    if (coding_markers.size() < total_expected_markers)
    {
        return false;
    }

    // Convert markers to keypoints
    std::vector<cv::KeyPoint> keypoints;
    keypoints.reserve(coding_markers.size());
    for (const auto& marker : coding_markers)
    {
        const float size = static_cast<float>(marker.width_ring_ + marker.height_ring_) / 2.0f;
        keypoints.emplace_back(cv::Point2f(marker.col_, marker.row_), size);
    }

    // Create a dummy image (findCirclesGrid needs an image for size info)
    cv::Mat1b dummy_image = cv::Mat1b::zeros(1, 1);

    // Create a blob detector that returns our pre-detected keypoints
    cv::Ptr<cv::Feature2D> blob_detector = cv::makePtr<PredetectedBlobDetector>(keypoints);

    const cv::Size pattern_size(board.cols_, board.rows_);
    std::vector<cv::Point2f> centers;
    int flags = board.is_asymetric_ ? cv::CALIB_CB_ASYMMETRIC_GRID : cv::CALIB_CB_SYMMETRIC_GRID;
    flags |= cv::CALIB_CB_CLUSTERING;  // Use clustering algorithm which works better with pre-detected points

    const bool success = cv::findCirclesGrid(dummy_image, pattern_size, centers, flags, blob_detector);
    if (success)
    {
        indices.clear();
        for (size_t idx_marker = 0; idx_marker < coding_markers.size(); ++idx_marker)
        {
            bool found = false;
            const base::MarkerCoding& marker = coding_markers[idx_marker];
            for (size_t idx_center = 0; idx_center < centers.size(); ++idx_center)
            {
                const cv::Point2f& center = centers[idx_center];
                if (found = marker.row_ == center.y && marker.col_ == center.x; found)
                {
                    indices.push_back(int(idx_center));
                    break;
                }
            }

            if (!found)
            {
                spdlog::warn("Center id {} ({:0.1f}, {:0.1f}) was not identified!", idx_marker, marker.col_,
                             marker.row_);
            }
        }

        if (total_expected_markers != indices.size())
        {
            throw std::runtime_error(
                std::format("Identified {} markers of {}!", indices.size(), total_expected_markers));
        }
    }

    return success;
}

}  // namespace identification
