#include "identification_circle.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <set>

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
    std::optional<Eigen::Vector2f> direction;
    Eigen::Vector2f point_on_line;
    std::vector<MarkerIdx> marker_indices;
    std::optional<float> mean_spacing;
};

/**
 * @brief The function try to find unindentified_indices on the board and place their info either on existing row or
 *        creates new row and assignes the marker to that row.
 * Stage 1: Assign unindentified markers to existing rows:
 *  step 1.1 compute distance form unindentified marker to line created from RowInfo::direction and
 *           RowInfo::point_on_line,
 *  step 1.2 check which unindentified markers lay on which existing row (use some tolerance);
 *  step 1.3 assign global_id for the indentified marker based on id of it's neighbour;
 * Stage 2: If some unassigned markers left, then assign them to new rows:
 *  step 2.1: collect directions of existing rows,
 *  step 2.2: compute mean perpendicular direction and create line from the perpendicular direction and one of
 *            RowInfo::point_on_line of some RowInfo,
 *  step 2.3: compute intersections of existing lines and the perpendicular line,
 *  step 2.4: compute mean offset as a mean distance between those points,
 *  step 2.5: create two candidate RowInfo::point_on_line (first uppon upper edge line and second below lower edge
 *            line) offsetted by computed in 2.4 mean distance form edge lines,
 *  step 2.6: compute direction for candidate lines passign through candidate RowInfo::point_on_line by
 *            extrapolating RowInfo::direction from two edge rows or if only single row has direction then use it
 *  step 2.7: check if the unindentified markers lay on the candidate lines (in some tolerance), if yes create the
 *            RowInfo for them and RowIdx and add it to row_infos
 * Stage 3: Assigning global_ids for unindentified markers which rows were identified:
 *  step 3.1: fit lines to identified markers that correspond to the same columns,
 *  step 3.2: using lines form 3.1 compute mean columns offset as in 2.2, 2.3, 2.4,
 *  step 3.3: if board.is_asymetric_ then the columns locations are computed as (2 * col + row % 2) * board.spacing_
 *            else the column location is computed as board.spacing_ * col, so for symmetric case check which
 *            unindentified markers having identified rows lay on which fitted line form step 3.1, and for
 *            assymmetric case check which unindentified markers having EVEN identified rows lay in approx 0.25
 *            mean columns offset distance to the left from which line form step 3.1 and which unindentified markers
 *            with ODD identified rows lay in approx 0.25 mean columns offset distance to the right from which line
 *            form step 3.1.
 *  step 3.4: based on the information from step 3.4 assign columns to the unindentified markers with indentified
 *            rows
 * @param row_infos partial data about detected markers and their current rows correspondance
 * @param unindentified_indices markers indices to indentify
 * @param all_markers markers that were detected on current frame. Those which have been detected on previous frame has
 *        their global_id different then -1.
 * @param board Board definition for id_to_row_and_col conversion
 */
void try_fill_missing_rows(std::map<RowIdx, RowInfo>& row_infos, const std::vector<MarkerIdx>& unindentified_indices,
                           std::vector<base::MarkerRing>& all_markers, const BoardCircleGrid& board)
{
    constexpr float kLineDistanceTolerance = 15.0f;

    // Track which unidentified markers have been assigned
    std::set<MarkerIdx> assigned_markers;

    // ===================================================================================
    // STAGE 1: Assign unidentified markers to existing rows
    // ===================================================================================

    // Compute mean direction and mean spacing from all rows (for fallback)
    Eigen::Vector2f mean_row_direction = Eigen::Vector2f::Zero();
    float global_mean_spacing = 0.f;
    int spacing_count = 0;
    for (const auto& [row_idx, row_info] : row_infos)
    {
        if (row_info.direction.has_value())
        {
            mean_row_direction += row_info.direction.value();
        }
        if (row_info.mean_spacing.has_value())
        {
            global_mean_spacing += row_info.mean_spacing.value();
            ++spacing_count;
        }
    }
    if (mean_row_direction.norm() > 1e-6f)
    {
        mean_row_direction.normalize();
    }
    if (spacing_count > 0)
    {
        global_mean_spacing /= static_cast<float>(spacing_count);
    }

    // Step 1.1 & 1.2: For each unidentified marker, find the closest existing row line
    for (const MarkerIdx marker_idx : unindentified_indices)
    {
        const base::MarkerRing& marker = all_markers[marker_idx];
        const Eigen::Vector2f pos(marker.col_, marker.row_);

        float min_dist = kLineDistanceTolerance;
        RowIdx best_row = -1;

        for (const auto& [row_idx, row_info] : row_infos)
        {
            // Use row's own direction if available, otherwise use mean direction
            Eigen::Vector2f dir;
            if (row_info.direction.has_value())
            {
                dir = row_info.direction.value();
            }
            else if (mean_row_direction.norm() > 1e-6f)
            {
                dir = mean_row_direction;
            }
            else
            {
                continue;  // No valid direction available
            }

            // Compute perpendicular distance to line: |v × direction|
            const Eigen::Vector2f v = pos - row_info.point_on_line;
            const float dist = std::abs(v.x() * (-dir.y()) + v.y() * dir.x());

            if (dist < min_dist)
            {
                min_dist = dist;
                best_row = row_idx;
            }
        }

        if (best_row >= 0)
        {
            // Step 1.3: Assign global_id based on neighbor's id
            const RowInfo& row_info = row_infos.at(best_row);

            // Find closest marker in this row
            float min_marker_dist = std::numeric_limits<float>::max();
            int closest_marker_idx = -1;

            for (const MarkerIdx idx : row_info.marker_indices)
            {
                const float dist = std::hypot(marker.col_ - all_markers[idx].col_, marker.row_ - all_markers[idx].row_);
                if (dist < min_marker_dist)
                {
                    min_marker_dist = dist;
                    closest_marker_idx = idx;
                }
            }

            // Use row's mean_spacing if available, otherwise use global mean spacing
            const float spacing = row_info.mean_spacing.value_or(global_mean_spacing);

            if (closest_marker_idx >= 0 && spacing > 1e-6f)
            {
                const int col_offset = static_cast<int>(std::round(min_marker_dist / spacing));
                if (col_offset > 0)
                {
                    const float direction = (marker.col_ > all_markers[closest_marker_idx].col_) ? 1.0f : -1.0f;
                    const int ref_col = board.id_to_row_and_col(all_markers[closest_marker_idx].global_id_)(1);
                    const int new_col = ref_col + static_cast<int>(direction * static_cast<float>(col_offset));

                    if (new_col >= 0 && new_col < board.cols_)
                    {
                        all_markers[marker_idx].global_id_ = board.row_and_col_to_id(best_row, new_col);
                        row_infos[best_row].marker_indices.push_back(marker_idx);
                        assigned_markers.insert(marker_idx);
                        spdlog::debug("Stage1: Assigned marker {} to row={}, col={}", marker_idx, best_row, new_col);
                    }
                }
            }
        }
    }

    // Collect remaining unassigned markers
    std::vector<MarkerIdx> remaining_unassigned;
    for (const MarkerIdx idx : unindentified_indices)
    {
        if (assigned_markers.find(idx) == assigned_markers.end())
        {
            remaining_unassigned.push_back(idx);
        }
    }

    if (remaining_unassigned.empty())
    {
        return;
    }

    // ===================================================================================
    // STAGE 2: Create new rows for remaining unassigned markers
    // ===================================================================================

    // Step 2.1: Collect directions of existing rows
    std::vector<Eigen::Vector2f> row_directions;
    std::vector<RowIdx> rows_with_direction;
    for (const auto& [row_idx, row_info] : row_infos)
    {
        if (row_info.direction.has_value())
        {
            row_directions.push_back(row_info.direction.value());
            rows_with_direction.push_back(row_idx);
        }
    }

    if (row_directions.empty())
    {
        return;
    }

    // Step 2.2: Compute mean direction and perpendicular
    Eigen::Vector2f mean_direction = Eigen::Vector2f::Zero();
    for (const auto& dir : row_directions)
    {
        mean_direction += dir;
    }
    mean_direction.normalize();
    const Eigen::Vector2f perpendicular(-mean_direction.y(), mean_direction.x());

    // Pick a reference point (from the first row with direction)
    const Eigen::Vector2f ref_point = row_infos.at(rows_with_direction[0]).point_on_line;

    // Step 2.3 & 2.4: Compute intersections with perpendicular line and mean offset
    std::vector<std::pair<float, RowIdx>> intersections;  // (distance along perpendicular, row_idx)

    for (const RowIdx row_idx : rows_with_direction)
    {
        const RowInfo& ri = row_infos.at(row_idx);
        // Intersection of perpendicular line (ref_point + t * perpendicular) with row line
        // ri.point_on_line + s * ri.direction = ref_point + t * perpendicular
        // Solve for t: (ri.point_on_line - ref_point) = t * perpendicular - s * ri.direction

        const Eigen::Vector2f diff = ri.point_on_line - ref_point;
        const Eigen::Vector2f& d = ri.direction.value();

        // Using 2D cross product to solve
        const float denom = perpendicular.x() * d.y() - perpendicular.y() * d.x();
        if (std::abs(denom) < 1e-6f)
        {
            continue;
        }

        const float t = (diff.x() * d.y() - diff.y() * d.x()) / denom;
        intersections.emplace_back(t, row_idx);
    }

    if (intersections.size() < 2)
    {
        return;
    }

    // Sort by distance along perpendicular
    std::sort(intersections.begin(), intersections.end());

    // Compute mean row offset
    float total_offset = 0.f;
    int offset_count = 0;
    for (size_t i = 1; i < intersections.size(); ++i)
    {
        const int row_diff = std::abs(intersections[i].second - intersections[i - 1].second);
        if (row_diff > 0)
        {
            total_offset += (intersections[i].first - intersections[i - 1].first) / static_cast<float>(row_diff);
            ++offset_count;
        }
    }

    if (offset_count == 0)
    {
        return;
    }

    const float mean_row_offset = total_offset / static_cast<float>(offset_count);

    // Step 2.5 & 2.6: Create candidate rows above and below
    const RowIdx min_row = intersections.front().second;
    const RowIdx max_row = intersections.back().second;
    const float min_t = intersections.front().first;
    const float max_t = intersections.back().first;

    // Direction extrapolation: use edge row directions or mean direction
    Eigen::Vector2f direction_for_candidates = mean_direction;
    if (rows_with_direction.size() >= 2)
    {
        // Could extrapolate, but for simplicity use mean
        direction_for_candidates = mean_direction;
    }

    // Create candidate rows
    std::map<RowIdx, Eigen::Vector2f> candidate_row_points;

    // Candidates above (smaller row indices)
    for (int candidate_row = min_row - 1; candidate_row >= 0; --candidate_row)
    {
        const int steps = min_row - candidate_row;
        const float t = min_t - steps * mean_row_offset;
        candidate_row_points[candidate_row] = ref_point + t * perpendicular;
    }

    // Candidates below (larger row indices)
    for (int candidate_row = max_row + 1; candidate_row < board.rows_; ++candidate_row)
    {
        const int steps = candidate_row - max_row;
        const float t = max_t + steps * mean_row_offset;
        candidate_row_points[candidate_row] = ref_point + t * perpendicular;
    }

    // Step 2.7: Check which unassigned markers lie on candidate rows
    for (const MarkerIdx marker_idx : remaining_unassigned)
    {
        const base::MarkerRing& marker = all_markers[marker_idx];
        const Eigen::Vector2f pos(marker.col_, marker.row_);

        float min_dist = kLineDistanceTolerance;
        RowIdx best_candidate_row = -1;

        for (const auto& [row_idx, point_on_line] : candidate_row_points)
        {
            // Distance to candidate line
            const Eigen::Vector2f v = pos - point_on_line;
            const float dist = std::abs(v.x() * (-direction_for_candidates.y()) + v.y() * direction_for_candidates.x());

            if (dist < min_dist)
            {
                min_dist = dist;
                best_candidate_row = row_idx;
            }
        }

        if (best_candidate_row >= 0)
        {
            // Create or update RowInfo for this candidate row
            if (row_infos.find(best_candidate_row) == row_infos.end())
            {
                RowInfo new_row;
                new_row.point_on_line = candidate_row_points[best_candidate_row];
                new_row.direction = direction_for_candidates;
                new_row.mean_spacing = row_infos.begin()->second.mean_spacing;  // Use existing row's spacing
                row_infos[best_candidate_row] = new_row;
            }
            row_infos[best_candidate_row].marker_indices.push_back(marker_idx);
            assigned_markers.insert(marker_idx);
            spdlog::debug("Stage2: Assigned marker {} to candidate row={}", marker_idx, best_candidate_row);
        }
    }

    // ===================================================================================
    // STAGE 3: Assign global_ids (columns) to markers with identified rows
    // ===================================================================================

    // Collect all markers that have row but no global_id yet
    std::vector<std::pair<RowIdx, MarkerIdx>> markers_needing_col;
    for (const auto& [row_idx, row_info] : row_infos)
    {
        for (const MarkerIdx marker_idx : row_info.marker_indices)
        {
            if (all_markers[marker_idx].global_id_ < 0)
            {
                markers_needing_col.emplace_back(row_idx, marker_idx);
            }
        }
    }

    if (markers_needing_col.empty())
    {
        return;
    }

    // Step 3.1: Collect identified markers grouped by column and row parity
    // For asymmetric grids, even and odd rows have different horizontal offsets,
    // so we only compare markers from rows with the same parity
    std::map<int, std::vector<MarkerIdx>> markers_by_col_even;  // col -> markers in even rows
    std::map<int, std::vector<MarkerIdx>> markers_by_col_odd;   // col -> markers in odd rows
    for (const auto& [row_idx, row_info] : row_infos)
    {
        for (const MarkerIdx marker_idx : row_info.marker_indices)
        {
            if (all_markers[marker_idx].global_id_ >= 0)
            {
                const int col = board.id_to_row_and_col(all_markers[marker_idx].global_id_)(1);
                if (row_idx % 2 == 0)
                {
                    markers_by_col_even[col].push_back(marker_idx);
                }
                else
                {
                    markers_by_col_odd[col].push_back(marker_idx);
                }
            }
        }
    }

    // Step 3.2: Compute row direction (direction along which columns are separated)
    // Columns are separated along the row direction (horizontal), not perpendicular to it
    Eigen::Vector2f row_direction = Eigen::Vector2f::Zero();
    for (const auto& [row_idx, row_info] : row_infos)
    {
        if (row_info.direction.has_value())
        {
            row_direction += row_info.direction.value();
        }
    }
    if (row_direction.norm() < 1e-6f)
    {
        return;  // No valid row directions
    }
    row_direction.normalize();

    // Helper lambda to compute column positions for a given parity
    auto compute_col_positions = [&](const std::map<int, std::vector<MarkerIdx>>& markers_by_col)
        -> std::tuple<std::vector<std::pair<float, int>>, Eigen::Vector2f, float> {
        if (markers_by_col.empty())
        {
            return {{}, Eigen::Vector2f::Zero(), 0.f};
        }

        // Get reference point from first marker
        const MarkerIdx first_marker = markers_by_col.begin()->second.front();
        const Eigen::Vector2f ref_point(all_markers[first_marker].col_, all_markers[first_marker].row_);

        // For each column, compute mean position projected onto perpendicular direction
        std::vector<std::pair<float, int>> col_positions;
        for (const auto& [col, indices] : markers_by_col)
        {
            float sum_proj = 0.f;
            for (const MarkerIdx idx : indices)
            {
                const Eigen::Vector2f pos(all_markers[idx].col_, all_markers[idx].row_);
                const Eigen::Vector2f diff = pos - ref_point;
                sum_proj += diff.dot(row_direction);
            }
            const float mean_proj = sum_proj / static_cast<float>(indices.size());
            col_positions.emplace_back(mean_proj, col);
        }
        std::sort(col_positions.begin(), col_positions.end());

        // Compute mean column spacing
        float total_spacing = 0.f;
        int spacing_count = 0;
        for (size_t i = 1; i < col_positions.size(); ++i)
        {
            const int col_diff = std::abs(col_positions[i].second - col_positions[i - 1].second);
            if (col_diff > 0)
            {
                total_spacing +=
                    (col_positions[i].first - col_positions[i - 1].first) / static_cast<float>(col_diff);
                ++spacing_count;
            }
        }
        const float mean_spacing = (spacing_count > 0) ? total_spacing / static_cast<float>(spacing_count) : 0.f;

        return {col_positions, ref_point, mean_spacing};
    };

    // Compute column positions for even and odd rows separately
    auto [col_positions_even, ref_point_even, spacing_even] = compute_col_positions(markers_by_col_even);
    auto [col_positions_odd, ref_point_odd, spacing_odd] = compute_col_positions(markers_by_col_odd);

    spdlog::debug("Stage3: {} markers need col, {} even cols, {} odd cols, spacing_even={}, spacing_odd={}",
                  markers_needing_col.size(), col_positions_even.size(), col_positions_odd.size(), spacing_even,
                  spacing_odd);

    // Step 3.3 & 3.4: Assign columns to markers with identified rows
    for (const auto& [row_idx, marker_idx] : markers_needing_col)
    {
        const base::MarkerRing& marker = all_markers[marker_idx];
        const Eigen::Vector2f pos(marker.col_, marker.row_);

        // Select column positions based on row parity
        const bool is_even = (row_idx % 2 == 0);
        const auto& col_positions = is_even ? col_positions_even : col_positions_odd;
        const auto& ref_point = is_even ? ref_point_even : ref_point_odd;
        const float mean_col_spacing = is_even ? spacing_even : spacing_odd;

        if (col_positions.empty() || mean_col_spacing < 1e-6f)
        {
            continue;
        }

        // Project marker position onto row direction to get column position
        const Eigen::Vector2f diff = pos - ref_point;
        const float proj = diff.dot(row_direction);

        // Find closest column
        float min_col_dist = std::abs(mean_col_spacing) * 0.5f;  // Half column tolerance
        int best_col = -1;

        for (const auto& [col_proj, col] : col_positions)
        {
            const float dist = std::abs(proj - col_proj);
            if (dist < min_col_dist)
            {
                min_col_dist = dist;
                best_col = col;
            }
        }

        // Also check for columns between existing ones by interpolation
        if (best_col < 0)
        {
            const float col_float =
                (proj - col_positions.front().first) / mean_col_spacing + col_positions.front().second;
            const int estimated_col = static_cast<int>(std::round(col_float));
            if (estimated_col >= 0 && estimated_col < board.cols_)
            {
                best_col = estimated_col;
            }
        }

        if (best_col >= 0 && best_col < board.cols_)
        {
            all_markers[marker_idx].global_id_ = board.row_and_col_to_id(row_idx, best_col);
            spdlog::debug("Stage3: Assigned marker {} to row={}, col={}", marker_idx, row_idx, best_col);
        }
    }
}

}  // namespace

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
    std::vector<MarkerIdx> unindentified_indices;
    std::map<RowIdx, RowInfo> row_infos;
    for (size_t marker_idx = 0; marker_idx < markers.size(); ++marker_idx)
    {
        if (markers[marker_idx].global_id_ < 0)
        {
            unindentified_indices.push_back(MarkerIdx(marker_idx));
            continue;
        }
        const RowIdx row = board.id_to_row_and_col(markers[marker_idx].global_id_)(0);
        row_infos[row].marker_indices.push_back(MarkerIdx(marker_idx));
    }

    for (auto& [row_idx, row_info] : row_infos)
    {
        if (row_info.marker_indices.size() < 2)
        {
            // Use the actual marker in this row, not markers[0]
            const MarkerIdx idx = row_info.marker_indices[0];
            row_info.point_on_line = Eigen::Vector2f(markers[idx].col_, markers[idx].row_);
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
        int adjacent_pairs = 0;
        for (size_t i = 1; i < sorted.size(); ++i)
        {
            const base::MarkerRing& marker_a = markers[sorted[i - 1].second];
            const base::MarkerRing& marker_b = markers[sorted[i].second];
            if (std::abs(marker_a.global_id_ - marker_b.global_id_) != 1)
            {
                continue;
            }
            total_dist += std::hypot(marker_b.col_ - marker_a.col_, marker_b.row_ - marker_a.row_);
            ++adjacent_pairs;
        }
        if (adjacent_pairs > 0)
        {
            row_info.mean_spacing.emplace(total_dist / static_cast<float>(adjacent_pairs));
        }

        row_info.marker_indices.clear();
        for (const auto& [col, idx] : sorted)
        {
            row_info.marker_indices.push_back(idx);
        }
    }

    if (row_infos.empty())
    {
        return;
    }

    if (!unindentified_indices.empty())
    {
        spdlog::debug("Trying to find {} unindentified markers and their lines.", unindentified_indices.size());
        try_fill_missing_rows(row_infos, unindentified_indices, markers, board);
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
        RowIdx closest_row = -1;

        for (const auto& [row, ri] : row_infos)
        {
            if (!ri.direction.has_value())
            {
                continue;
            }
            const Eigen::Vector2f v = pos - ri.point_on_line;
            const float dist = std::abs(v.x() * (-ri.direction->y()) + v.y() * ri.direction->x());
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

        if (closest_marker_idx < 0 || !row_infos[closest_row].mean_spacing.has_value() ||
            row_infos[closest_row].mean_spacing.value() < 1e-6f)
        {
            continue;
        }

        const int col_offset =
            static_cast<int>(std::round(min_marker_dist / row_infos[closest_row].mean_spacing.value()));
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
