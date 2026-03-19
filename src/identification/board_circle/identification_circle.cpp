#include "identification_circle.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <map>
#include <numeric>
#include <random>
#include <set>

#include <spdlog/spdlog.h>
#include <Eigen/Core>
#include <opencv2/calib3d.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video.hpp>

#include "constants.hpp"

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
    std::optional<cv::Point2f> direction;
    cv::Point2f point_on_line;
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
    constexpr float kLineDistanceTolerance = 5.0f;

    // Track which unidentified markers have been assigned
    std::set<MarkerIdx> assigned_markers;

    // ===================================================================================
    // STAGE 1: Assign unidentified markers to existing rows
    // ===================================================================================

    // Compute mean direction and mean spacing from all rows (for fallback)
    cv::Point2f mean_row_direction;
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
    if (const double norm = cv::norm(mean_row_direction); norm > 1e-6)
    {
        mean_row_direction /= norm;
    }
    if (spacing_count > 0)
    {
        global_mean_spacing /= static_cast<float>(spacing_count);
    }

    // Step 1.1 & 1.2: For each unidentified marker, find the closest existing row line
    for (const MarkerIdx marker_idx : unindentified_indices)
    {
        const base::MarkerRing& marker = all_markers[marker_idx];
        const cv::Point2f pos(marker.col_, marker.row_);

        float min_dist = kLineDistanceTolerance;
        RowIdx best_row = -1;

        for (const auto& [row_idx, row_info] : row_infos)
        {
            // Use row's own direction if available, otherwise use mean direction
            cv::Point2f dir;
            if (row_info.direction.has_value())
            {
                dir = row_info.direction.value();
            }
            else if (cv::norm(mean_row_direction) > 1e-6f)
            {
                dir = mean_row_direction;
            }
            else
            {
                continue;  // No valid direction available
            }

            // Compute perpendicular distance to line: |v × direction|
            const cv::Point2f v = pos - row_info.point_on_line;
            const float dist = std::abs(v.x * (-dir.y) + v.y * dir.x);

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

            if (closest_marker_idx < 0 || spacing < 1e-6f)
            {
                continue;
            }

            const int col_offset = static_cast<int>(std::round(min_marker_dist / spacing));
            if (col_offset < 1)
            {
                continue;
            }
            const float direction = (marker.col_ > all_markers[closest_marker_idx].col_) ? 1.0f : -1.0f;
            const int ref_col = board.id_to_row_and_col(all_markers[closest_marker_idx].global_id_)(1);
            const int new_col = ref_col + static_cast<int>(direction * static_cast<float>(col_offset));

            if (new_col < 0 || new_col >= board.cols_)
            {
                continue;
            }

            // Geometric check: verify the direction from reference marker to unidentified marker
            // aligns with the row direction. If mostly perpendicular, the reference is likely
            // in a different row and shouldn't be used for column calculation.
            if (row_info.direction.has_value())
            {
                const cv::Point2f ref_to_marker(marker.col_ - all_markers[closest_marker_idx].col_,
                                                marker.row_ - all_markers[closest_marker_idx].row_);
                const double ref_to_marker_len = cv::norm(ref_to_marker);
                if (ref_to_marker_len > 1e-6)
                {
                    const cv::Point2f ref_to_marker_dir = ref_to_marker / ref_to_marker_len;
                    const float alignment = std::abs(ref_to_marker_dir.dot(row_info.direction.value()));
                    const float kMinAlignment = std::cos(10 / 180.f * pi);  // cos(10 deg) ~ 0.9848
                    if (alignment < kMinAlignment)
                    {
                        spdlog::debug("Stage1: Skipping marker {} - direction alignment {:.2f} < {:.2f}", marker_idx,
                                      alignment, kMinAlignment);
                        continue;
                    }
                }
            }

            const int new_global_id = board.row_and_col_to_id(best_row, new_col);

            all_markers[marker_idx].global_id_ = new_global_id;
            row_infos[best_row].marker_indices.push_back(marker_idx);
            assigned_markers.insert(marker_idx);
            spdlog::debug("Stage1: Assigned marker {} to row={}, col={}", marker_idx, best_row, new_col);
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
    std::vector<cv::Point2f> row_directions;
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
    cv::Point2f mean_direction;
    for (const auto& dir : row_directions)
    {
        mean_direction += dir;
    }
    mean_direction /= cv::norm(mean_direction);
    const cv::Point2f perpendicular(-mean_direction.y, mean_direction.x);

    // Pick a reference point (from the first row with direction)
    const cv::Point2f ref_point = row_infos.at(rows_with_direction[0]).point_on_line;

    // Step 2.3 & 2.4: Compute intersections with perpendicular line and mean offset
    std::vector<std::pair<float, RowIdx>> intersections;  // (distance along perpendicular, row_idx)

    for (const RowIdx row_idx : rows_with_direction)
    {
        const RowInfo& ri = row_infos.at(row_idx);
        // Intersection of perpendicular line (ref_point + t * perpendicular) with row line
        // ri.point_on_line + s * ri.direction = ref_point + t * perpendicular
        // Solve for t: (ri.point_on_line - ref_point) = t * perpendicular - s * ri.direction

        const cv::Point2f diff = ri.point_on_line - ref_point;
        const cv::Point2f& d = ri.direction.value();

        // Using 2D cross product to solve
        const float denom = perpendicular.x * d.y - perpendicular.y * d.x;
        if (std::abs(denom) < 1e-6f)
        {
            continue;
        }

        const float t = (diff.x * d.y - diff.y * d.x) / denom;
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
    cv::Point2f direction_for_candidates = mean_direction;
    if (rows_with_direction.size() >= 2)
    {
        // Could extrapolate, but for simplicity use mean
        direction_for_candidates = mean_direction;
    }

    // Create candidate rows
    std::map<RowIdx, cv::Point2f> candidate_row_points;

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
        const cv::Point2f pos(marker.col_, marker.row_);

        float min_dist = kLineDistanceTolerance;
        RowIdx best_candidate_row = -1;

        for (const auto& [row_idx, point_on_line] : candidate_row_points)
        {
            // Distance to candidate line
            const cv::Point2f v = pos - point_on_line;
            const float dist = std::abs(v.x * (-direction_for_candidates.y) + v.y * direction_for_candidates.x);

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
    cv::Point2f row_direction;
    for (const auto& [row_idx, row_info] : row_infos)
        if (row_info.direction.has_value())
            row_direction += row_info.direction.value();

    if (const double norm = cv::norm(row_direction); norm > 1e-6)
        row_direction /= norm;
    else
        return;  // No valid row directions

    // Helper lambda to compute column positions for a given parity
    const auto compute_col_positions = [&](const std::map<int, std::vector<MarkerIdx>>& markers_by_col)
        -> std::tuple<std::vector<std::pair<float, int>>, cv::Point2f, float>
    {
        if (markers_by_col.empty())
        {
            return {{}, cv::Point2f(), 0.f};
        }

        // Get reference point from first marker
        const auto first_marker = all_markers.at(markers_by_col.begin()->second.front());
        const cv::Point2f ref_point(first_marker.col_, first_marker.row_);

        // For each column, compute mean position projected onto perpendicular direction
        std::vector<std::pair<float, int>> col_positions;
        for (const auto& [col, indices] : markers_by_col)
        {
            float sum_proj = 0.f;
            for (const MarkerIdx idx : indices)
            {
                const cv::Point2f pos(all_markers[idx].col_, all_markers[idx].row_);
                const cv::Point2f diff = pos - ref_point;
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
                total_spacing += (col_positions[i].first - col_positions[i - 1].first) / static_cast<float>(col_diff);
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
        const cv::Point2f pos(marker.col_, marker.row_);

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
        const cv::Point2f diff = pos - ref_point;
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
using circlegrid::MarkerTrack;
using circlegrid::HungarianTrackingResult;
using circlegrid::ORBMotionField;
using circlegrid::BlobVelocityField;

/// Helper: solve similarity model (vCx, vCy, ω, σ) from marker velocities via SVD least-squares.
/// Equations: v_i = v_C + ω × r_i + σ · r_i
///   vx_i = vCx - ω*dy_i + σ*dx_i
///   vy_i = vCy + ω*dx_i + σ*dy_i
/// 4 unknowns, 2 equations per marker → minimum 2 markers.
static bool solve_similarity_2d(const std::vector<cv::Point2f>& positions,
                                 const std::vector<cv::Point2f>& velocities,
                                 const std::vector<size_t>& indices,
                                 int count,
                                 const cv::Point2f& centroid,
                                 float& vCx, float& vCy, float& omega, float& sigma)
{
    cv::Mat A(2 * count, 4, CV_32F);
    cv::Mat b(2 * count, 1, CV_32F);

    for (int i = 0; i < count; ++i)
    {
        const auto& p = positions[indices[i]];
        const auto& v = velocities[indices[i]];
        const float dx = p.x - centroid.x;
        const float dy = p.y - centroid.y;

        // vx_i = vCx - ω*dy + σ*dx
        A.at<float>(2*i, 0)     = 1.f;
        A.at<float>(2*i, 1)     = 0.f;
        A.at<float>(2*i, 2)     = -dy;
        A.at<float>(2*i, 3)     = dx;
        b.at<float>(2*i, 0)     = v.x;

        // vy_i = vCy + ω*dx + σ*dy
        A.at<float>(2*i+1, 0)   = 0.f;
        A.at<float>(2*i+1, 1)   = 1.f;
        A.at<float>(2*i+1, 2)   = dx;
        A.at<float>(2*i+1, 3)   = dy;
        b.at<float>(2*i+1, 0)   = v.y;
    }

    cv::Mat params;
    cv::solve(A, b, params, cv::DECOMP_SVD);
    vCx   = params.at<float>(0);
    vCy   = params.at<float>(1);
    omega = params.at<float>(2);
    sigma = params.at<float>(3);
    return true;
}

/// Compute velocity residual for a single marker given the similarity model
static float velocity_residual(const cv::Point2f& pos, const cv::Point2f& vel,
                                const cv::Point2f& centroid,
                                float vCx, float vCy, float omega, float sigma)
{
    const float dx = pos.x - centroid.x;
    const float dy = pos.y - centroid.y;
    const float pred_vx = vCx - omega * dy + sigma * dx;
    const float pred_vy = vCy + omega * dx + sigma * dy;
    const float ex = vel.x - pred_vx;
    const float ey = vel.y - pred_vy;
    return std::sqrt(ex * ex + ey * ey);
}

/// Fit similarity model (vCx, vCy, ω, σ) from all markers using RANSAC.
/// Uses 8-marker samples (16 eqs, 4 unknowns = heavily overdetermined) for robust hypotheses.
/// Falls back to direct solve if fewer than 8 markers available.
static bool fit_similarity_ransac(const std::vector<cv::Point2f>& positions,
                                   const std::vector<cv::Point2f>& velocities,
                                   const cv::Point2f& centroid,
                                   float inlier_threshold,
                                   int max_iterations,
                                   float& out_vCx, float& out_vCy, float& out_omega,
                                   float& out_sigma,
                                   std::vector<bool>& out_inliers,
                                   float& out_rms)
{
    const int n = static_cast<int>(positions.size());
    if (n < 2) return false;

    out_inliers.assign(n, false);
    out_rms = std::numeric_limits<float>::max();
    out_sigma = 0.f;

    constexpr int kSampleSize = 8;

    // If fewer markers than sample size, solve directly from all (no RANSAC)
    if (n <= kSampleSize)
    {
        std::vector<size_t> all_idx(n);
        std::iota(all_idx.begin(), all_idx.end(), 0);
        solve_similarity_2d(positions, velocities, all_idx, n, centroid,
                            out_vCx, out_vCy, out_omega, out_sigma);
        float rms_sum = 0.f;
        for (int i = 0; i < n; ++i)
        {
            const float r = velocity_residual(positions[i], velocities[i], centroid,
                                               out_vCx, out_vCy, out_omega, out_sigma);
            out_inliers[i] = (r < inlier_threshold);
            rms_sum += r * r;
        }
        out_rms = std::sqrt(rms_sum / static_cast<float>(n));
        return true;
    }

    // RANSAC: sample 8 markers per iteration (16 eqs for 4 unknowns = overdetermined)
    // With 8-marker samples and 80% inlier rate: P(all inliers) = 0.8^8 ≈ 0.17
    // Need ~200 iterations for 99.99% success at 70% inlier rate
    std::mt19937 rng(static_cast<unsigned>(n * 31 + 17));  // deterministic seed
    std::uniform_int_distribution<int> dist(0, n - 1);

    int best_inlier_count = 0;
    float best_vCx = 0.f, best_vCy = 0.f, best_omega = 0.f, best_sigma = 0.f;

    for (int iter = 0; iter < max_iterations; ++iter)
    {
        // Sample kSampleSize distinct markers
        std::vector<size_t> sample;
        sample.reserve(kSampleSize);
        while (static_cast<int>(sample.size()) < kSampleSize)
        {
            const auto idx = static_cast<size_t>(dist(rng));
            if (std::find(sample.begin(), sample.end(), idx) == sample.end())
                sample.push_back(idx);
        }

        float vCx_h, vCy_h, omega_h, sigma_h;
        solve_similarity_2d(positions, velocities, sample, kSampleSize, centroid,
                            vCx_h, vCy_h, omega_h, sigma_h);

        int inlier_count = 0;
        for (int j = 0; j < n; ++j)
        {
            if (velocity_residual(positions[j], velocities[j], centroid,
                                   vCx_h, vCy_h, omega_h, sigma_h) < inlier_threshold)
            {
                ++inlier_count;
            }
        }

        if (inlier_count > best_inlier_count)
        {
            best_inlier_count = inlier_count;
            best_vCx = vCx_h;
            best_vCy = vCy_h;
            best_omega = omega_h;
            best_sigma = sigma_h;
        }
    }

    // Collect inlier indices and refit
    std::vector<size_t> inlier_indices;
    inlier_indices.reserve(best_inlier_count);
    for (int j = 0; j < n; ++j)
    {
        if (velocity_residual(positions[j], velocities[j], centroid,
                               best_vCx, best_vCy, best_omega, best_sigma) < inlier_threshold)
        {
            inlier_indices.push_back(static_cast<size_t>(j));
        }
    }

    if (inlier_indices.size() < 2) return false;

    // Refit from all inliers
    solve_similarity_2d(positions, velocities, inlier_indices,
                        static_cast<int>(inlier_indices.size()), centroid,
                        out_vCx, out_vCy, out_omega, out_sigma);

    // Compute per-marker residuals and RMS
    float rms_sum = 0.f;
    int inlier_count_final = 0;
    for (int j = 0; j < n; ++j)
    {
        const float r = velocity_residual(positions[j], velocities[j], centroid,
                                           out_vCx, out_vCy, out_omega, out_sigma);
        out_inliers[j] = (r < inlier_threshold);
        if (out_inliers[j])
        {
            rms_sum += r * r;
            ++inlier_count_final;
        }
    }
    out_rms = inlier_count_final > 0
        ? std::sqrt(rms_sum / static_cast<float>(inlier_count_final))
        : 0.f;
    return true;
}

/// Evaluate the pre-fitted global similarity model at a query point.
cv::Point2f BlobVelocityField::transport_predict(const cv::Point2f& query, float dt) const
{
    if (!valid) return {0.f, 0.f};

    const float rqx = query.x - centroid.x;
    const float rqy = query.y - centroid.y;

    // v_q = v_C + ω × r_q + σ · r_q
    const float vqx = vCx - omega * rqy + sigma * rqx;
    const float vqy = vCy + omega * rqx + sigma * rqy;

    float aqx = 0.f, aqy = 0.f;
    if (has_acceleration)
    {
        // a_q = a_C + ε × r_q + σ_dot · r_q - (ω² - σ²) · r_q
        // The (ω²-σ²) term combines centripetal and scale effects
        const float omega_sq_minus_sigma_sq = omega * omega - sigma * sigma;
        aqx = aCx - epsilon * rqy + sigma_dot * rqx - omega_sq_minus_sigma_sq * rqx;
        aqy = aCy + epsilon * rqx + sigma_dot * rqy - omega_sq_minus_sigma_sq * rqy;
    }

    return {vqx * dt + 0.5f * aqx * dt * dt,
            vqy * dt + 0.5f * aqy * dt * dt};
}

void TrackingState::update(const std::vector<base::MarkerCoding>& markers, const std::vector<int>& global_ids,
                           const cv::Mat1b& image)
{
    update_tracks(markers, global_ids);
    update_blob_velocity_fields(markers);
    prev_markers_ = markers;
    prev_global_ids_ = global_ids;
    prev_image_ = image.clone();
    has_previous_ = true;

    // Store current blob positions as prev for next frame
    prev_blob_positions_.clear();
    prev_blob_positions_.reserve(markers.size());
    for (const auto& m : markers)
        prev_blob_positions_.emplace_back(m.col_, m.row_);

    // Store all detected positions for velocity interpolation (3-frame rolling buffer)
    detected_positions_history_[history_write_idx_ % 3] = prev_blob_positions_;
    ++history_write_idx_;

    ++frame_counter_;
}

/// Helper: fit global similarity model to a BlobVelocityField (velocity + optional acceleration).
/// After fitting, outlier velocities are replaced with model-predicted values (smoothing).
static void fit_global_model(BlobVelocityField& field)
{
    const int n = static_cast<int>(field.positions.size());
    if (n < 2) { field.valid = false; return; }

    // Compute centroid
    field.centroid = {0.f, 0.f};
    for (const auto& p : field.positions) field.centroid += p;
    field.centroid /= static_cast<float>(n);

    // RANSAC fit for similarity velocity model: (vCx, vCy, ω, σ)
    // 8-marker samples need ~200 iterations for 99% success at 70% inlier rate
    const bool ok = fit_similarity_ransac(
        field.positions, field.velocities, field.centroid,
        3.0f,  // inlier threshold (px)
        200,   // max iterations (8-marker samples)
        field.vCx, field.vCy, field.omega, field.sigma,
        field.velocity_inliers, field.velocity_residual_rms);

    if (!ok) { field.valid = false; return; }
    field.velocity_inlier_count = static_cast<int>(
        std::count(field.velocity_inliers.begin(), field.velocity_inliers.end(), true));
    field.valid = field.velocity_inlier_count >= 2;

    // Smooth outlier velocities: replace with model-predicted values.
    // This removes motion peaks from wrong track assignments.
    int smoothed_count = 0;
    for (int i = 0; i < n; ++i)
    {
        if (field.velocity_inliers[i]) continue;

        const float dx = field.positions[i].x - field.centroid.x;
        const float dy = field.positions[i].y - field.centroid.y;
        const cv::Point2f model_vel(field.vCx - field.omega * dy + field.sigma * dx,
                                     field.vCy + field.omega * dx + field.sigma * dy);
        const float residual = static_cast<float>(cv::norm(field.velocities[i] - model_vel));
        field.velocities[i] = model_vel;
        field.velocity_inliers[i] = true;
        ++smoothed_count;

        spdlog::debug("Motion field: smoothed outlier idx={} residual={:.1f}px → model vel=({:.1f},{:.1f})",
                       i, residual, model_vel.x, model_vel.y);
    }
    if (smoothed_count > 0)
    {
        field.velocity_inlier_count = n;
        float rms_sum = 0.f;
        for (int i = 0; i < n; ++i)
        {
            const float r = velocity_residual(field.positions[i], field.velocities[i],
                                               field.centroid, field.vCx, field.vCy,
                                               field.omega, field.sigma);
            rms_sum += r * r;
        }
        field.velocity_residual_rms = std::sqrt(rms_sum / static_cast<float>(n));
    }

    // Acceleration fit: use markers with nonzero acceleration
    // Remove centripetal+scale terms, then RANSAC fit (aCx, aCy, ε, σ_dot)
    if (!field.accelerations.empty() && field.accelerations.size() == field.positions.size())
    {
        std::vector<cv::Point2f> acc_positions;
        std::vector<cv::Point2f> acc_corrected;
        const float omega_sq_minus_sigma_sq = field.omega * field.omega - field.sigma * field.sigma;
        for (int i = 0; i < n; ++i)
        {
            if (cv::norm(field.accelerations[i]) < 0.01f) continue;

            const float rx = field.positions[i].x - field.centroid.x;
            const float ry = field.positions[i].y - field.centroid.y;
            // Remove centripetal+scale: a_corr = a_measured + (ω²-σ²)·r
            acc_positions.push_back(field.positions[i]);
            acc_corrected.push_back(field.accelerations[i]
                + cv::Point2f(omega_sq_minus_sigma_sq * rx, omega_sq_minus_sigma_sq * ry));
        }

        if (static_cast<int>(acc_positions.size()) >= 5)
        {
            std::vector<bool> acc_inliers;
            float acc_rms;
            float acc_sigma;  // this is σ_dot for acceleration
            const bool acc_ok = fit_similarity_ransac(
                acc_positions, acc_corrected, field.centroid,
                5.0f,  // slightly larger threshold for acceleration noise
                100,   // 8-marker samples
                field.aCx, field.aCy, field.epsilon, acc_sigma,
                acc_inliers, acc_rms);
            field.sigma_dot = acc_sigma;
            field.has_acceleration = acc_ok;

            // Smooth outlier accelerations with model-predicted values
            if (acc_ok)
            {
                for (int i = 0; i < n; ++i)
                {
                    if (cv::norm(field.accelerations[i]) < 0.01f) continue;
                    const float rx = field.positions[i].x - field.centroid.x;
                    const float ry = field.positions[i].y - field.centroid.y;
                    const cv::Point2f a_corr = field.accelerations[i]
                        + cv::Point2f(omega_sq_minus_sigma_sq * rx, omega_sq_minus_sigma_sq * ry);
                    const cv::Point2f model_a_corr(field.aCx - field.epsilon * ry + field.sigma_dot * rx,
                                                    field.aCy + field.epsilon * rx + field.sigma_dot * ry);
                    if (cv::norm(a_corr - model_a_corr) > 5.0f)
                    {
                        field.accelerations[i] = cv::Point2f(
                            model_a_corr.x - omega_sq_minus_sigma_sq * rx,
                            model_a_corr.y - omega_sq_minus_sigma_sq * ry);
                    }
                }
            }
        }
    }
}

void TrackingState::update_blob_velocity_fields(const std::vector<base::MarkerCoding>& curr_markers)
{
    forward_blob_field_ = {};
    backward_blob_field_ = {};

    if (!has_previous_ || prev_global_ids_.empty()) return;

    // Collect per-marker velocities from identified tracks.
    // Each track that was seen on consecutive frames gives an exact velocity.
    for (const auto& [gid, track] : tracks_)
    {
        if (track.history_count < 2) continue;

        const cv::Point2f prev_pos = track.position_history[track.history_count - 2];
        const cv::Point2f curr_pos = track.position_history[track.history_count - 1];
        const int curr_frame = track.frame_ids[track.history_count - 1];

        // Only use recent tracks
        if (frame_counter_ - curr_frame > 2) continue;

        const cv::Point2f vel = curr_pos - prev_pos;  // velocity (px/frame, dt=1)

        // Acceleration from central difference: a(t-1) = [s(t) - 2s(t-1) + s(t-2)] / dt²
        cv::Point2f acc(0.f, 0.f);
        if (track.history_count >= 3)
        {
            const cv::Point2f pp_pos = track.position_history[track.history_count - 3];
            acc = curr_pos - 2.f * prev_pos + pp_pos;
        }

        // Forward field: anchored at previous position
        forward_blob_field_.positions.push_back(prev_pos);
        forward_blob_field_.velocities.push_back(vel);
        forward_blob_field_.accelerations.push_back(acc);

        // Backward field: anchored at current position
        backward_blob_field_.positions.push_back(curr_pos);
        backward_blob_field_.velocities.push_back(vel);
        backward_blob_field_.accelerations.push_back(acc);
    }

    // Fit global rigid-body model via RANSAC for both fields
    fit_global_model(forward_blob_field_);
    fit_global_model(backward_blob_field_);

    if (forward_blob_field_.valid)
    {
        spdlog::debug("Blob velocity field: {} markers, inl={}/{}, v=({:.1f},{:.1f}), "
                      "ω={:.4f}rad/f, σ={:.5f}/f, rms={:.2f}px, acc={}",
                      forward_blob_field_.positions.size(),
                      forward_blob_field_.velocity_inlier_count,
                      forward_blob_field_.positions.size(),
                      forward_blob_field_.vCx, forward_blob_field_.vCy,
                      forward_blob_field_.omega, forward_blob_field_.sigma,
                      forward_blob_field_.velocity_residual_rms,
                      forward_blob_field_.has_acceleration ? "yes" : "no");
    }
}

void TrackingState::clear()
{
    prev_markers_.clear();
    prev_global_ids_.clear();
    prev_image_.release();
    has_previous_ = false;
    tracks_.clear();
    orientation_locked_ = false;
    frame_counter_ = 0;
    prev_findcircles_centers_.clear();
    prev_orb_keypoints_.clear();
    prev_orb_descriptors_.release();
}

void TrackingState::update_orb(const cv::Mat1b& image)
{
    if (image.empty()) return;
    auto orb = cv::ORB::create(500);
    orb->detectAndCompute(image, cv::noArray(), prev_orb_keypoints_, prev_orb_descriptors_);
}

ORBMotionField TrackingState::estimate_motion_field(const cv::Mat1b& current_image) const
{
    ORBMotionField field;
    if (prev_orb_descriptors_.empty() || current_image.empty()) return field;

    auto orb = cv::ORB::create(500);
    std::vector<cv::KeyPoint> curr_keypoints;
    cv::Mat curr_descriptors;
    orb->detectAndCompute(current_image, cv::noArray(), curr_keypoints, curr_descriptors);

    if (curr_descriptors.empty()) return field;

    cv::BFMatcher matcher(cv::NORM_HAMMING);
    std::vector<std::vector<cv::DMatch>> knn_matches;
    matcher.knnMatch(prev_orb_descriptors_, curr_descriptors, knn_matches, 2);

    // Lowe's ratio test
    for (const auto& m : knn_matches)
    {
        if (m.size() >= 2 && m[0].distance < 0.75f * m[1].distance)
        {
            const auto& prev_pt = prev_orb_keypoints_[m[0].queryIdx].pt;
            const auto& curr_pt = curr_keypoints[m[0].trainIdx].pt;
            field.locations.push_back(curr_pt);
            field.displacements.push_back(curr_pt - prev_pt);
        }
    }

    field.valid = field.locations.size() >= 8;
    if (field.valid)
    {
        spdlog::debug("ORB motion field: {} matched features", field.locations.size());
    }
    return field;
}

cv::Point2f TrackingState::predict_position_with_orb(
    const int global_id, const int current_frame, const ORBMotionField& motion_field) const
{
    cv::Point2f base = predict_position(global_id, current_frame);

    auto it = tracks_.find(global_id);
    if (it == tracks_.end()) return base;
    const auto& track = it->second;

    // If track is recent (<= 2 frames old), standard prediction is fine
    if (current_frame - track.last_seen_frame <= 2 && track.history_count >= 2)
        return base;

    if (!motion_field.valid || motion_field.locations.empty())
        return base;

    // Find 4 nearest ORB features within 10% of shorter image edge
    const float max_radius = std::min(
        static_cast<float>(prev_image_.cols),
        static_cast<float>(prev_image_.rows)) * 0.1f;

    struct Neighbor { float dist; cv::Point2f displacement; };
    std::vector<Neighbor> neighbors;
    const cv::Point2f query = track.last_position;

    for (size_t i = 0; i < motion_field.locations.size(); ++i)
    {
        const float d = static_cast<float>(cv::norm(motion_field.locations[i] - query));
        if (d < max_radius)
            neighbors.push_back({d, motion_field.displacements[i]});
    }

    if (neighbors.size() < 2) return base;

    std::sort(neighbors.begin(), neighbors.end(),
              [](const auto& a, const auto& b) { return a.dist < b.dist; });
    const int k = std::min(4, static_cast<int>(neighbors.size()));

    // Inverse-distance weighted interpolation
    cv::Point2f weighted_disp(0.f, 0.f);
    float w_sum = 0.f;
    for (int i = 0; i < k; ++i)
    {
        const float w = 1.f / (neighbors[i].dist + 1e-6f);
        weighted_disp += neighbors[i].displacement * w;
        w_sum += w;
    }
    weighted_disp /= w_sum;

    const int gap = current_frame - track.last_seen_frame;
    return track.last_position + weighted_disp * static_cast<float>(gap);
}

void TrackingState::update_tracks(const std::vector<base::MarkerCoding>& markers, const std::vector<int>& global_ids)
{
    for (size_t i = 0; i < markers.size(); ++i)
    {
        if (i >= global_ids.size() || global_ids[i] < 0)
        {
            continue;
        }
        const int gid = global_ids[i];
        const cv::Point2f curr_pos(markers[i].col_, markers[i].row_);

        auto it = tracks_.find(gid);
        if (it != tracks_.end())
        {
            auto& track = it->second;
            // Shift history buffer
            if (track.history_count < 3)
            {
                track.position_history[track.history_count] = curr_pos;
                track.frame_ids[track.history_count] = frame_counter_;
                ++track.history_count;
            }
            else
            {
                track.position_history[0] = track.position_history[1];
                track.position_history[1] = track.position_history[2];
                track.position_history[2] = curr_pos;
                track.frame_ids[0] = track.frame_ids[1];
                track.frame_ids[1] = track.frame_ids[2];
                track.frame_ids[2] = frame_counter_;
            }
            track.last_position = curr_pos;
            track.last_seen_frame = frame_counter_;
            ++track.age;
        }
        else
        {
            MarkerTrack track;
            track.position_history[0] = curr_pos;
            track.frame_ids[0] = frame_counter_;
            track.history_count = 1;
            track.last_position = curr_pos;
            track.global_id = gid;
            track.last_seen_frame = frame_counter_;
            track.age = 1;
            tracks_[gid] = track;
        }
    }

    // Prune tracks not seen for >10 frames
    std::erase_if(tracks_, [this](const auto& pair) { return frame_counter_ - pair.second.last_seen_frame > 10; });
}

cv::Point2f TrackingState::predict_position(const int global_id, const int current_frame) const
{
    auto it = tracks_.find(global_id);
    if (it == tracks_.end())
    {
        return {0.f, 0.f};
    }
    const auto& track = it->second;

    if (track.history_count >= 3)
    {
        // Least-squares velocity from 3 points: v = sum((p_i - p_mean) * (t_i - t_mean)) / sum((t_i - t_mean)^2)
        const float t0 = static_cast<float>(track.frame_ids[0]);
        const float t1 = static_cast<float>(track.frame_ids[1]);
        const float t2 = static_cast<float>(track.frame_ids[2]);
        const float t_mean = (t0 + t1 + t2) / 3.f;

        const cv::Point2f p_mean = (track.position_history[0] + track.position_history[1] + track.position_history[2]) / 3.f;

        float sum_tt = 0.f;
        cv::Point2f sum_tp(0.f, 0.f);
        for (int k = 0; k < 3; ++k)
        {
            const float dt = static_cast<float>(track.frame_ids[k]) - t_mean;
            sum_tt += dt * dt;
            sum_tp += (track.position_history[k] - p_mean) * dt;
        }

        if (sum_tt > 1e-6f)
        {
            const cv::Point2f velocity = sum_tp / sum_tt;
            const float dt = static_cast<float>(current_frame) - t2;
            return track.position_history[2] + velocity * dt;
        }
    }
    else if (track.history_count == 2)
    {
        const float dt_hist = static_cast<float>(track.frame_ids[1] - track.frame_ids[0]);
        if (dt_hist > 0.f)
        {
            const cv::Point2f velocity = (track.position_history[1] - track.position_history[0]) / dt_hist;
            const float dt = static_cast<float>(current_frame - track.frame_ids[1]);
            return track.position_history[1] + velocity * dt;
        }
    }

    return track.last_position;
}

// --- Hungarian Algorithm (Jonker-Volgenant) ---

std::vector<int> circlegrid::hungarian_assignment(const std::vector<std::vector<float>>& cost_matrix, float max_cost)
{
    if (cost_matrix.empty())
    {
        return {};
    }

    const int n_rows = static_cast<int>(cost_matrix.size());
    const int n_cols = static_cast<int>(cost_matrix[0].size());
    const int n = std::max(n_rows, n_cols);

    // Pad to square
    std::vector<std::vector<float>> c(n, std::vector<float>(n, max_cost));
    for (int i = 0; i < n_rows; ++i)
    {
        for (int j = 0; j < n_cols; ++j)
        {
            c[i][j] = cost_matrix[i][j];
        }
    }

    // JV algorithm with successive shortest paths
    constexpr float kInf = 1e18f;
    std::vector<float> u(n + 1, 0.f), v(n + 1, 0.f);
    std::vector<int> p(n + 1, 0), way(n + 1, 0);

    for (int i = 1; i <= n; ++i)
    {
        p[0] = i;
        int j0 = 0;
        std::vector<float> minv(n + 1, kInf);
        std::vector<bool> used(n + 1, false);

        do
        {
            used[j0] = true;
            int i0 = p[j0];
            float delta = kInf;
            int j1 = -1;

            for (int j = 1; j <= n; ++j)
            {
                if (!used[j])
                {
                    const float cur = c[i0 - 1][j - 1] - u[i0] - v[j];
                    if (cur < minv[j])
                    {
                        minv[j] = cur;
                        way[j] = j0;
                    }
                    if (minv[j] < delta)
                    {
                        delta = minv[j];
                        j1 = j;
                    }
                }
            }

            for (int j = 0; j <= n; ++j)
            {
                if (used[j])
                {
                    u[p[j]] += delta;
                    v[j] -= delta;
                }
                else
                {
                    minv[j] -= delta;
                }
            }

            j0 = j1;
        } while (p[j0] != 0);

        do
        {
            const int j1 = way[j0];
            p[j0] = p[j1];
            j0 = j1;
        } while (j0);
    }

    // Extract assignment: row i -> col result[i]
    std::vector<int> result(n_rows, -1);
    for (int j = 1; j <= n; ++j)
    {
        if (p[j] > 0 && p[j] <= n_rows)
        {
            const int row = p[j] - 1;
            const int col = j - 1;
            if (col < n_cols && cost_matrix[row][col] <= max_cost)
            {
                result[row] = col;
            }
        }
    }
    return result;
}

// --- Velocity-Predicted Hungarian Tracking ---

HungarianTrackingResult circlegrid::identify_with_hungarian_tracking(const TrackingState& state,
                                                                     const std::vector<base::MarkerCoding>& curr_markers,
                                                                     const BoardCircleGrid& board,
                                                                     float max_distance, float ransac_threshold,
                                                                     const ORBMotionField* motion_field)
{
    HungarianTrackingResult result;
    result.global_ids.assign(curr_markers.size(), -1);

    // Collect all tracked global_ids
    std::vector<int> prev_ids;
    std::vector<cv::Point2f> predicted_positions;
    for (const auto& [gid, track] : state.tracks_)
    {
        if (track.age > 0)
        {
            prev_ids.push_back(gid);
            if (motion_field && motion_field->valid)
                predicted_positions.push_back(state.predict_position_with_orb(gid, state.frame_counter_, *motion_field));
            else
                predicted_positions.push_back(state.predict_position(gid, state.frame_counter_));
        }
    }

    if (prev_ids.empty() || curr_markers.empty())
    {
        return result;
    }

    const int n_prev = static_cast<int>(prev_ids.size());
    const int n_curr = static_cast<int>(curr_markers.size());

    // Build cost matrix
    std::vector<std::vector<float>> cost(n_prev, std::vector<float>(n_curr));
    for (int i = 0; i < n_prev; ++i)
    {
        for (int j = 0; j < n_curr; ++j)
        {
            const cv::Point2f curr_pos(curr_markers[j].col_, curr_markers[j].row_);
            cost[i][j] = static_cast<float>(cv::norm(predicted_positions[i] - curr_pos));
        }
    }

    // Solve
    const auto assignment = hungarian_assignment(cost, max_distance);

    // Extract matches for RANSAC validation
    std::vector<cv::Point2f> src_pts, dst_pts;
    std::vector<std::pair<int, int>> matches;  // (prev_global_id, curr_idx)
    for (int i = 0; i < n_prev; ++i)
    {
        if (assignment[i] >= 0)
        {
            const auto& track = state.tracks_.at(prev_ids[i]);
            src_pts.push_back(track.last_position);
            dst_pts.emplace_back(curr_markers[assignment[i]].col_, curr_markers[assignment[i]].row_);
            matches.emplace_back(prev_ids[i], assignment[i]);
        }
    }

    // RANSAC validation
    if (matches.size() >= 4)
    {
        std::vector<uchar> inlier_mask;
        const cv::Mat H = cv::findHomography(src_pts, dst_pts, cv::RANSAC, ransac_threshold, inlier_mask);
        if (!H.empty())
        {
            float total_cost = 0.f;
            float max_cost_val = 0.f;
            int count = 0;
            for (size_t k = 0; k < matches.size(); ++k)
            {
                if (inlier_mask[k])
                {
                    const int curr_idx = matches[k].second;
                    result.global_ids[curr_idx] = matches[k].first;
                    const float c = cost[static_cast<int>(
                        std::find(prev_ids.begin(), prev_ids.end(), matches[k].first) - prev_ids.begin())][curr_idx];
                    total_cost += c;
                    max_cost_val = std::max(max_cost_val, c);
                    ++count;
                }
            }
            result.matched_count = count;
            result.avg_cost = count > 0 ? total_cost / static_cast<float>(count) : 0.f;
            result.max_cost = max_cost_val;

            // Bidirectional blob-velocity acceptance check.
            // Only apply on near-square grids where row/col swaps actually occur.
            // Non-square grids (5x7, aspect>1.3) don't suffer from orientation ambiguity.
            const float board_width = board.is_asymetric_
                ? static_cast<float>((2 * (board.cols_ - 1) + 1) * board.spacing_)
                : static_cast<float>((board.cols_ - 1) * board.spacing_);
            const float board_height = static_cast<float>((board.rows_ - 1) * board.spacing_);
            const float board_aspect = std::max(board_width, board_height)
                / std::max(1.f, std::min(board_width, board_height));
            const bool board_is_near_square = board_aspect < 1.3f;

            const float img_short_edge = static_cast<float>(
                std::min(state.prev_image_.cols, state.prev_image_.rows));
            const float absolute_cap = 0.05f * img_short_edge;

            if (board_is_near_square &&
                state.forward_blob_field_.valid && state.backward_blob_field_.valid && absolute_cap > 0.f)
            {
                for (size_t k = 0; k < matches.size(); ++k)
                {
                    if (!inlier_mask[k]) continue;
                    const int gid = matches[k].first;
                    const int curr_idx = matches[k].second;
                    if (result.global_ids[curr_idx] != gid) continue;

                    auto track_it = state.tracks_.find(gid);
                    if (track_it == state.tracks_.end() || track_it->second.history_count < 1)
                        continue;

                    const cv::Point2f prev_pos = track_it->second.last_position;
                    const cv::Point2f curr_pos(curr_markers[curr_idx].col_, curr_markers[curr_idx].row_);

                    // Forward check: predict current from previous
                    const cv::Point2f v_fwd = state.forward_blob_field_.transport_predict(prev_pos);
                    const cv::Point2f predicted_curr = prev_pos + v_fwd;
                    const float fwd_err = static_cast<float>(cv::norm(predicted_curr - curr_pos));

                    // Backward check: predict previous from current
                    const cv::Point2f v_bwd = state.backward_blob_field_.transport_predict(curr_pos);
                    const cv::Point2f predicted_prev = curr_pos - v_bwd;
                    const float bwd_err = static_cast<float>(cv::norm(predicted_prev - prev_pos));

                    // Displacement-proportional tolerance: prediction error scales with
                    // motion speed (acceleration, lens distortion, direction changes).
                    // Allow 50% of predicted displacement, minimum 15px, capped at 5% edge.
                    const float displacement = static_cast<float>(cv::norm(v_fwd));
                    const float tolerance = std::min(std::max(0.5f * displacement, 15.f), absolute_cap);

                    if (fwd_err > tolerance || bwd_err > tolerance)
                    {
                        spdlog::info("Blob-velocity reject: gid {} fwd_err={:.1f} bwd_err={:.1f} "
                                     "(tol={:.1f} disp={:.1f}) prev=({:.0f},{:.0f}) curr=({:.0f},{:.0f})",
                                     gid, fwd_err, bwd_err, tolerance, displacement,
                                     prev_pos.x, prev_pos.y, curr_pos.x, curr_pos.y);
                        result.global_ids[curr_idx] = -1;
                        --result.matched_count;
                    }
                }
            }
        }
    }
    else if (!matches.empty())
    {
        // Too few for RANSAC, accept all
        float total_cost = 0.f;
        for (const auto& [gid, curr_idx] : matches)
        {
            result.global_ids[curr_idx] = gid;
            const float c = cost[static_cast<int>(std::find(prev_ids.begin(), prev_ids.end(), gid) - prev_ids.begin())]
                                [curr_idx];
            total_cost += c;
            result.max_cost = std::max(result.max_cost, c);
        }
        result.matched_count = static_cast<int>(matches.size());
        result.avg_cost = total_cost / static_cast<float>(matches.size());
    }

    spdlog::debug("Hungarian tracking: {} matched of {} prev, {} curr, avg_cost={:.1f}, max_cost={:.1f}",
                  result.matched_count, n_prev, n_curr, result.avg_cost, result.max_cost);
    return result;
}

// --- 180-Degree Ambiguity ---

bool circlegrid::board_has_180_ambiguity(const BoardCircleGrid& board)
{
    if (board.is_asymetric_)
    {
        return board.rows_ % 2 == 0;
    }
    return (board.rows_ % 2 == 0) && (board.cols_ % 2 == 0);
}

std::vector<int> circlegrid::flip_ids_180(const std::vector<int>& global_ids, const BoardCircleGrid& board)
{
    std::vector<int> flipped(global_ids.size(), -1);
    for (size_t i = 0; i < global_ids.size(); ++i)
    {
        if (global_ids[i] < 0)
        {
            continue;
        }
        const auto rc = board.id_to_row_and_col(global_ids[i]);
        const int flipped_row = board.rows_ - 1 - rc(0);
        const int flipped_col = board.cols_ - 1 - rc(1);
        flipped[i] = board.row_and_col_to_id(flipped_row, flipped_col);
    }
    return flipped;
}

std::vector<int> circlegrid::resolve_180_ambiguity(const std::vector<int>& global_ids,
                                                    const std::vector<base::MarkerCoding>& curr_markers,
                                                    const TrackingState& state, const BoardCircleGrid& board)
{
    if (!board_has_180_ambiguity(board) || state.orientation_locked_)
    {
        return global_ids;
    }

    if (state.tracks_.empty())
    {
        return global_ids;
    }

    // Compute velocity variance for original and flipped
    auto compute_velocity_variance = [&](const std::vector<int>& ids) -> float
    {
        std::vector<cv::Point2f> velocities;
        for (size_t i = 0; i < ids.size(); ++i)
        {
            if (ids[i] < 0)
            {
                continue;
            }
            auto it = state.tracks_.find(ids[i]);
            if (it == state.tracks_.end() || it->second.age < 1)
            {
                continue;
            }
            const cv::Point2f curr_pos(curr_markers[i].col_, curr_markers[i].row_);
            velocities.push_back(curr_pos - it->second.last_position);
        }

        if (velocities.size() < 3)
        {
            return std::numeric_limits<float>::max();
        }

        cv::Point2f mean_vel(0.f, 0.f);
        for (const auto& v : velocities)
        {
            mean_vel += v;
        }
        mean_vel /= static_cast<float>(velocities.size());

        float variance = 0.f;
        for (const auto& v : velocities)
        {
            const cv::Point2f diff = v - mean_vel;
            variance += diff.x * diff.x + diff.y * diff.y;
        }
        return variance / static_cast<float>(velocities.size());
    };

    const auto flipped_ids = flip_ids_180(global_ids, board);
    const float orig_var = compute_velocity_variance(global_ids);
    const float flip_var = compute_velocity_variance(flipped_ids);

    spdlog::debug("180-ambiguity: orig_var={:.1f}, flip_var={:.1f}", orig_var, flip_var);

    if (flip_var < orig_var * 0.5f)
    {
        spdlog::info("180-ambiguity: FLIPPED (flip_var={:.1f} < orig_var={:.1f} * 0.5)", flip_var, orig_var);
        return flipped_ids;
    }
    return global_ids;
}

// --- Local Homography Re-identification ---

void circlegrid::identify_unmatched_by_local_homography(std::vector<base::MarkerRing>& markers,
                                                         const BoardCircleGrid& board)
{
    // Collect identified markers with their board coordinates
    std::vector<cv::Point2f> board_pts;
    std::vector<cv::Point2f> image_pts;
    std::vector<int> identified_indices;
    std::set<int> used_global_ids;

    for (size_t i = 0; i < markers.size(); ++i)
    {
        if (markers[i].global_id_ >= 0)
        {
            const auto rc = board.id_to_row_and_col(markers[i].global_id_);
            const int row = rc(0);
            const int col = rc(1);

            float bx, by;
            if (board.is_asymetric_)
            {
                bx = static_cast<float>((2 * col + row % 2) * board.spacing_);
                by = static_cast<float>(row * board.spacing_);
            }
            else
            {
                bx = static_cast<float>(col * board.spacing_);
                by = static_cast<float>(row * board.spacing_);
            }

            board_pts.emplace_back(bx, by);
            image_pts.emplace_back(markers[i].col_, markers[i].row_);
            identified_indices.push_back(static_cast<int>(i));
            used_global_ids.insert(markers[i].global_id_);
        }
    }

    // Require sufficient seed markers for reliable homography.
    // Near-square grids need more seeds because the homography is prone to
    // fitting swapped configurations. Non-square grids are safer with fewer seeds.
    const float bw = board.is_asymetric_
        ? static_cast<float>((2 * (board.cols_ - 1) + 1) * board.spacing_)
        : static_cast<float>((board.cols_ - 1) * board.spacing_);
    const float bh = static_cast<float>((board.rows_ - 1) * board.spacing_);
    const float board_aspect = std::max(bw, bh) / std::max(1.f, std::min(bw, bh));
    const int total_markers = board.rows_ * board.cols_;
    // Near-square: 25% of total (seed validation catches bad IDs via RANSAC reproj).
    // Non-square: fixed 8.
    const int min_seeds = board_aspect < 1.3f
        ? std::max(8, static_cast<int>(total_markers * 0.25f))
        : 8;
    if (static_cast<int>(board_pts.size()) < min_seeds)
    {
        spdlog::debug("Local homography: only {} identified markers, need {}+ (aspect={:.1f})",
                       board_pts.size(), min_seeds, board_aspect);
        return;
    }

    // Compute global RANSAC homography: board -> image
    cv::Mat H_global = cv::findHomography(board_pts, image_pts, cv::RANSAC, 5.0);
    if (H_global.empty())
    {
        spdlog::debug("Local homography: global homography failed");
        return;
    }

    // Validate seed markers: compute reproj error for each identified marker against H.
    // Remove seeds with high reproj error (these have wrong gids and corrupt H).
    {
        std::vector<float> seed_reproj(board_pts.size());
        for (size_t j = 0; j < board_pts.size(); ++j)
        {
            const cv::Mat pt = (cv::Mat_<double>(3, 1) << board_pts[j].x, board_pts[j].y, 1.0);
            const cv::Mat proj = H_global * pt;
            const cv::Point2f projected(static_cast<float>(proj.at<double>(0) / proj.at<double>(2)),
                                         static_cast<float>(proj.at<double>(1) / proj.at<double>(2)));
            seed_reproj[j] = static_cast<float>(cv::norm(projected - image_pts[j]));
        }

        // Compute median reproj for threshold
        std::vector<float> sorted_reproj = seed_reproj;
        std::sort(sorted_reproj.begin(), sorted_reproj.end());
        const float median_reproj = sorted_reproj[sorted_reproj.size() / 2];
        const float outlier_threshold = std::max(3.f * median_reproj, 5.f);

        // Remove outlier seeds and rebuild arrays
        std::vector<cv::Point2f> clean_board, clean_image;
        std::vector<int> clean_indices;
        int removed = 0;
        for (size_t j = 0; j < board_pts.size(); ++j)
        {
            if (seed_reproj[j] <= outlier_threshold)
            {
                clean_board.push_back(board_pts[j]);
                clean_image.push_back(image_pts[j]);
                clean_indices.push_back(identified_indices[j]);
            }
            else
            {
                // Unset the bad seed marker
                markers[identified_indices[j]].global_id_ = -1;
                used_global_ids.erase(markers[identified_indices[j]].global_id_);
                ++removed;
            }
        }

        if (removed > 0)
        {
            spdlog::info("Local homography: removed {} outlier seed markers (median_reproj={:.1f}px, threshold={:.1f}px)",
                          removed, median_reproj, outlier_threshold);
            board_pts = std::move(clean_board);
            image_pts = std::move(clean_image);
            identified_indices = std::move(clean_indices);

            if (static_cast<int>(board_pts.size()) < min_seeds)
            {
                spdlog::debug("Local homography: only {} clean seeds remaining, need 8+", board_pts.size());
                return;
            }

            // Recompute H from cleaned seeds
            H_global = cv::findHomography(board_pts, image_pts, cv::RANSAC, 5.0);
            if (H_global.empty())
            {
                spdlog::debug("Local homography: recomputed homography failed");
                return;
            }
        }
    }

    // Pre-compute all expected board positions
    struct BoardPosition
    {
        cv::Point2f board_pt;
        int global_id;
    };
    std::vector<BoardPosition> all_board_positions;
    for (int r = 0; r < board.rows_; ++r)
    {
        for (int c = 0; c < board.cols_; ++c)
        {
            const int gid = board.row_and_col_to_id(r, c);
            float bx, by;
            if (board.is_asymetric_)
            {
                bx = static_cast<float>((2 * c + r % 2) * board.spacing_);
                by = static_cast<float>(r * board.spacing_);
            }
            else
            {
                bx = static_cast<float>(c * board.spacing_);
                by = static_cast<float>(r * board.spacing_);
            }
            all_board_positions.push_back({cv::Point2f(bx, by), gid});
        }
    }

    // Compute mean marker spacing in image (for relative threshold)
    float mean_spacing = 0.f;
    int spacing_count = 0;
    for (size_t i = 1; i < image_pts.size(); ++i)
    {
        for (size_t j = 0; j < i; ++j)
        {
            const float dist = static_cast<float>(cv::norm(image_pts[i] - image_pts[j]));
            // Only count nearby pairs (within 2x expected spacing)
            if (dist < 200.f)
            {
                mean_spacing += dist;
                ++spacing_count;
            }
        }
    }
    if (spacing_count > 0)
    {
        mean_spacing /= static_cast<float>(spacing_count);
    }
    else
    {
        mean_spacing = 50.f;  // fallback
    }

    int newly_identified = 0;

    for (size_t i = 0; i < markers.size(); ++i)
    {
        if (markers[i].global_id_ >= 0)
        {
            continue;
        }

        const cv::Point2f img_pos(markers[i].col_, markers[i].row_);

        // Try global homography first
        float best_dist = mean_spacing * 0.3f;  // 30% of spacing threshold
        int best_gid = -1;

        // Project all board positions through global H
        for (const auto& bp : all_board_positions)
        {
            if (used_global_ids.count(bp.global_id))
            {
                continue;
            }
            const cv::Mat pt = (cv::Mat_<double>(3, 1) << bp.board_pt.x, bp.board_pt.y, 1.0);
            const cv::Mat projected = H_global * pt;
            const cv::Point2f proj_img(static_cast<float>(projected.at<double>(0) / projected.at<double>(2)),
                                        static_cast<float>(projected.at<double>(1) / projected.at<double>(2)));
            const float dist = static_cast<float>(cv::norm(img_pos - proj_img));
            if (dist < best_dist)
            {
                best_dist = dist;
                best_gid = bp.global_id;
            }
        }

        // Try local homography (4 closest identified neighbors).
        // Disabled for near-square grids: local H from 4 neighbors propagates
        // errors when seeds have wrong gids (common on 10x7 boards).
        const float bw = board.is_asymetric_
            ? static_cast<float>((2 * (board.cols_ - 1) + 1) * board.spacing_)
            : static_cast<float>((board.cols_ - 1) * board.spacing_);
        const float bh = static_cast<float>((board.rows_ - 1) * board.spacing_);
        const float board_aspect = std::max(bw, bh) / std::max(1.f, std::min(bw, bh));
        const bool local_h_allowed = board_aspect >= 1.3f;  // only for non-square grids

        if (best_gid < 0 && local_h_allowed && identified_indices.size() >= 4)
        {
            std::vector<std::pair<float, int>> neighbor_dists;
            for (size_t k = 0; k < identified_indices.size(); ++k)
            {
                const float d = static_cast<float>(cv::norm(img_pos - image_pts[k]));
                neighbor_dists.emplace_back(d, static_cast<int>(k));
            }
            std::partial_sort(neighbor_dists.begin(),
                              neighbor_dists.begin() + std::min(4, static_cast<int>(neighbor_dists.size())),
                              neighbor_dists.end());

            std::vector<cv::Point2f> local_board, local_image;
            for (int k = 0; k < std::min(4, static_cast<int>(neighbor_dists.size())); ++k)
            {
                const int idx = neighbor_dists[k].second;
                local_board.push_back(board_pts[idx]);
                local_image.push_back(image_pts[idx]);
            }

            const cv::Mat H_local = cv::findHomography(local_board, local_image, 0);
            if (!H_local.empty())
            {
                for (const auto& bp : all_board_positions)
                {
                    if (used_global_ids.count(bp.global_id))
                    {
                        continue;
                    }
                    const cv::Mat pt = (cv::Mat_<double>(3, 1) << bp.board_pt.x, bp.board_pt.y, 1.0);
                    const cv::Mat projected = H_local * pt;
                    const cv::Point2f proj_img(
                        static_cast<float>(projected.at<double>(0) / projected.at<double>(2)),
                        static_cast<float>(projected.at<double>(1) / projected.at<double>(2)));
                    const float dist = static_cast<float>(cv::norm(img_pos - proj_img));
                    if (dist < best_dist)
                    {
                        best_dist = dist;
                        best_gid = bp.global_id;
                    }
                }
            }
        }

        if (best_gid >= 0)
        {
            markers[i].global_id_ = best_gid;
            used_global_ids.insert(best_gid);
            ++newly_identified;
            spdlog::debug("Local homography: marker at ({:.1f}, {:.1f}) -> global_id={} (dist={:.1f})", markers[i].col_,
                          markers[i].row_, best_gid, best_dist);
        }
    }

    spdlog::debug("Local homography: identified {} new markers", newly_identified);
}

void circlegrid::validate_and_correct_topology(std::vector<base::MarkerRing>& markers, const BoardCircleGrid& board)
{
    // No-op: row swap detection is now done in detection.cpp using tracker state comparison,
    // which is more reliable than spatial-ordering approaches.
    (void)markers;
    (void)board;
}

std::optional<std::vector<int>> circlegrid::identify_with_tracking(const std::vector<base::MarkerCoding>& prev_markers,
                                                                   const std::vector<base::MarkerCoding>& curr_markers,
                                                                   const std::vector<int>& prev_ids,
                                                                   float distance_threshold, float ransac_threshold)
{
    constexpr float kRatioThreshold = 0.75f;
    constexpr size_t kMinCorrespondences = 4;

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
    if (curr_markers.size() > prev_markers.size())
    {
        cv::swap(prev_pts, curr_pts);
    }

    spdlog::debug("identify_with_tracking: starting KNN match");
    cv::BFMatcher matcher(cv::NORM_L2);
    std::vector<std::vector<cv::DMatch>> knn_matches;
    matcher.knnMatch(curr_pts, prev_pts, knn_matches, 2);
    spdlog::debug("identify_with_tracking: KNN match done, {} matches", knn_matches.size());

    std::vector<cv::Point2f> src_pts, dst_pts;
    std::vector<std::pair<int, int>> correspondences;

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

        if (curr_markers.size() > prev_markers.size())
        {
            src_pts.emplace_back(curr_markers.at(best.trainIdx).col_, curr_markers.at(best.trainIdx).row_);
            dst_pts.emplace_back(prev_markers.at(i).col_, prev_markers.at(i).row_);
            correspondences.emplace_back(static_cast<int>(i), best.trainIdx);
        }
        else
        {
            src_pts.emplace_back(curr_markers.at(i).col_, curr_markers.at(i).row_);
            dst_pts.emplace_back(prev_markers.at(best.trainIdx).col_, prev_markers.at(best.trainIdx).row_);
            correspondences.emplace_back(best.trainIdx, static_cast<int>(i));
        }
    }

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

bool circlegrid::validate_tracking_with_ecc(const std::vector<base::MarkerCoding>& prev_markers,
                                            const std::vector<base::MarkerCoding>& curr_markers,
                                            const cv::Mat1b& prev_image, const cv::Mat1b& curr_image,
                                            float distance_threshold, float ransac_threshold, float ecc_threshold)
{
    constexpr float kRatioThreshold = 0.75f;
    constexpr size_t kMinCorrespondences = 4;

    if (prev_markers.empty() || curr_markers.empty())
    {
        return false;
    }

    // Build point matrices for BFMatcher
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

    // Swap if needed (smaller set as query)
    bool swapped = false;
    if (curr_markers.size() > prev_markers.size())
    {
        cv::swap(prev_pts, curr_pts);
        swapped = true;
    }

    // KNN matching
    cv::BFMatcher matcher(cv::NORM_L2);
    std::vector<std::vector<cv::DMatch>> knn_matches;
    matcher.knnMatch(curr_pts, prev_pts, knn_matches, 2);

    std::vector<cv::Point2f> src_pts, dst_pts;

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

        if (swapped)
        {
            src_pts.emplace_back(curr_markers[best.trainIdx].col_, curr_markers[best.trainIdx].row_);
            dst_pts.emplace_back(prev_markers[i].col_, prev_markers[i].row_);
        }
        else
        {
            src_pts.emplace_back(curr_markers[i].col_, curr_markers[i].row_);
            dst_pts.emplace_back(prev_markers[best.trainIdx].col_, prev_markers[best.trainIdx].row_);
        }
    }

    if (src_pts.size() < kMinCorrespondences)
    {
        spdlog::debug("ECC validation: insufficient correspondences ({})", src_pts.size());
        return false;
    }

    // Compute homography
    std::vector<uchar> inlier_mask;
    const cv::Mat H = cv::findHomography(src_pts, dst_pts, cv::RANSAC, ransac_threshold, inlier_mask);

    if (H.empty())
    {
        spdlog::debug("ECC validation: homography computation failed");
        return false;
    }

    // Warp current image to align with previous
    cv::Mat1b warped_curr;
    cv::warpPerspective(curr_image, warped_curr, H, prev_image.size());

    // Compute ECC between warped current and previous
    try
    {
        cv::Mat warp_matrix = cv::Mat::eye(2, 3, CV_32F);  // Identity for translation-only refinement
        const double ecc =
            cv::findTransformECC(prev_image, warped_curr, warp_matrix, cv::MOTION_TRANSLATION,
                                 cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 50, 0.001));

        spdlog::debug("ECC validation: ECC score = {:.3f} (threshold = {:.1f})", ecc, ecc_threshold);
        return ecc >= ecc_threshold;
    }
    catch (const cv::Exception& e)
    {
        spdlog::debug("ECC validation: findTransformECC failed - {}", e.what());
        return false;
    }
}

void circlegrid::identify_new_markers_by_row_lines(std::vector<base::MarkerRing>& markers, const BoardCircleGrid& board)
{
    constexpr float kLineDistanceThreshold = 10.0f;

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
            const auto& marker = markers[row_info.marker_indices[0]];
            row_info.point_on_line = cv::Point2f(marker.col_, marker.row_);
            continue;
        }
        std::vector<std::pair<double, MarkerIdx>> sorted;

        // Find the marker with the lowest global_id_ in this row
        MarkerIdx reference_idx = row_info.marker_indices[0];
        for (const MarkerIdx idx : row_info.marker_indices)
        {
            const auto& marker = markers[idx];
            const auto& reference_marker = markers[reference_idx];
            if (marker.global_id_ < reference_marker.global_id_)
            {
                reference_idx = idx;
            }
        }

        const cv::Point2f reference_point(markers[reference_idx].col_, markers[reference_idx].row_);

        for (const MarkerIdx idx : row_info.marker_indices)
        {
            const cv::Point2f point(markers[idx].col_, markers[idx].row_);
            const double distance = cv::norm(point - reference_point);
            sorted.emplace_back(distance, idx);
        }

        std::sort(sorted.begin(), sorted.end());

        const int first = sorted.front().second;
        const int last = sorted.back().second;
        const cv::Point2f p1(markers[first].col_, markers[first].row_);
        const cv::Point2f p2(markers[last].col_, markers[last].row_);

        const double length = cv::norm(p2 - p1);
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

    for (auto& marker : markers)
    {
        if (marker.global_id_ >= 0)
        {
            continue;
        }

        const cv::Point2f pos(marker.col_, marker.row_);
        float min_dist = kLineDistanceThreshold;
        RowIdx closest_row = -1;

        for (const auto& [row, ri] : row_infos)
        {
            if (!ri.direction.has_value())
            {
                continue;
            }
            const cv::Point2f v = pos - ri.point_on_line;
            const float dist = std::abs(v.x * (-ri.direction->y) + v.y * ri.direction->x);
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

bool populate_indices(std::vector<int>& indices, const std::vector<base::MarkerCoding>& coding_markers,
                      const std::vector<cv::Point2f>& centers)
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
            indices.push_back(-1);
            spdlog::warn("Center id {} ({:0.1f}, {:0.1f}) was not identified!", idx_marker, marker.col_, marker.row_);
        }
    }

    if (coding_markers.size() != indices.size())
    {
        throw std::runtime_error(std::format("Identified {} markers of {}!", indices.size(), coding_markers.size()));
    }
    return true;
}

/// Apply orientation transform to findCirclesGrid centers.
/// For a fixed pattern_size, the only valid orientations are:
///   0 = IDENTITY: raw findCirclesGrid order
///   1 = FLIP_180: reverse the entire array (180° rotation of the grid)
static std::vector<cv::Point2f> apply_orientation(const std::vector<cv::Point2f>& centers,
                                                   int /*rows*/, int /*cols*/, int orientation)
{
    if (orientation == 1)
    {
        auto result = centers;
        std::reverse(result.begin(), result.end());
        return result;
    }
    return centers;  // IDENTITY
}

/// Compute board 2D coordinates for all grid positions
static std::vector<cv::Point2f> compute_board_points(const BoardCircleGrid& board)
{
    std::vector<cv::Point2f> pts;
    pts.reserve(board.rows_ * board.cols_);
    for (int r = 0; r < board.rows_; ++r)
    {
        for (int c = 0; c < board.cols_; ++c)
        {
            float bx, by;
            if (board.is_asymetric_)
            {
                bx = static_cast<float>((2 * c + r % 2)) * board.spacing_;
                by = static_cast<float>(r) * board.spacing_;
            }
            else
            {
                bx = static_cast<float>(c) * board.spacing_;
                by = static_cast<float>(r) * board.spacing_;
            }
            pts.emplace_back(bx, by);
        }
    }
    return pts;
}

/// Compute homography reprojection error for a given orientation.
/// Returns average squared reprojection error per point.
static float compute_orientation_cost(const std::vector<cv::Point2f>& centers,
                                       const std::vector<cv::Point2f>& board_pts,
                                       int rows, int cols, int orientation)
{
    const auto remapped = apply_orientation(centers, rows, cols, orientation);
    const cv::Mat H = cv::findHomography(board_pts, remapped, 0);
    if (H.empty()) return std::numeric_limits<float>::max();

    float total_err = 0.f;
    for (size_t i = 0; i < board_pts.size(); ++i)
    {
        const cv::Mat pt = (cv::Mat_<double>(3, 1) << board_pts[i].x, board_pts[i].y, 1.0);
        const cv::Mat proj = H * pt;
        const float px = static_cast<float>(proj.at<double>(0) / proj.at<double>(2));
        const float py = static_cast<float>(proj.at<double>(1) / proj.at<double>(2));
        const float dx = px - remapped[i].x;
        const float dy = py - remapped[i].y;
        total_err += dx * dx + dy * dy;
    }
    return total_err / static_cast<float>(board_pts.size());
}

bool circlegrid::test_find_circles_grid(std::vector<int>& indices,
                                        const std::vector<base::MarkerCoding>& coding_markers,
                                        const BoardCircleGrid& board,
                                        TrackingState& tracker_state)
{
    const int total = board.rows_ * board.cols_;
    if (static_cast<int>(coding_markers.size()) < total)
    {
        spdlog::debug("test_find_circles_grid: insufficient markers: {} < {}", coding_markers.size(), total);
        return false;
    }

    // Convert markers to keypoints
    std::vector<cv::KeyPoint> keypoints;
    keypoints.reserve(coding_markers.size());
    float max_x = 0.f, max_y = 0.f;
    for (const auto& marker : coding_markers)
    {
        const float size = static_cast<float>(marker.width_ring_ + marker.height_ring_) / 2.0f;
        keypoints.emplace_back(cv::Point2f(marker.col_, marker.row_), size);
        max_x = std::max(max_x, marker.col_);
        max_y = std::max(max_y, marker.row_);
    }

    // Create dummy image and run findCirclesGrid
    const int img_w = static_cast<int>(std::ceil(max_x)) + 100;
    const int img_h = static_cast<int>(std::ceil(max_y)) + 100;
    cv::Mat1b dummy_image = cv::Mat1b::zeros(img_h, img_w);

    const cv::Size pattern_size(board.cols_, board.rows_);
    const int base_flags = board.is_asymetric_ ? cv::CALIB_CB_ASYMMETRIC_GRID : cv::CALIB_CB_SYMMETRIC_GRID;

    std::vector<cv::Point2f> centers;
    bool found = false;

    // Try clustering first, then non-clustering
    {
        cv::Ptr<cv::Feature2D> blob_detector = cv::makePtr<PredetectedBlobDetector>(keypoints);
        found = cv::findCirclesGrid(dummy_image, pattern_size, centers, base_flags | cv::CALIB_CB_CLUSTERING, blob_detector);
    }
    if (!found)
    {
        cv::Ptr<cv::Feature2D> blob_detector = cv::makePtr<PredetectedBlobDetector>(keypoints);
        found = cv::findCirclesGrid(dummy_image, pattern_size, centers, base_flags, blob_detector);
    }

    if (!found || static_cast<int>(centers.size()) != total)
    {
        spdlog::debug("test_find_circles_grid: findCirclesGrid failed ({} keypoints, pattern {}x{})",
                      keypoints.size(), board.cols_, board.rows_);
        return false;
    }

    // Fix asymmetric grid row ordering: OpenCV's findCirclesGrid may return odd rows before
    // even rows. We try both orderings (original and row-pair-swapped) and pick the one with
    // lower homography reprojection error against the board model.
    if (board.is_asymetric_ && board.rows_ >= 2)
    {
        const int cols = board.cols_;
        const int rows = board.rows_;

        // Build row-pair-swapped version
        std::vector<cv::Point2f> swapped(centers.size());
        for (int i = 0; i < total; ++i)
        {
            const int det_row = i / cols;
            const int det_col = i % cols;
            int new_row = (det_row % 2 == 0) ? det_row + 1 : det_row - 1;
            if (new_row >= rows) new_row = det_row;
            swapped[new_row * cols + det_col] = centers[i];
        }

        // Compare homography fit for both orderings
        const auto board_pts = compute_board_points(board);
        const float cost_original = compute_orientation_cost(centers, board_pts, rows, cols, 0);
        const float cost_swapped = compute_orientation_cost(swapped, board_pts, rows, cols, 0);

        spdlog::info("test_find_circles_grid: row-pair cost original={:.1f}, swapped={:.1f}",
                     cost_original, cost_swapped);

        if (cost_swapped < cost_original * 0.9f)
        {
            spdlog::info("test_find_circles_grid: applying row-pair swap");
            centers = swapped;
        }
    }

    // Resolve 180° orientation ambiguity.
    // findCirclesGrid can return the grid in two orientations (starting from opposite corners).
    // For non-square grids, the homography cost distinguishes them.
    // For near-square grids, both orientations have identical cost.
    //
    // Solution: store the first frame's board→image mapping as reference. For each subsequent
    // frame, compare both orientations against the reference and pick the consistent one.
    // The reference is stored as the set of (global_id → image_position) pairs.
    const auto board_pts = compute_board_points(board);

    // For non-square grids: use homography cost to pick orientation (decisive)
    const float cost_identity = compute_orientation_cost(centers, board_pts, board.rows_, board.cols_, 0);
    const float cost_flip180 = compute_orientation_cost(centers, board_pts, board.rows_, board.cols_, 1);

    int best_orientation = 0;

    if (cost_flip180 < cost_identity * 0.8f)
    {
        best_orientation = 1;
    }
    else if (cost_identity < cost_flip180 * 0.8f)
    {
        best_orientation = 0;
    }
    else
    {
        // Costs are similar (near-square grid): compare against a LOCKED reference.
        // Use the previous findCirclesGrid frame's centers for comparison.
        // Also maintain an immutable first-frame reference for robustness across restarts.
        auto& prev_centers = tracker_state.prev_findcircles_centers_;

        // Immutable reference: set once, never updated. Survives optimizer restarts
        // because it's static (per-process, shared across all TrackingState instances).
        static std::vector<cv::Point2f> immutable_reference;

        if (immutable_reference.empty())
        {
            // Very first successful detection in this process — lock as reference
            immutable_reference = apply_orientation(centers, board.rows_, board.cols_, 0);
            prev_centers = immutable_reference;
            best_orientation = 0;
            spdlog::info("test_find_circles_grid: locked immutable 180° reference ({} centers)", total);
        }
        else
        {
            // Compare against immutable reference (robust across restarts)
            const auto identity_centers = apply_orientation(centers, board.rows_, board.cols_, 0);
            const auto flipped_centers = apply_orientation(centers, board.rows_, board.cols_, 1);

            float cost_id = 0.f, cost_flip = 0.f;
            for (int i = 0; i < total; ++i)
            {
                cost_id += static_cast<float>(cv::norm(identity_centers[i] - immutable_reference[i]));
                cost_flip += static_cast<float>(cv::norm(flipped_centers[i] - immutable_reference[i]));
            }

            best_orientation = (cost_flip < cost_id) ? 1 : 0;

            // Also update prev_centers for Hungarian tracking consistency
            prev_centers = (best_orientation == 0) ? identity_centers : flipped_centers;
        }
    }

    spdlog::debug("test_find_circles_grid: orientation={}", best_orientation == 0 ? "IDENTITY" : "FLIP_180");

    // Apply the best orientation
    const auto final_centers = apply_orientation(centers, board.rows_, board.cols_, best_orientation);
    return populate_indices(indices, coding_markers, final_centers);
}

}  // namespace identification
