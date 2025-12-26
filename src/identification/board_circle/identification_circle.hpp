#pragma once

#include <optional>
#include <vector>

#include <opencv2/core.hpp>

#include "board.hpp"
#include "calibration.hpp"

class BoardCircleGrid;

namespace identification::circlegrid
{

struct TrackingState
{
    std::vector<base::MarkerCoding> prev_markers_;
    std::vector<int> prev_global_ids_;
    bool has_previous_ = false;

    void update(const std::vector<base::MarkerCoding>& markers, const std::vector<int>& global_ids);
    void clear();
};

/**
 * @brief Fallback identification using frame-to-frame tracking with KNN + RANSAC
 *
 * @param prev_markers Markers detected in previous frame
 * @param curr_markers Markers detected in current frame
 * @param prev_ids Global IDs of previous frame markers
 * @param distance_threshold Maximum distance for KNN matching
 * @param ransac_threshold RANSAC reprojection threshold
 * @return Vector of global IDs for current markers, or nullopt if tracking fails
 */
std::optional<std::vector<int>> identify_with_tracking(const std::vector<base::MarkerCoding>& prev_markers,
                                                       const std::vector<base::MarkerCoding>& curr_markers,
                                                       const std::vector<int>& prev_ids, float distance_threshold,
                                                       float ransac_threshold);

/**
 * @brief Identify new markers by fitting lines through identified markers in each row
 *
 * Uses the assumption that markers in the same row are collinear (small camera distortion).
 * For each unidentified marker, finds the closest row line and computes column offset
 * based on mean spacing between markers.
 *
 * @param markers Vector of markers to update (modifies global_id_ for new identifications)
 * @param board Board definition
 */
void identify_new_markers_by_row_lines(std::vector<base::MarkerRing>& markers, const BoardCircleGrid& board);

/**
 * @brief Tests if the detected markers can form a valid grid pattern using findCirclesGrid.
 *
 * @param indices global indices for each coding marker
 * @param coding_markers Detected coding markers
 * @param board Board definition
 * @return true if findCirclesGrid successfully detects the full pattern
 */
bool test_find_circles_grid(std::vector<int>& indices, const std::vector<base::MarkerCoding>& coding_markers,
                            const BoardCircleGrid& board);

}  // namespace identification::circlegrid
