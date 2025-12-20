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
 * @brief Primary identification using OpenCV's findCirclesGrid
 *
 * @param image Input grayscale image
 * @param coding_markers Detected coding markers (full black circles)
 * @param board Board definition with rows, cols, and asymmetric flag
 * @return Vector of global IDs for each marker, or nullopt if detection fails
 */
std::optional<std::vector<int>> identify_with_findCirclesGrid(const cv::Mat1b& image,
                                                               const std::vector<base::MarkerCoding>& coding_markers,
                                                               const BoardCircleGrid& board);

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

}  // namespace identification::circlegrid
