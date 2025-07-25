#pragma once

#include "calibration.hpp"

namespace marker::debug
{
static constexpr std::string_view kMarkersSubdir = "markers";

void save_inner_markers_and_unique(const cv::Mat1b &image, const std::vector<base::MarkerUnidentified> &markers_core,
                                   const std::vector<base::MarkerUnidentified> &ring_and_coding, const int image_idx,
                                   const size_t scale_idx);

void save_inner_markers_and_unique(const cv::Mat1b &image, const std::vector<base::MarkerCoding> &coding,
                                   const std::vector<base::MarkerRing> &ring, const int image_idx,
                                   const size_t scale_idx);

void save_marker_identification(const cv::Mat1b &image, const Eigen::Matrix<std::optional<int>, -1, -1> &ordering,
                                const std::vector<base::MarkerRing> &markers, const int image_idx);

void save_neighbors_edges(const cv::Mat1b &image, const std::vector<base::MarkerNeighborhood> &neighbors,
                          const std::vector<base::MarkerCoding> &coding, const std::vector<base::MarkerRing> &ring,
                          const int image_idx);
}  // namespace marker::debug
