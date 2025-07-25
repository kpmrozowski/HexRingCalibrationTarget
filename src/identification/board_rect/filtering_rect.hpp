#pragma once

#include <optional>

#include <Eigen/Core>

#include <vector>

namespace base
{
struct MarkerRing;
}  // namespace base

class BoardHexGrid;

namespace identification
{
bool test_smoothens_of_edges(const Eigen::Matrix<std::optional<int>, -1, -1> &ordering,
                             const std::vector<base::MarkerRing> &ring, const float edge_average_difference_allowed);

}  // namespace identification
