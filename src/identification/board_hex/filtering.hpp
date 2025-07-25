#pragma once

#include <vector>

namespace base
{
struct MarkerNeighborhood;
struct MarkerRing;
}  // namespace base

class BoardHexGrid;

namespace identification
{

void filter_out_flase_positives(std::vector<base::MarkerNeighborhood> &neighbors_graph);

std::vector<int> create_markers_order(const std::vector<base::MarkerNeighborhood> &neighbors_graph,
                                      const std::vector<base::MarkerRing> markers,
                                      const std::pair<int, int> coding_pair, const BoardHexGrid &board);

}  // namespace identification
