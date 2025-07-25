#pragma once

#include <optional>
#include <set>

#include <Eigen/Core>

namespace base
{
struct MarkerNeighborhood;
struct MarkerRing;
}  // namespace base

class BoardRectGrid;
class BoardHexGrid;

namespace identification
{
struct IdxToExpand
{
    int row_ = -1;
    int col_ = -1;
    int connection_count_ = -1;

    IdxToExpand() = default;
    IdxToExpand(int row, int col, int connection_count);
};

struct LocalCoordinateGrid
{
    Eigen::Vector2f origin_ = Eigen::Vector2f(0, 0);
    Eigen::Vector2f col_grid_ = Eigen::Vector2f(0, 0);
    Eigen::Vector2f row_grid_ = Eigen::Vector2f(0, 0);

    bool col_grid_valid_ = false;
    bool row_grid_valid_ = false;

    LocalCoordinateGrid() = default;
    LocalCoordinateGrid(const Eigen::Vector2f &origin, const Eigen::Vector2f &col_grid, const Eigen::Vector2f &row_grid,
                        const bool col_valid, const bool row_valid);
};

class OrderingBoardRect
{
   public:
    Eigen::Matrix<std::optional<int>, -1, -1> ordering_;
    Eigen::Matrix<bool, -1, -1> coordinate_grid_defined_;

    bool top_row_valid(const int row, const int col) const;
    bool bottom_row_valid(const int row, const int col) const;
    bool left_col_valid(const int row, const int col) const;
    bool right_col_valid(const int row, const int col) const;

    bool top_row_grid_defined(const int row, const int col) const;
    bool bottom_row_grid_defined(const int row, const int col) const;
    bool left_col_grid_defined(const int row, const int col) const;
    bool right_col_grid_defined(const int row, const int col) const;

    void assign_coding_idx(const BoardRectGrid &board, const int coding_idx);
    void assign_coding_idx(const BoardHexGrid &board, const int coding_idx);

    bool assign_index_from_line(const std::vector<int> &ordering_ring,
                                const std::vector<std::pair<int, int>> &ordering_line);

    IdxToExpand get_next_best_to_expand(const std::set<int> &to_skip) const;

    // Return coodinate grid aligned to increasing direction values
    LocalCoordinateGrid get_point_coordinate_grid(const int row, const int col,
                                                  const std::vector<base::MarkerRing> &rings) const;

    float average_trust_radius(const int row, const int col, const std::vector<base::MarkerRing> &rings) const;
    int decoded_markers_count() const;

    void set_defined_grid_in_neigbour(const int row, const int col);
};

class OrderingBoardHex
{
   public:
    Eigen::Matrix<std::optional<int>, -1, -1> ordering_;
    Eigen::Matrix<bool, -1, -1> coordinate_grid_defined_;
};

Eigen::Vector2f location_from_connections(const std::array<LocalCoordinateGrid, 4> &neigbouring_grid);

std::array<LocalCoordinateGrid, 4> estimate_local_coordinate_grid(const OrderingBoardRect &ordering_set,
                                                                  const IdxToExpand &idx_to_expand,
                                                                  const std::vector<base::MarkerRing> &rings_expanded);
/**
 * @brief Assigns global order to
 * @param graph: a vector of adjacent indices representing detected markers (vertices of the graph)
 * @param markers: a vector of size of graph which holds an information about each marker sub-pixel location
 * on an image
 * @param coding_pair: indices of coding markers
 */
std::vector<int> create_markers_order(const std::vector<base::MarkerNeighborhood> &graph,
                                      const std::vector<base::MarkerRing> markers,
                                      const std::pair<int, int> coding_pair, const BoardHexGrid &board);

}  // namespace identification
