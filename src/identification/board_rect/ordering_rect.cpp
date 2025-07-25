#include "../ordering.hpp"

#include "board.hpp"
#include "calibration.hpp"

namespace identification
{

IdxToExpand::IdxToExpand(int row, int col, int connection_count)
    : row_(row), col_(col), connection_count_(connection_count)
{
}

LocalCoordinateGrid::LocalCoordinateGrid(const Eigen::Vector2f &origin, const Eigen::Vector2f &col_grid,
                                         const Eigen::Vector2f &row_grid, const bool col_valid, const bool row_valid)
    : origin_(origin), col_grid_(col_grid), row_grid_(row_grid), col_grid_valid_(col_valid), row_grid_valid_(row_valid)
{
}

bool OrderingBoardRect::top_row_valid(const int row, const int col) const
{
    return (row > 0 && ordering_(row - 1, col).has_value());
}

bool OrderingBoardRect::bottom_row_valid(const int row, const int col) const
{
    return (row < ordering_.rows() - 1 && ordering_(row + 1, col).has_value());
}

bool OrderingBoardRect::left_col_valid(const int row, const int col) const
{
    return (col > 0 && ordering_(row, col - 1).has_value());
}

bool OrderingBoardRect::right_col_valid(const int row, const int col) const
{
    return (col < ordering_.cols() - 1 && ordering_(row, col + 1).has_value());
}

bool OrderingBoardRect::top_row_grid_defined(const int row, const int col) const
{
    return (row > 0 && coordinate_grid_defined_(row - 1, col));
}

bool OrderingBoardRect::bottom_row_grid_defined(const int row, const int col) const
{
    return (row < coordinate_grid_defined_.rows() - 1 && coordinate_grid_defined_(row + 1, col));
}

bool OrderingBoardRect::left_col_grid_defined(const int row, const int col) const
{
    return (col > 0 && coordinate_grid_defined_(row, col - 1));
}

bool OrderingBoardRect::right_col_grid_defined(const int row, const int col) const
{
    return (col < coordinate_grid_defined_.cols() - 1 && coordinate_grid_defined_(row, col + 1));
}

void OrderingBoardRect::assign_coding_idx(const BoardRectGrid &board, const int coding_idx)
{
    ordering_(board.row_top_, board.col_top_) = coding_idx;
    ordering_(board.row_down_, board.col_down_) = coding_idx + 1;
    ordering_(board.row_right_, board.col_right_) = coding_idx + 2;
}

void OrderingBoardRect::assign_coding_idx(const BoardHexGrid &board, const int coding_idx)
{
    ordering_(board.row_left_, board.col_left_) = coding_idx;
    ordering_(board.row_right_, board.col_right_) = coding_idx + 1;
}
bool OrderingBoardRect::assign_index_from_line(const std::vector<int> &ordering_ring,
                                               const std::vector<std::pair<int, int>> &ordering_line)
{
    for (size_t idx = 0; idx < ordering_ring.size(); ++idx)
    {
        if (ordering_(ordering_line[idx].first, ordering_line[idx].second).has_value())
        {
            // on that point we should not have multi assignment, it error indicator
            return false;
        }
        ordering_(ordering_line[idx].first, ordering_line[idx].second) = ordering_ring[idx];
    }
    return true;
}

IdxToExpand OrderingBoardRect::get_next_best_to_expand(const std::set<int> &to_skip) const
{
    std::array<IdxToExpand, 3> next_best_to_expand;
    for (int row = 0; row < ordering_.rows(); ++row)
    {
        for (int col = 0; col < ordering_.cols(); ++col)
        {
            if (ordering_(row, col).has_value())
            {
                continue;
            }

            const int idx = row * ordering_.cols() + col;
            if (to_skip.find(idx) != to_skip.cend())
            {
                continue;
            }

            const int connections = int(top_row_grid_defined(row, col)) + int(bottom_row_grid_defined(row, col)) +
                                    int(right_col_grid_defined(row, col)) + int(left_col_grid_defined(row, col));

            if (connections == 4)
            {
                // just fill 4 connected
                return {row, col, 4};
            }
            if (connections == 0)
            {
                continue;
            }

            next_best_to_expand[connections - 1] = IdxToExpand(row, col, connections);
        }
    }

    if (next_best_to_expand[2].connection_count_ != -1)
    {
        return next_best_to_expand[2];
    }
    if (next_best_to_expand[1].connection_count_ != -1)
    {
        return next_best_to_expand[1];
    }
    if (next_best_to_expand[0].connection_count_ != -1)
    {
        return next_best_to_expand[0];
    }
    return next_best_to_expand[2];
}

// Return coordinate grid aligned to increasing direction values
LocalCoordinateGrid OrderingBoardRect::get_point_coordinate_grid(const int row, const int col,
                                                                 const std::vector<base::MarkerRing> &rings) const
{
    const auto &myself = rings[ordering_(row, col).value()];

    Eigen::Vector2f col_grid_coordinates(0, 0);
    int col_estimates = 0;

    if (left_col_valid(row, col))
    {
        const auto &left = rings[ordering_(row, col - 1).value()];
        col_grid_coordinates += Eigen::Vector2f(myself.row_, myself.col_) - Eigen::Vector2f(left.row_, left.col_);
        col_estimates++;
    }
    if (right_col_valid(row, col))
    {
        const auto &right = rings[ordering_(row, col + 1).value()];
        col_grid_coordinates += Eigen::Vector2f(right.row_, right.col_) - Eigen::Vector2f(myself.row_, myself.col_);
        col_estimates++;
    }

    Eigen::Vector2f row_grid_coordinates(0, 0);
    int row_estimates = 0;

    if (top_row_valid(row, col))
    {
        const auto &top = rings[ordering_(row - 1, col).value()];
        row_grid_coordinates += Eigen::Vector2f(myself.row_, myself.col_) - Eigen::Vector2f(top.row_, top.col_);
        row_estimates++;
    }
    if (bottom_row_valid(row, col))
    {
        const auto &bottom = rings[ordering_(row + 1, col).value()];
        row_grid_coordinates += Eigen::Vector2f(bottom.row_, bottom.col_) - Eigen::Vector2f(myself.row_, myself.col_);
        row_estimates++;
    }

    return LocalCoordinateGrid(
        Eigen::Vector2f(myself.row_, myself.col_), col_grid_coordinates / (col_estimates != 0 ? col_estimates : 1),
        row_grid_coordinates / (row_estimates != 0 ? row_estimates : 1), col_estimates != 0, row_estimates != 0);
}

float OrderingBoardRect::average_trust_radius(const int row, const int col,
                                              const std::vector<base::MarkerRing> &rings) const
{
    float average_radius = 0.0f;
    int counter = 0;
    if (left_col_valid(row, col))
    {
        const auto &left = rings[ordering_(row, col - 1).value()];
        average_radius += (left.height_inner_ + left.width_inner_) / 2.0f;
        counter++;
    }
    if (right_col_valid(row, col))
    {
        const auto &right = rings[ordering_(row, col + 1).value()];
        average_radius += (right.height_inner_ + right.width_inner_) / 2.0f;
        counter++;
    }
    if (top_row_valid(row, col))
    {
        const auto &top = rings[ordering_(row - 1, col).value()];
        average_radius += (top.height_inner_ + top.width_inner_) / 2.0f;
        counter++;
    }
    if (bottom_row_valid(row, col))
    {
        const auto &bottom = rings[ordering_(row + 1, col).value()];
        average_radius += (bottom.height_inner_ + bottom.width_inner_) / 2.0f;
        counter++;
    }
    return average_radius / counter;
}

int OrderingBoardRect::decoded_markers_count() const
{
    int decoded_items = 0;
    for (int row = 0; row < ordering_.rows(); ++row)
    {
        for (int col = 0; col < ordering_.cols(); ++col)
        {
            decoded_items += ordering_(row, col) != -1;
        }
    }
    return decoded_items;
}

void OrderingBoardRect::set_defined_grid_in_neigbour(const int row, const int col)
{
    coordinate_grid_defined_(row, col) = true;
    if (left_col_valid(row, col))
    {
        coordinate_grid_defined_(row, col - 1) = true;
    }
    if (right_col_valid(row, col))
    {
        coordinate_grid_defined_(row, col + 1) = true;
    }
    if (top_row_valid(row, col))
    {
        coordinate_grid_defined_(row - 1, col) = true;
    }
    if (bottom_row_valid(row, col))
    {
        coordinate_grid_defined_(row + 1, col) = true;
    }
}

Eigen::Vector2f location_from_connections(const std::array<identification::LocalCoordinateGrid, 4> &neigbouring_grid)
{
    Eigen::Vector2f predicted_location(0, 0);
    int estimated_from = 0;
    const auto &top = neigbouring_grid[base::direction::kTop];
    if (top.row_grid_valid_)
    {
        predicted_location += top.origin_ + top.row_grid_;
        estimated_from++;
    }
    const auto &left = neigbouring_grid[base::direction::kLeft];
    if (left.col_grid_valid_)
    {
        predicted_location += left.origin_ + left.col_grid_;
        estimated_from++;
    }
    const auto &bottom = neigbouring_grid[base::direction::kBottom];
    if (bottom.row_grid_valid_)
    {
        predicted_location += bottom.origin_ - bottom.row_grid_;
        estimated_from++;
    }
    const auto &right = neigbouring_grid[base::direction::kRight];
    if (right.col_grid_valid_)
    {
        predicted_location += right.origin_ - right.col_grid_;
        estimated_from++;
    }
    return predicted_location / estimated_from;
}

std::array<LocalCoordinateGrid, 4> estimate_local_coordinate_grid(const OrderingBoardRect &ordering_set,
                                                                  const IdxToExpand &idx_to_expand,
                                                                  const std::vector<base::MarkerRing> &rings_expanded)
{
    std::array<LocalCoordinateGrid, 4> neigbouring_grid;
    if (ordering_set.top_row_valid(idx_to_expand.row_, idx_to_expand.col_))
    {
        neigbouring_grid[base::direction::kTop] =
            ordering_set.get_point_coordinate_grid(idx_to_expand.row_ - 1, idx_to_expand.col_, rings_expanded);
    }
    if (ordering_set.left_col_valid(idx_to_expand.row_, idx_to_expand.col_))
    {
        neigbouring_grid[base::direction::kLeft] =
            ordering_set.get_point_coordinate_grid(idx_to_expand.row_, idx_to_expand.col_ - 1, rings_expanded);
    }
    if (ordering_set.bottom_row_valid(idx_to_expand.row_, idx_to_expand.col_))
    {
        neigbouring_grid[base::direction::kBottom] =
            ordering_set.get_point_coordinate_grid(idx_to_expand.row_ + 1, idx_to_expand.col_, rings_expanded);
    }
    if (ordering_set.right_col_valid(idx_to_expand.row_, idx_to_expand.col_))
    {
        neigbouring_grid[base::direction::kRight] =
            ordering_set.get_point_coordinate_grid(idx_to_expand.row_, idx_to_expand.col_ + 1, rings_expanded);
    }
    return neigbouring_grid;
}
}  // namespace identification
