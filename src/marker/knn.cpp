#include "knn.hpp"

namespace
{
std::tuple<int, int, int, int> find_min_max_coordinates(const std::vector<base::MarkerRing> &rings)
{
    int min_row = std::numeric_limits<int>::max();
    int min_col = std::numeric_limits<int>::max();
    int max_row = std::numeric_limits<int>::min();
    int max_col = std::numeric_limits<int>::min();
    for (const auto &marker : rings)
    {
        min_col = std::min(int(marker.col_), min_col);
        min_row = std::min(int(marker.row_), min_row);
        max_col = std::max(int(marker.col_), max_col);
        max_row = std::max(int(marker.row_), max_row);
    }
    min_col -= 5;
    min_row -= 5;
    max_col += 5;
    max_row += 5;

    return {min_row, min_col, max_row, max_col};
}

Eigen::Matrix<std::vector<int>, -1, -1> scatter_to_tiles(const std::vector<base::MarkerRing> &rings, const int min_row,
                                                         const int min_col, const int row_step, const int col_step,
                                                         const int tiles_row, const int tiles_col)
{
    Eigen::Matrix<std::vector<int>, -1, -1> tiled_markers(tiles_row + 1, tiles_col + 1);

    for (int idx = 0; idx < rings.size(); ++idx)
    {
        const auto &marker = rings[idx];
        const int row_tile = (int(marker.row_) - min_row) / row_step;
        const int col_tile = (int(marker.col_) - min_col) / col_step;

        tiled_markers(row_tile, col_tile).emplace_back(idx);
    }

    return tiled_markers;
}
}  // namespace

namespace marker
{
MarkerSearch::MarkerSearch(const std::vector<base::MarkerRing> &rings, const int tiles_row, const int tiles_col)
    : rings_(rings)
{
    std::tie(min_row_, min_col_, max_row_, max_col_) = find_min_max_coordinates(rings);

    row_step_ = (max_row_ - min_row_) / tiles_row;
    col_step_ = (max_col_ - min_col_) / tiles_col;

    tiled_markers_ = scatter_to_tiles(rings, min_row_, min_col_, row_step_, col_step_, tiles_row, tiles_col);
}

std::pair<float, int> MarkerSearch::closest_marker(const Eigen::Vector2f &point,
                                                   const bool use_approximate_distance) const
{
    const int row_tile = std::clamp((int(point(0)) - min_row_) / row_step_, 1, int(tiled_markers_.rows() - 2));
    const int col_tile = std::clamp((int(point(1)) - min_col_) / col_step_, 1, int(tiled_markers_.cols() - 2));

    float min_distance = std::numeric_limits<float>::max();
    int closest = -1;

    for (int row_tile_current = row_tile - 1; row_tile_current < row_tile + 2; ++row_tile_current)
    {
        for (int col_tile_current = col_tile - 1; col_tile_current < col_tile + 2; ++col_tile_current)
        {
            for (const auto &marker_id : tiled_markers_(row_tile_current, col_tile_current))
            {
                const float distance = (Eigen::Vector2f(rings_[marker_id].row_, rings_[marker_id].col_) - point).norm();
                if (distance < min_distance)
                {
                    min_distance = distance;
                    closest = marker_id;
                }
            }
        }
    }

    if (closest == -1 && !use_approximate_distance)
    {
        return marker::closest_marker(point, rings_);
    }
    return {min_distance, closest};
}

std::pair<float, int> closest_marker(const Eigen::Vector2f &point, const std::vector<base::MarkerRing> &rings)
{
    float min_distance = std::numeric_limits<float>::max();
    int closest = 0;
    for (int idx = 0; idx < rings.size(); ++idx)
    {
        const float distance = (Eigen::Vector2f(rings[idx].row_, rings[idx].col_) - point).norm();
        if (distance < min_distance)
        {
            min_distance = distance;
            closest = idx;
        }
    }
    return {min_distance, closest};
}
}  // namespace marker
