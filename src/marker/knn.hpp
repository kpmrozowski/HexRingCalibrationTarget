#pragma once

#include "calibration.hpp"

namespace marker
{
class MarkerSearch
{
   private:
    int min_col_, min_row_;
    int max_col_, max_row_;

    int row_step_, col_step_;

    const std::vector<base::MarkerRing> &rings_;

    Eigen::Matrix<std::vector<int>, -1, -1> tiled_markers_;

   public:
    MarkerSearch(const std::vector<base::MarkerRing> &rings, const int tiles_row, const int tiles_col);

    // use_approximate_neigbour indicate to search only if point is withing +-1 tile, if not, we revert to full search
    // if no neigour is found
    std::pair<float, int> closest_marker(const Eigen::Vector2f &point, const bool use_approximate_neigbour) const;
};

std::pair<float, int> closest_marker(const Eigen::Vector2f &point, const std::vector<base::MarkerRing> &rings);
}  // namespace marker