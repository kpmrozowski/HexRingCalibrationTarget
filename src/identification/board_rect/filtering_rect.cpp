#include "filtering_rect.hpp"

#include "calibration.hpp"

bool identification::test_smoothens_of_edges(const Eigen::Matrix<std::optional<int>, -1, -1> &ordering,
                                             const std::vector<base::MarkerRing> &ring,
                                             const float edge_average_difference_allowed)
{
    const std::array<std::array<int, 2>, 4> offsets = {std::array<int, 2>{-1, 0}, std::array<int, 2>{0, -1},
                                                       std::array<int, 2>{1, 0}, std::array<int, 2>{0, 1}};
    for (int row = 1; row < ordering.rows() - 1; ++row)
    {
        for (int col = 1; col < ordering.cols() - 1; ++col)
        {
            if (!ordering(row, col).has_value())
            {
                continue;
            }
            const int own_idx = ordering(row, col).value();

            std::vector<float> edge_length;
            float average = 0.0f;
            for (const auto [off_row, off_col] : offsets)
            {
                if (!ordering(row + off_row, col + off_col).has_value())
                {
                    continue;
                }
                const int neigbour_idx = ordering(row + off_row, col + off_col).value();
                edge_length.emplace_back((Eigen::Vector2f(ring[own_idx].row_, ring[own_idx].col_) -
                                          Eigen::Vector2f(ring[neigbour_idx].row_, ring[neigbour_idx].col_))
                                             .norm());
                average += edge_length.back();
            }
            average /= edge_length.size();
            const float min = average * edge_average_difference_allowed;
            const float max = average * (1.0f + edge_average_difference_allowed);

            for (const auto len : edge_length)
            {
                if (len < min || len > max)
                {
                    return false;
                }
            }
        }
    }

    return true;
}
