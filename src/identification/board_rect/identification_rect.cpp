#include "../identification.hpp"

#include <cmath>

#include "board.hpp"
#include "filtering_rect.hpp"
#include "marker/knn.hpp"

namespace
{
float trust_radius(const base::MarkerCoding &start, const base::MarkerCoding &stop)
{
    return ((start.height_ring_ + start.width_ring_) / 2.0 + (stop.height_ring_ + stop.width_ring_) / 2.0f) / 2.0f;
}

struct Line
{
    Eigen::ParametrizedLine<float, 2> line_;
    Eigen::Vector2f start_, stop_;
};

Line fit_line(const base::MarkerCoding &start, const base::MarkerCoding &stop)
{
    Line line;
    line.start_ = Eigen::Vector2f(start.row_, start.col_);
    line.stop_ = Eigen::Vector2f(stop.row_, stop.col_);
    line.line_ = Eigen::ParametrizedLine<float, 2>::Through(line.start_, line.stop_);
    return line;
}

std::vector<int> find_rings_on_line(const Line &line, const std::vector<base::MarkerRing> &rings,
                                    const float trust_radius)
{
    std::vector<std::pair<float, int>> on_line_with_distance_to_start;
    for (int ring_idx = 0; ring_idx < rings.size(); ++ring_idx)
    {
        const auto &ring = rings[ring_idx];
        const Eigen::Vector2f ring_point(ring.row_, ring.col_);
        const float distance = line.line_.distance(ring_point);
        if (distance > trust_radius)
        {
            continue;
        }
        const Eigen::Vector2f direction_to_start = (ring_point - line.start_).normalized();
        const float dot_direction_start = direction_to_start.dot(line.line_.direction());
        if (dot_direction_start < 0.0f)
        {
            // point in oposite direction
            continue;
        }
        const Eigen::Vector2f direction_to_end = (ring_point - line.stop_).normalized();
        const float dot_direction_end = direction_to_end.dot(line.line_.direction());
        if (dot_direction_end > 0.0f)
        {
            // point in oposite direction
            continue;
        }

        on_line_with_distance_to_start.emplace_back((ring_point - line.start_).norm(), ring_idx);
    }

    std::sort(on_line_with_distance_to_start.begin(), on_line_with_distance_to_start.end());  // sort on first element

    std::vector<int> ordering_ids;
    for (const auto &[distance, id] : on_line_with_distance_to_start)
    {
        ordering_ids.emplace_back(id);
    }
    return ordering_ids;
}

/**
 * @brief Define marker indexes that goes on detected board markers from given coding marker to next (so it stores
 * initial assignment of what markers are identified). Vector does not include coding markers from board (so it only has
 * ring markers).
 */
struct LineMarkerIdx
{
    std::vector<int> zero_one_marker_idx_;
    std::vector<int> one_two_marker_idx_;

    LineMarkerIdx(const std::vector<int> &zero_one_marker_idx, const std::vector<int> &one_two_marker_idx)
        : zero_one_marker_idx_(zero_one_marker_idx), one_two_marker_idx_(one_two_marker_idx)
    {
    }
};

struct InitialOrdering
{
    std::vector<std::array<int, 3>> coding_markers_;
    std::vector<LineMarkerIdx> initial_decoding_lines_;

    size_t size() const { return coding_markers_.size(); }
};

InitialOrdering define_initial_ordering_set(const std::vector<base::MarkerCoding> &coding,
                                            const std::vector<base::MarkerRing> &rings, const BoardRectGrid &board)
{
    // we count markers BETWEEN coding markers, they are one less
    const int zero_one_markers_count = board.marker_distance_01() - 1;
    const int one_two_markers_count = board.marker_distance_12() - 1;

    InitialOrdering candidate_ordering;

    // iterate over all coding markers and find their global IDs
    for (int idx_1st = 0; idx_1st < coding.size(); ++idx_1st)
    {
        for (int idx_2nd = 0; idx_2nd < coding.size(); ++idx_2nd)
        {
            if (idx_1st == idx_2nd)
            {
                continue;
            }
            const auto first_line = fit_line(coding[idx_1st], coding[idx_2nd]);
            const auto first_line_ordering =
                find_rings_on_line(first_line, rings, trust_radius(coding[idx_1st], coding[idx_2nd]));

            if (first_line_ordering.size() != zero_one_markers_count)
            {
                continue;
            }
            for (int idx_3rd = 0; idx_3rd < coding.size(); ++idx_3rd)
            {
                if (idx_1st == idx_3rd)
                {
                    continue;
                }
                if (idx_2nd == idx_3rd)
                {
                    continue;
                }

                const auto second_line = fit_line(coding[idx_2nd], coding[idx_3rd]);
                const auto second_line_ordering =
                    find_rings_on_line(second_line, rings, trust_radius(coding[idx_2nd], coding[idx_3rd]));

                if (second_line_ordering.size() != one_two_markers_count)
                {
                    continue;
                }
                candidate_ordering.coding_markers_.emplace_back(std::array<int, 3>{idx_1st, idx_2nd, idx_3rd});
                candidate_ordering.initial_decoding_lines_.emplace_back(first_line_ordering, second_line_ordering);
            }
        }
    }
    return candidate_ordering;
}

std::vector<base::MarkerRing> create_extended_ring_set(const std::vector<base::MarkerCoding> &coding,
                                                       const std::array<int, 3> &coding_set_idx,
                                                       const std::vector<base::MarkerRing> &rings)
{
    auto rings_expanded = rings;

    rings_expanded.emplace_back(coding[coding_set_idx[0]]);
    rings_expanded.emplace_back(coding[coding_set_idx[1]]);
    rings_expanded.emplace_back(coding[coding_set_idx[2]]);

    return rings_expanded;
}

bool set_from_lines(identification::OrderingBoardRect &ordering_set, const LineMarkerIdx &lines,
                    const BoardRectGrid &board, const int initial_ring_count)
{
    ordering_set.ordering_ =
        Eigen::Matrix<std::optional<int>, -1, -1>::Constant(board.rows_, board.cols_, std::nullopt);
    ordering_set.coordinate_grid_defined_ = Eigen::Matrix<bool, -1, -1>::Constant(board.rows_, board.cols_, false);
    ordering_set.assign_coding_idx(board, initial_ring_count);

    if (!ordering_set.assign_index_from_line(lines.zero_one_marker_idx_, board.marker_lines_01_locations()))
    {
        return false;
    }

    if (!ordering_set.assign_index_from_line(lines.one_two_marker_idx_, board.marker_lines_12_locations()))
    {
        return false;
    }

    return true;
}

bool set_initial_L_connection(identification::OrderingBoardRect &ordering_set, const BoardRectGrid &board,
                              const std::vector<base::MarkerRing> &rings_expanded)
{
    identification::IdxToExpand L_connected(board.row_down_ - 1, board.col_down_ + 1, 2);
    std::array<identification::LocalCoordinateGrid, 4> neigbouring_grid;

    if (!ordering_set.left_col_valid(L_connected.row_, L_connected.col_))
    {
        return false;
    }

    neigbouring_grid[base::direction::kLeft] =
        ordering_set.get_point_coordinate_grid(L_connected.row_, L_connected.col_ - 1, rings_expanded);

    if (!ordering_set.bottom_row_valid(L_connected.row_, L_connected.col_))
    {
        return false;
    }
    neigbouring_grid[base::direction::kBottom] =
        ordering_set.get_point_coordinate_grid(L_connected.row_ + 1, L_connected.col_, rings_expanded);

    const Eigen::Vector2f bottom_prediction =
        neigbouring_grid[base::direction::kBottom].origin_ - neigbouring_grid[base::direction::kLeft].row_grid_;
    const Eigen::Vector2f left_prediction =
        neigbouring_grid[base::direction::kLeft].origin_ + neigbouring_grid[base::direction::kBottom].col_grid_;

    const auto [distance, idx] = marker::closest_marker((bottom_prediction + left_prediction) / 2.0f, rings_expanded);
    const float trust_radius = ordering_set.average_trust_radius(L_connected.row_, L_connected.col_, rings_expanded);
    if (distance > trust_radius)
    {
        return false;
    }

    // set point found, and 4 points (including coding marker) as points with defined coordinate grid
    ordering_set.ordering_(L_connected.row_, L_connected.col_) = idx;
    ordering_set.coordinate_grid_defined_(L_connected.row_, L_connected.col_) = true;
    ordering_set.coordinate_grid_defined_(board.row_down_ - 1, board.col_down_) = true;
    ordering_set.coordinate_grid_defined_(board.row_down_, board.col_down_) = true;
    ordering_set.coordinate_grid_defined_(board.row_down_, board.col_down_ + 1) = true;

    return true;
}

}  // namespace

const Eigen::Matrix<std::optional<int>, -1, -1> &identification::Decoded::ordering() const
{
    switch (ordering_type)
    {
        case BoardType::RECT:
            return ordering_rect.ordering_;
        case BoardType::HEX:
            return ordering_hex.ordering_;
    }
}
Eigen::Matrix<std::optional<int>, -1, -1> &identification::Decoded::ordering()
{
    switch (ordering_type)
    {
        case BoardType::RECT:
            return ordering_rect.ordering_;
        case BoardType::HEX:
            return ordering_hex.ordering_;
    }
}

std::pair<std::optional<identification::Decoded>, std::vector<base::MarkerNeighborhood>>
identification::assign_global_IDs(const std::vector<base::MarkerCoding> &coding,
                                  const std::vector<base::MarkerRing> &rings, const BoardRectGrid &board)
{
    using RetB_t = std::vector<base::MarkerNeighborhood>;

    if (coding.size() < 3)
    {
        return {std::nullopt, RetB_t()};
    }

    const auto candidate_ordering = define_initial_ordering_set(coding, rings, board);

    std::vector<identification::Decoded> ordering_with_score;

    for (size_t ordering_idx = 0; ordering_idx < candidate_ordering.size(); ++ordering_idx)
    {
        auto &decoded = ordering_with_score.emplace_back();
        auto &ordering_set = decoded.ordering_rect;
        auto &rings_expanded = decoded.markers;

        rings_expanded = create_extended_ring_set(coding, candidate_ordering.coding_markers_[ordering_idx], rings);
        const marker::MarkerSearch search_engine(rings_expanded, 4 * 2, 7 * 2);

        if (!set_from_lines(ordering_set, candidate_ordering.initial_decoding_lines_[ordering_idx], board,
                            rings.size()))
        {
            decoded.identified_markers = -1;
            continue;
        }

        if (!set_initial_L_connection(ordering_set, board, rings_expanded))
        {
            decoded.identified_markers = -1;
            continue;
        }

        bool was_change = true;
        while (was_change)
        {
            std::set<int> to_skip;
            was_change = false;
            while (true)
            {
                const auto best_to_expand = ordering_set.get_next_best_to_expand(to_skip);
                if (best_to_expand.connection_count_ == -1)
                {
                    // nothing to expand
                    break;
                }

                const auto neigbouring_grid =
                    estimate_local_coordinate_grid(ordering_set, best_to_expand, rings_expanded);

                const Eigen::Vector2f predicted_location = location_from_connections(neigbouring_grid);
                const auto [distance, idx] = search_engine.closest_marker(predicted_location, true);

                if (idx == -1)
                {
                    to_skip.insert(best_to_expand.row_ * ordering_set.ordering_.cols() + best_to_expand.col_);
                    continue;
                }

                const float trust_radius =
                    ordering_set.average_trust_radius(best_to_expand.row_, best_to_expand.col_, rings_expanded);
                if (distance > trust_radius)
                {
                    to_skip.insert(best_to_expand.row_ * ordering_set.ordering_.cols() + best_to_expand.col_);
                    continue;
                }

                was_change = true;

                ordering_set.ordering_(best_to_expand.row_, best_to_expand.col_) = idx;
                ordering_set.set_defined_grid_in_neigbour(best_to_expand.row_, best_to_expand.col_);
            }
        }

        decoded.identified_markers = ordering_set.decoded_markers_count();
    }

    int best_decoding_idx = 0;
    int best_decoding_count = std::numeric_limits<int>::min();

    for (size_t idx = 0; idx < ordering_with_score.size(); ++idx)
    {
        if (ordering_with_score[idx].identified_markers > best_decoding_count)
        {
            best_decoding_idx = idx;
            best_decoding_count = ordering_with_score[idx].identified_markers;
        }
    }

    if (best_decoding_count < 0)
    {
        return {std::nullopt, RetB_t()};
    }

    if (!test_smoothens_of_edges(ordering_with_score[best_decoding_idx].ordering_rect.ordering_,
                                 ordering_with_score[best_decoding_idx].markers,
                                 board.edge_average_difference_allowed_))
    {
        return {std::nullopt, RetB_t()};
    }

    ordering_with_score[best_decoding_idx].ordering_type = BoardType::RECT;
    return std::pair(std::move(ordering_with_score[best_decoding_idx]), RetB_t());
}

std::pair<std::optional<identification::Decoded>, std::vector<base::MarkerNeighborhood>>
identification::assign_global_IDs(const std::vector<base::MarkerCoding> &coding,
                                  const std::vector<base::MarkerRing> &rings, const std::unique_ptr<Board> &board)
{
    BoardRectGrid *board_grid = dynamic_cast<BoardRectGrid *>(board.get());
    if (board_grid)
    {
        return identification::assign_global_IDs(coding, rings, *board_grid);
    }
    BoardHexGrid *board_hex = dynamic_cast<BoardHexGrid *>(board.get());
    if (board_hex)
    {
        return identification::assign_global_IDs(coding, rings, *board_hex);
    }
    return {};
}
