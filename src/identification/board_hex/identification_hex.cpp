#include "../identification.hpp"

#include <cmath>
#include <numeric>
#include <queue>

#include "board.hpp"
#include "calibration.hpp"
#include "constants.hpp"
#include "filtering.hpp"

#include <spdlog/spdlog.h>

namespace
{

float unroll_angle(float angle)
{
    while (angle < -pi)
    {
        angle += 2. * pi;
    }
    while (angle > pi)
    {
        angle -= 2. * pi;
    }
    return angle;
}

std::vector<base::MarkerNeighborhood> find_neighbors(const std::vector<base::MarkerRing> &markers)
{
    static constexpr float kAngleThresh = 0.5f * pi / 6.f;
    static const float kDistRatioThresh = 2.5f * std::sqrt(3.f) / 2.f;
    static const float kMaxDistRatioThresh = 3.0f;

    std::vector<std::vector<float>> distances(markers.size(), std::vector<float>(markers.size()));
    std::vector<std::vector<float>> angles(markers.size(), std::vector<float>(markers.size()));
    for (size_t id_1st = 0; id_1st < markers.size(); ++id_1st)
    {
        for (size_t id_2nd = 0; id_2nd < markers.size(); ++id_2nd)
        {
            const float dist = std::pow(std::pow(markers[id_1st].row_ - markers[id_2nd].row_, 2.f) +
                                            std::pow(markers[id_1st].col_ - markers[id_2nd].col_, 2.f),
                                        0.5f);
            const float angle =
                std::atan2(markers[id_2nd].row_ - markers[id_1st].row_, markers[id_2nd].col_ - markers[id_1st].col_);

            distances[id_1st][id_2nd] = dist;
            angles[id_1st][id_2nd] = angle;
        }
    }

    std::vector<base::MarkerNeighborhood> neighbors_graph(markers.size());
    auto &graph = neighbors_graph;
    int removed_with_angle_thresh = 0;
    int removed_with_dist_thresh = 0;
    int removed_with_ratio_thresh = 0;
    for (size_t id_1st = 0; id_1st < markers.size(); ++id_1st)
    {
        // sort indices based on distance
        std::vector<int> neighbors_1st(distances[id_1st].size());
        std::iota(neighbors_1st.begin(), neighbors_1st.end(), 0);
        std::ranges::sort(neighbors_1st, [id_1st, &distances](const int i, const int j)
                          { return distances[id_1st][i] < distances[id_1st][j]; });

        for (size_t idx_2nd = 0; idx_2nd < markers.size(); ++idx_2nd)
        {
            const int id_2nd = neighbors_1st[idx_2nd];
            const float dist = distances[id_1st][id_2nd];
            const float angle = angles[id_1st][id_2nd];

            if ((int)id_1st == id_2nd)
            {
                continue;
            }

            // skip ring if has similar angle
            int neighbors_cnt = 0;
            float neighbors_dist_sum = 0.;
            bool similar_angle = false;
            for (int neighbour_idx = 0; neighbour_idx < (int)graph[id_1st].neighbors.size(); ++neighbour_idx)
            {
                ++neighbors_cnt;
                neighbors_dist_sum += graph[id_1st].distances[neighbour_idx];

                const float angle_diff = std::abs(unroll_angle(graph[id_1st].angles[neighbour_idx] - angle));
                if (angle_diff < kAngleThresh)
                {
                    similar_angle = true;
                    break;
                }
            }
            if (similar_angle)
            {
                removed_with_angle_thresh++;
                continue;
            }
            const float neighbors_dist_mean = neighbors_dist_sum / (float)neighbors_cnt;
            if (dist > kDistRatioThresh * neighbors_dist_mean)
            {
                removed_with_dist_thresh++;
                continue;
            }

            if (!graph[id_1st].distances.empty())
            {
                const float neighbors_dist_closest = graph[id_1st].distances.front();
                const float dist_ratio = dist / neighbors_dist_closest;
                if (dist_ratio > kMaxDistRatioThresh)
                {
                    removed_with_ratio_thresh++;
                    continue;
                }
            }

            graph[id_1st].neighbors.push_back(id_2nd);
            graph[id_1st].distances.push_back(dist);
            graph[id_1st].angles.push_back(angle);
            if (graph[id_1st].neighbors.size() == 10)
            {
                break;
            }
        }
    }
    spdlog::info(
        "Removed {} edges with angle threshold, removed {} edges with distance threshold, removed {} edges with ratio "
        "threshold",
        removed_with_angle_thresh, removed_with_dist_thresh, removed_with_ratio_thresh);

    size_t num_edges = 0;
    for (int id_1st = 0; id_1st < (int)neighbors_graph.size(); ++id_1st)
    {
        for (int idx_2nd = 0; idx_2nd < (int)neighbors_graph[id_1st].neighbors.size(); ++idx_2nd)
        {
            ++num_edges;
        }
    }
    if (!num_edges)
    {
        spdlog::warn("Neighbors graph has no edges!");
        return {};
    }

    return neighbors_graph;
}

/**
 * @brief makes sure that left coding marker has less col_ coord then right marker
 */
std::pair<int, int> swap_coding_pair(const std::pair<int, int> coding_pair,
                                     const std::vector<base::MarkerRing> coding_then_rings)
{
    std::pair<int, int> coding_pair_swapped = coding_pair;
    if (coding_then_rings[coding_pair.first].col_ > coding_then_rings[coding_pair.second].col_)
    {
        std::swap(coding_pair_swapped.first, coding_pair_swapped.second);
    }
    return coding_pair_swapped;
}

std::vector<int> get_neighborhood_from_distance(const int id_1st, const int distance,
                                                const std::vector<base::MarkerNeighborhood> &graph)
{
    if (id_1st >= (int)graph.size())
    {
        return {};
    }
    if (distance <= 0)
    {
        return {id_1st};
    }

    std::vector<int> neighborhood;
    std::vector<bool> visited(graph.size());
    std::priority_queue<std::array<int, 2>> to_visit;
    for (size_t idx_2nd = 0; idx_2nd < graph[id_1st].neighbors.size(); ++idx_2nd)
    {
        const int id_2nd = graph[id_1st].neighbors[idx_2nd];
        to_visit.push({(int)distance - 1, id_2nd});
        visited[id_2nd] = true;
    }
    visited[id_1st] = true;
    while (!to_visit.empty())
    {
        const auto [priority, id_2nd] = to_visit.top();
        to_visit.pop();
        if (priority == 0)
        {
            neighborhood.push_back(id_2nd);
            continue;
        }
        for (size_t idx_3rd = 0; idx_3rd < graph[id_2nd].neighbors.size(); ++idx_3rd)
        {
            const int id_3rd = graph[id_2nd].neighbors[idx_3rd];
            if (visited[id_3rd])
            {
                continue;
            }
            to_visit.push({priority - 1, id_3rd});
            visited[id_3rd] = true;
        }
    }
    return neighborhood;
}

/// @returns pairs of possible coding markers each of which lies in a 'coding_distance' from each other
std::vector<std::pair<int, int>> get_coding_pairs(const std::vector<base::MarkerNeighborhood> &neighbors_graph,
                                                  const size_t num_found_coding, const size_t coding_distance)
{
    if (neighbors_graph.size() <= num_found_coding || num_found_coding < 2)
    {
        return {};
    }

    size_t num_valid_coding = 0;
    for (auto coding_id = neighbors_graph.size() - num_found_coding; coding_id < neighbors_graph.size(); ++coding_id)
    {
        num_valid_coding += !neighbors_graph[coding_id].neighbors.empty();
    }

    if (num_valid_coding < 2)
    {
        return {};
    }

    std::vector<std::pair<int, std::vector<int>>> neighborhoods;
    for (int coding_id = 0; coding_id < int(num_found_coding); ++coding_id)
    {
        auto neighborhood = get_neighborhood_from_distance((int)coding_id, (int)coding_distance, neighbors_graph);
        if (!neighborhood.empty())
        {
            neighborhoods.emplace_back(coding_id, std::move(neighborhood));
        }
    }
    if (neighborhoods.size() < 2)
    {
        return {};
    }

    std::vector<std::pair<int, int>> coding_pairs;
    for (size_t yin = 0; yin < neighborhoods.size(); ++yin)
    {
        const auto &[id_1st, neighborhood_1st] = neighborhoods[yin];
        for (size_t yang = yin + 1; yang < neighborhoods.size(); ++yang)
        {
            const auto &[id_2nd, neighborhood_2nd] = neighborhoods[yang];
            const bool is_2nd_in_1st_neighborhood =
                (neighborhood_1st.end() != std::ranges::find(neighborhood_1st, id_2nd));
            const bool is_1st_in_2nd_neighborhood =
                (neighborhood_2nd.end() != std::ranges::find(neighborhood_2nd, id_1st));
            if (is_2nd_in_1st_neighborhood && is_1st_in_2nd_neighborhood)
            {
                coding_pairs.emplace_back(id_1st, id_2nd);
            }
        }
    }
    return coding_pairs;
}

std::optional<std::pair<int, int>> define_initial_ordering_set(
    const std::vector<base::MarkerNeighborhood> &neighbors_graph, const size_t num_found_coding,
    const BoardHexGrid &board)
{
    const int coding_distance = board.col_right_ - board.col_left_;
    std::vector<std::pair<int, int>> coding_pairs =
        get_coding_pairs(neighbors_graph, num_found_coding, coding_distance);
    if (coding_pairs.size() != 1)
    {
        return std::nullopt;
    }
    return coding_pairs[0];
}

auto concat_coding_rings(const std::vector<base::MarkerRing> &rings, const std::vector<base::MarkerCoding> &coding)
{
    std::vector<base::MarkerRing> coding_then_rings;

    coding_then_rings.reserve(rings.size() + coding.size());
    for (size_t marker_idx = 0; marker_idx < coding.size(); ++marker_idx)
    {
        const base::MarkerCoding &src = coding[marker_idx];
        base::MarkerRing dst;
        dst.row_ = src.row_;
        dst.col_ = src.col_;
        dst.width_ring_ = src.width_ring_;
        dst.height_ring_ = src.height_ring_;
        dst.lalbe_inner_ = -1 - static_cast<int>(marker_idx);
        coding_then_rings.push_back(std::move(dst));
    }
    coding_then_rings.insert(coding_then_rings.end(), rings.begin(), rings.end());
    return coding_then_rings;
}

identification::OrderingBoardHex create_ordering(const std::vector<int> &markers_order, const BoardHexGrid &board)
{
    identification::OrderingBoardHex ordering_set;
    ordering_set.ordering_ =
        Eigen::Matrix<std::optional<int>, -1, -1>::Constant(board.rows_, board.cols_, std::nullopt);
    ordering_set.coordinate_grid_defined_ = Eigen::Matrix<bool, -1, -1>::Constant(board.rows_, board.cols_, false);
    for (int idx = 0; idx < int(markers_order.size()); ++idx)
    {
        if (markers_order[idx] == -1)
        {
            continue;
        }
        Eigen::Vector2i rowcol = board.id_to_row_and_col(markers_order[idx]);
        ordering_set.ordering_(rowcol(0), rowcol(1)) = idx;
        ordering_set.coordinate_grid_defined_(rowcol(0), rowcol(1)) = true;
    }
    return ordering_set;
}

}  // namespace

std::pair<std::optional<identification::Decoded>, std::vector<base::MarkerNeighborhood>>
identification::assign_global_IDs(const std::vector<base::MarkerCoding> &coding,
                                  const std::vector<base::MarkerRing> &rings, const BoardHexGrid &board)
{
    std::pair<std::optional<identification::Decoded>, std::vector<base::MarkerNeighborhood>> ret;
    auto &[decoded, neighbors] = ret;

    std::vector<base::MarkerRing> coding_then_rings = concat_coding_rings(rings, coding);
    neighbors = find_neighbors(coding_then_rings);
    identification::filter_out_flase_positives(neighbors);
    const auto coding_pair_initial = define_initial_ordering_set(neighbors, coding.size(), board);
    if (!coding_pair_initial.has_value())
    {
        return ret;
    }
    const std::pair<int, int> coding_pair = swap_coding_pair(*coding_pair_initial, coding_then_rings);
    const std::vector<int> markers_order =
        identification::create_markers_order(neighbors, coding_then_rings, coding_pair, board);
    if (markers_order.empty())
    {
        return ret;
    }

    decoded = identification::Decoded();
    decoded->ordering_type = BoardType::HEX;
    decoded->ordering_hex = create_ordering(markers_order, board);
    decoded->markers = std::move(coding_then_rings);
    decoded->identified_markers = (int)markers_order.size();
    return ret;
}
