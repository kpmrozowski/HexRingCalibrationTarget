#include "filtering.hpp"

#include <numeric>
#include <ranges>
#include <set>
#include <vector>

#include <spdlog/spdlog.h>

#include "calibration.hpp"
#include "constants.hpp"

namespace
{

using Edge = std::array<int, 2>;

std::set<Edge> get_obtuse_edges(const std::vector<base::MarkerNeighborhood> &adj)
{
    static constexpr float kMaxObtuseAngle = 1.1f * pi / 2.f;
    std::set<Edge> out;

    // remove edge if it's an edge of the triangle lying opposite the obtuse angle
    for (int id_1st = 0; id_1st < (int)adj.size(); ++id_1st)
    {
        if (adj[id_1st].neighbors.empty())
        {
            continue;
        }
        std::map<int, int> neighbors_1st_order;
        std::transform(adj[id_1st].neighbors.begin(), adj[id_1st].neighbors.end(),
                       std::inserter(neighbors_1st_order, neighbors_1st_order.begin()),
                       [idx = 0](const int id_first) mutable { return std::pair{id_first, idx++}; });

        for (size_t idx_2nd = 0; idx_2nd < adj[id_1st].neighbors.size(); ++idx_2nd)
        {
            const int id_2nd = adj[id_1st].neighbors[idx_2nd];

            std::map<int, int> neighbors_2nd_order;
            std::transform(adj[id_2nd].neighbors.begin(), adj[id_2nd].neighbors.end(),
                           std::inserter(neighbors_2nd_order, neighbors_2nd_order.begin()),
                           [idx = 0](const int id_second) mutable { return std::pair{id_second, idx++}; });

            std::vector<int> neighbors_common;
            std::ranges::set_intersection(neighbors_1st_order | std::ranges::views::keys,
                                          neighbors_2nd_order | std::ranges::views::keys,
                                          std::back_inserter(neighbors_common));
            if (neighbors_common.empty())
            {
                continue;
            }

            for (size_t idx_3rd = 0; idx_3rd < neighbors_common.size(); ++idx_3rd)
            {
                const int id_3rd = neighbors_common[idx_3rd];
                const auto idx_3rd_in_1st =
                    std::distance(adj[id_1st].neighbors.begin(), std::ranges::find(adj[id_1st].neighbors, id_3rd));
                const auto idx_3rd_in_2nd =
                    std::distance(adj[id_2nd].neighbors.begin(), std::ranges::find(adj[id_2nd].neighbors, id_3rd));
                const float dist_1nd_to_2nd = std::pow(adj[id_1st].distances[idx_2nd], 2.f);
                const float dist_3rd_to_1nd = std::pow(adj[id_1st].distances[idx_3rd_in_1st], 2.f);
                const float dist_3rd_to_2nd = std::pow(adj[id_2nd].distances[idx_3rd_in_2nd], 2.f);
                const float obtuse_angle_1nd_to_2nd =
                    std::acos((dist_3rd_to_1nd + dist_3rd_to_2nd - dist_1nd_to_2nd) /
                              (2.f * adj[id_1st].distances[idx_3rd_in_1st] * adj[id_2nd].distances[idx_3rd_in_2nd]));
                const float obtuse_angle_3rd_to_1nd =
                    std::acos((dist_1nd_to_2nd + dist_3rd_to_2nd - dist_3rd_to_1nd) /
                              (2.f * adj[id_1st].distances[idx_2nd] * adj[id_2nd].distances[idx_3rd_in_2nd]));
                const float obtuse_angle_3rd_to_2nd =
                    std::acos((dist_1nd_to_2nd + dist_3rd_to_1nd - dist_3rd_to_2nd) /
                              (2.f * adj[id_1st].distances[idx_2nd] * adj[id_1st].distances[idx_3rd_in_1st]));
                if (obtuse_angle_1nd_to_2nd > kMaxObtuseAngle)
                {
                    out.insert({std::min(id_1st, id_2nd), std::max(id_1st, id_2nd)});
                }
                else if (obtuse_angle_3rd_to_1nd > kMaxObtuseAngle)
                {
                    out.insert({std::min(id_1st, id_3rd), std::max(id_1st, id_3rd)});
                }
                else if (obtuse_angle_3rd_to_2nd > kMaxObtuseAngle)
                {
                    out.insert({std::min(id_3rd, id_2nd), std::max(id_3rd, id_2nd)});
                }
            }
        }
    }

    return out;
}

std::set<Edge> get_vertex_without_neighbors_knowing_each_other(const std::vector<base::MarkerNeighborhood> &graph)
{
    std::set<Edge> no_neighbors_knowing_each_other;
    for (int id_1st = 0; id_1st < (int)graph.size(); ++id_1st)
    {
        if (graph[id_1st].neighbors.empty())
        {
            continue;
        }
        if (graph[id_1st].neighbors.size() == 1)
        {
            const int id_2nd = graph[id_1st].neighbors[0];
            no_neighbors_knowing_each_other.insert({std::min(id_1st, id_2nd), std::max(id_1st, id_2nd)});
            continue;
        }

        bool has_neighbors_knowing_each_other = false;
        for (int idx_2nd = 0; idx_2nd < (int)graph[id_1st].neighbors.size(); ++idx_2nd)
        {
            const int id_2nd = graph[id_1st].neighbors[idx_2nd];
            for (int idx_3rd = 0; idx_3rd < (int)graph[id_1st].neighbors.size(); ++idx_3rd)
            {
                if (idx_2nd == idx_3rd)
                {
                    continue;
                }
                const int id_3rd = graph[id_1st].neighbors[idx_3rd];
                if (std::ranges::find(graph[id_2nd].neighbors, id_3rd) == graph[id_2nd].neighbors.cend() ||
                    std::ranges::find(graph[id_3rd].neighbors, id_2nd) == graph[id_3rd].neighbors.cend())
                {
                    continue;
                }
                has_neighbors_knowing_each_other = true;
                break;
            }
            if (has_neighbors_knowing_each_other)
            {
                break;
            }
        }
        if (has_neighbors_knowing_each_other)
        {
            continue;
        }
        for (int idx_2nd = 0; idx_2nd < (int)graph[id_1st].neighbors.size(); ++idx_2nd)
        {
            const int id_2nd = graph[id_1st].neighbors[idx_2nd];
            no_neighbors_knowing_each_other.insert({std::min(id_1st, id_2nd), std::max(id_1st, id_2nd)});
        }
    }
    return no_neighbors_knowing_each_other;
}

// remove vertex if it has two neighbors which are not their neighbors
std::set<Edge> get_neighbors_dont_know_each_other(const std::vector<base::MarkerNeighborhood> &adj)
{
    std::set<Edge> my_only_two_neighbors_dont_know_each_other;
    for (int id_1st = 0; id_1st < (int)adj.size(); ++id_1st)
    {
        if (adj[id_1st].neighbors.size() != 2)
        {
            continue;
        }
        const int id_2nd = adj[id_1st].neighbors[0];
        const int id_3rd = adj[id_1st].neighbors[1];
        if (std::ranges::find(adj[id_2nd].neighbors, id_3rd) == adj[id_2nd].neighbors.cend() ||
            std::ranges::find(adj[id_3rd].neighbors, id_2nd) == adj[id_3rd].neighbors.cend())
        {
            my_only_two_neighbors_dont_know_each_other.insert({std::min(id_1st, id_2nd), std::max(id_1st, id_2nd)});
            my_only_two_neighbors_dont_know_each_other.insert({std::min(id_1st, id_3rd), std::max(id_1st, id_3rd)});
        }
    }
    return my_only_two_neighbors_dont_know_each_other;
}

// remove vertex if it has one neighbor
std::set<Edge> get_single_neighbour_vertice_edges(const std::vector<base::MarkerNeighborhood> &adj)
{
    std::set<Edge> single_neighbour_vertice_edges;

    std::vector<std::vector<int>> adj_copy;
    adj_copy.reserve(adj.size());
    for (const auto &node_data : adj)
    {
        adj_copy.push_back(node_data.neighbors);
    }

    for (int id_1st = 0; id_1st < (int)adj_copy.size(); ++id_1st)
    {
        int id_curr = id_1st;
        while (id_curr >= 0 && id_curr < (int)adj_copy.size() && adj_copy[id_curr].size() == 1)
        {
            const int id_next = adj_copy[id_curr][0];
            single_neighbour_vertice_edges.insert({std::min(id_curr, id_next), std::max(id_curr, id_next)});
            adj_copy[id_curr].clear();
            if (id_next >= 0 && id_next < (int)adj_copy.size())
            {
                auto &neighbors_of_next = adj_copy[id_next];
                auto it_neighbour = std::ranges::find(neighbors_of_next, id_curr);
                if (it_neighbour != neighbors_of_next.end())
                {
                    neighbors_of_next.erase(it_neighbour);
                }
            }
            id_curr = id_next;
        }
    }
    return single_neighbour_vertice_edges;
}

// remove my neighbour if I'm not his neighbor
std::set<Edge> get_neighbors_that_dont_know_me(const std::vector<base::MarkerNeighborhood> &adj)
{
    std::set<Edge> my_neighbors_that_dont_know_me;
    for (int id_1st = 0; id_1st < (int)adj.size(); ++id_1st)
    {
        for (int idx_2nd = 0; idx_2nd < (int)adj[id_1st].neighbors.size(); ++idx_2nd)
        {
            const int id_2nd = adj[id_1st].neighbors[idx_2nd];
            const auto &neighbors_2nd = adj[id_2nd].neighbors;
            if (std::ranges::find(neighbors_2nd, id_1st) == neighbors_2nd.cend())
            {
                my_neighbors_that_dont_know_me.insert({std::min(id_1st, id_2nd), std::max(id_1st, id_2nd)});
            }
        }
    }
    return my_neighbors_that_dont_know_me;
}

// sort neighbors by angle
void sort_neighbors_by_angle(std::vector<base::MarkerNeighborhood> &graph)
{
    for (size_t id_1st = 0; id_1st < graph.size(); ++id_1st)
    {
        if (graph[id_1st].neighbors.empty())
        {
            continue;
        }
        const size_t num_neighbors = graph[id_1st].neighbors.size();
        std::vector<size_t> order(num_neighbors);
        std::iota(order.begin(), order.end(), 0);
        std::ranges::sort(order, [&graph, id_1st](const size_t i, const size_t j)
                          { return graph[id_1st].angles[i] < graph[id_1st].angles[j]; });
        std::vector<int> neighbors_sorted(num_neighbors);
        std::vector<float> distances_sorted(num_neighbors);
        std::vector<float> angles_sorted(num_neighbors);
        for (size_t idx = 0; idx < num_neighbors; ++idx)
        {
            neighbors_sorted[idx] = graph[id_1st].neighbors[order[idx]];
            distances_sorted[idx] = graph[id_1st].distances[order[idx]];
            angles_sorted[idx] = graph[id_1st].angles[order[idx]];
        }
        graph[id_1st].neighbors = neighbors_sorted;
        graph[id_1st].distances = distances_sorted;
        graph[id_1st].angles = angles_sorted;
    }
}

// check if neighbors, distances and angles vectors are of the same size
void check_size(const std::vector<base::MarkerNeighborhood> &graph)
{
    for (size_t id_1st = 0; id_1st < graph.size(); ++id_1st)
    {
        if (graph[id_1st].neighbors.size() != graph[id_1st].distances.size() ||
            graph[id_1st].neighbors.size() != graph[id_1st].angles.size())
        {
            throw std::runtime_error("Neighbors, distances and angles vectors are not of the same size for id_1st: " +
                                     std::to_string(id_1st));
        }
    }
}

float unroll_angle(float angle)
{
    while (angle < -pi)
    {
        angle += 2.f * pi;
    }
    while (angle > pi)
    {
        angle -= 2.f * pi;
    }
    return angle;
}

// having an angle ordering find edges which rotated 180 degrees does not have corresponding edge
std::pair<std::set<Edge>, std::set<std::array<Edge, 2>>> get_set_of_longest_edge_of_4_neighbors(
    const std::vector<base::MarkerNeighborhood> &graph)
{
    std::pair<std::set<Edge>, std::set<std::array<Edge, 2>>> ret;
    std::set<Edge> &set_of_longest_edge_of_4_neighbors = std::get<0>(ret);
    std::set<std::array<Edge, 2>> &crossing_edge_pairs = std::get<1>(ret);

    for (int id_1st = 0; id_1st < (int)graph.size(); ++id_1st)
    {
        if (graph[id_1st].neighbors.size() < 3)
        {
            continue;
        }

        for (int idx_2nd = 0; idx_2nd < (int)graph[id_1st].neighbors.size(); ++idx_2nd)
        {
            const int id_2nd = graph[id_1st].neighbors[idx_2nd];
            for (int idx_3rd = 0; idx_3rd < (int)graph[id_1st].neighbors.size(); ++idx_3rd)
            {
                if (idx_2nd == idx_3rd)
                {
                    continue;
                }
                const int id_3rd = graph[id_1st].neighbors[idx_3rd];
                const auto it_3rd_in_2nd = std::ranges::find(graph[id_2nd].neighbors, id_3rd);
                const auto it_2nd_in_3rd = std::ranges::find(graph[id_3rd].neighbors, id_2nd);
                if (it_3rd_in_2nd == graph[id_2nd].neighbors.cend() || it_2nd_in_3rd == graph[id_3rd].neighbors.cend())
                {
                    continue;
                }
                for (int idx_4th = 0; idx_4th < (int)graph[id_1st].neighbors.size(); ++idx_4th)
                {
                    if (idx_4th == idx_2nd || idx_4th == idx_3rd)
                    {
                        continue;
                    }
                    const int id_4th = graph[id_1st].neighbors[idx_4th];

                    const auto it_4th_in_2nd = std::ranges::find(graph[id_2nd].neighbors, id_4th);
                    const auto it_2nd_in_4th = std::ranges::find(graph[id_4th].neighbors, id_2nd);
                    const auto it_4th_in_3rd = std::ranges::find(graph[id_3rd].neighbors, id_4th);
                    const auto it_3rd_in_4th = std::ranges::find(graph[id_4th].neighbors, id_3rd);

                    if (it_4th_in_2nd == graph[id_2nd].neighbors.cend() ||
                        it_2nd_in_4th == graph[id_4th].neighbors.cend() ||
                        it_4th_in_3rd == graph[id_3rd].neighbors.cend() ||
                        it_3rd_in_4th == graph[id_4th].neighbors.cend())
                    {
                        continue;
                    }
                    const auto idx_3rd_in_2nd = std::distance(graph[id_2nd].neighbors.begin(), it_3rd_in_2nd);
                    const auto idx_4th_in_2nd = std::distance(graph[id_2nd].neighbors.begin(), it_4th_in_2nd);
                    const auto idx_4th_in_3rd = std::distance(graph[id_3rd].neighbors.begin(), it_4th_in_3rd);
                    std::vector<std::pair<float, Edge>> distances{
                        {std::pair(graph[id_1st].distances[idx_2nd], std::array{id_1st, id_2nd}),
                         std::pair(graph[id_1st].distances[idx_3rd], std::array{id_1st, id_3rd}),
                         std::pair(graph[id_1st].distances[idx_4th], std::array{id_1st, id_4th}),
                         std::pair(graph[id_2nd].distances[idx_3rd_in_2nd], std::array{id_2nd, id_3rd}),
                         std::pair(graph[id_2nd].distances[idx_4th_in_2nd], std::array{id_2nd, id_4th}),
                         std::pair(graph[id_3rd].distances[idx_4th_in_3rd], std::array{id_3rd, id_4th})}};
                    std::ranges::sort(distances);  // sort based on the distance
                    const auto &longest_1st = *std::prev(distances.end(), 1);
                    const auto &longest_2nd = *std::prev(distances.end(), 2);
                    const float longest_diff = longest_1st.first - longest_2nd.first;
                    const float longest_ratio = longest_diff / std::min(longest_1st.first, longest_2nd.first);
                    if (longest_ratio > 0.05f)
                    {  // if the difference is to big, it's obvious it should be deleted
                        const int id_longest_1st = longest_1st.second[0];
                        const int id_longest_2nd = longest_1st.second[1];
                        set_of_longest_edge_of_4_neighbors.insert(
                            {std::min(id_longest_1st, id_longest_2nd), std::max(id_longest_1st, id_longest_2nd)});
                        continue;
                    }

                    // at this stage the longer edge could be the one that should stay but to be certain let's check
                    // which edge has more collinear edges connected to itself
                    crossing_edge_pairs.insert({std::array{std::min(longest_2nd.second[0], longest_2nd.second[1]),
                                                           std::max(longest_2nd.second[0], longest_2nd.second[1])},
                                                std::array{std::min(longest_1st.second[0], longest_1st.second[1]),
                                                           std::max(longest_1st.second[0], longest_1st.second[1])}});
                }
            }
        }
    }

    return ret;
}

std::set<Edge> get_incorrect_crossing_edges(const std::vector<base::MarkerNeighborhood> &graph,
                                            const std::set<std::array<Edge, 2>> &crossing_edge_pairs)
{
    static constexpr float kAngleThresh = 5.f * (pi / 180.f);
    static constexpr float kDistDiffRatioThresh = 0.07f;

    std::set<Edge> incorrect_crossing_edges;

    // (0: min({1:}, {2:}), 1: simmilar_to_a, 2: simmilar_to_b, 3-6: ids)
    using SimmilarEdges = std::vector<Edge>;
    std::vector<std::array<std::pair<Edge, SimmilarEdges>, 2>> edge_stats;

    for (const auto &edge : crossing_edge_pairs)
    {
        const auto &edge_a = edge[0];
        const auto &edge_b = edge[1];
        const int id_1st = edge_a[0];
        const int id_2nd = edge_a[1];
        const int id_3rd = edge_b[0];
        const int id_4th = edge_b[1];
        const auto it_2nd_in_1st = std::ranges::find(graph[id_1st].neighbors, id_2nd);
        const auto it_4th_in_3rd = std::ranges::find(graph[id_3rd].neighbors, id_4th);
        const float angle_a = graph[id_1st].angles[std::distance(graph[id_1st].neighbors.begin(), it_2nd_in_1st)];
        const float angle_b = graph[id_3rd].angles[std::distance(graph[id_3rd].neighbors.begin(), it_4th_in_3rd)];

        // count number of edges in each above vertex with simmilar angles to the edge_1st or edge_2nd
        std::vector<Edge> simmilar_to_a, simmilar_to_b;
        const std::vector<int> search_points{id_1st, id_2nd, id_3rd, id_4th};
        std::set<Edge> visited_edges;
        for (int idx = 0; idx < 4; ++idx)
        {
            const int id = search_points[idx];
            for (int idx_5th = 0; idx_5th < (int)graph[id].neighbors.size(); ++idx_5th)
            {
                const int id_5th = graph[id].neighbors[idx_5th];
                if (std::ranges::find(search_points, id_5th) != search_points.cend())
                {
                    continue;
                }
                for (int idx_6th = 0; idx_6th < (int)graph[id_5th].neighbors.size(); ++idx_6th)
                {
                    const int id_6th = graph[id_5th].neighbors[idx_6th];
                    const auto edge_c = std::array{std::min(id_5th, id_6th), std::max(id_5th, id_6th)};
                    if (visited_edges.contains(edge_c))
                    {
                        continue;
                    }
                    visited_edges.insert(edge_c);

                    const float angle_6th = graph[id_5th].angles[idx_6th];
                    const float angle_6th_to_1st = unroll_angle(angle_6th - angle_a);
                    const float angle_6th_to_1st_pi = unroll_angle(angle_6th - angle_a + pi);
                    const float angle_6th_to_2nd = unroll_angle(angle_6th - angle_b);
                    const float angle_6th_to_2nd_pi = unroll_angle(angle_6th - angle_b + pi);
                    if (std::abs(angle_6th_to_1st) < kAngleThresh || std::abs(angle_6th_to_1st_pi) < kAngleThresh)
                    {
                        simmilar_to_a.push_back({std::min(id_5th, id_6th), std::max(id_5th, id_6th)});
                    }
                    if (std::abs(angle_6th_to_2nd) < kAngleThresh || std::abs(angle_6th_to_2nd_pi) < kAngleThresh)
                    {
                        simmilar_to_b.push_back({std::min(id_5th, id_6th), std::max(id_5th, id_6th)});
                    }
                }
            }
        }
        edge_stats.push_back(std::array{std::pair{edge_a, simmilar_to_a}, std::pair{edge_b, simmilar_to_b}});
    }
    std::vector<int> processing_order(edge_stats.size());
    std::iota(processing_order.begin(), processing_order.end(), 0);
    std::sort(processing_order.begin(), processing_order.end(),
              [&edge_stats](const int i, const int j)
              {
                  const size_t diff_i = std::max(edge_stats[i][0].second.size(), edge_stats[i][1].second.size()) -
                                        std::min(edge_stats[i][0].second.size(), edge_stats[i][1].second.size());
                  const size_t diff_j = std::max(edge_stats[j][0].second.size(), edge_stats[j][1].second.size()) -
                                        std::min(edge_stats[j][0].second.size(), edge_stats[j][1].second.size());
                  return diff_i > diff_j;
              });

    for (const int idx : processing_order)
    {
        const auto &[edge_a, edge_b] = *std::next(crossing_edge_pairs.begin(), idx);
        const auto &[simmilar_to_a, simmilar_to_b] = edge_stats[idx];

        spdlog::debug("== crossing edges: [a({}, {}), b({}, {})] with simmilar edges a:{}, b:{}", edge_a[0], edge_a[1],
                      edge_b[0], edge_b[1], simmilar_to_a.second.size(), simmilar_to_b.second.size());
    }

    std::vector<int> dbg_actual_processing_order;
    bool crossing_exist = !edge_stats.empty();
    while (crossing_exist)
    {
        crossing_exist = false;
        int idx_biggest_diff = -1;
        int max_diff = 0;
        for (const int idx : processing_order)
        {
            if (!edge_stats[idx][0].second.empty() && !edge_stats[idx][1].second.empty())
            {
                crossing_exist = true;
            }
            const size_t diff = std::max(edge_stats[idx][0].second.size(), edge_stats[idx][1].second.size()) -
                                std::min(edge_stats[idx][0].second.size(), edge_stats[idx][1].second.size());
            if ((int)diff > max_diff)
            {
                max_diff = (int)diff;
                idx_biggest_diff = idx;
            }
        }
        if (!crossing_exist)
        {
            break;
        }

        dbg_actual_processing_order.push_back(idx_biggest_diff);

        // const auto &[edge_a, edge_b] = *std::next(crossing_edge_pairs.begin(), idx);
        const auto [edge_a_and_simmilar, edge_b_and_simmilar] = edge_stats[idx_biggest_diff];
        const auto [edge_a, simmilar_to_a] = edge_a_and_simmilar;
        const auto [edge_b, simmilar_to_b] = edge_b_and_simmilar;
        if (edge_a[1] <= 56 || edge_b[1] <= 56)
        {
            spdlog::debug("crossing edges: [a({}, {}), b({}, {})] with simmilar edges a:{}, b:{}, diff:{}", edge_a[0],
                          edge_a[1], edge_b[0], edge_b[1], simmilar_to_a.size(), simmilar_to_b.size(), max_diff);
        }
        const auto find_a = [&edge_a](const Edge &e)
        { return (e[0] == edge_a[0] && e[1] == edge_a[1]) || (e[0] == edge_a[1] && e[1] == edge_a[0]); };
        const auto find_b = [&edge_b](const Edge &e)
        { return (e[0] == edge_b[0] && e[1] == edge_b[1]) || (e[0] == edge_b[1] && e[1] == edge_b[0]); };

        if (simmilar_to_a.size() < simmilar_to_b.size())
        {  // add edge_a to remove
            incorrect_crossing_edges.insert(edge_a);
            // remove edge_a from edge_stats
            for (int idx_stats = 0; idx_stats < (int)edge_stats.size(); ++idx_stats)
            {
                if (idx_stats == idx_biggest_diff)
                {
                    continue;
                }
                const auto it_edge_a_in_0 = std::ranges::find_if(edge_stats[idx_stats][0].second, find_a);
                if (it_edge_a_in_0 != edge_stats[idx_stats][0].second.end())
                {
                    edge_stats[idx_stats][0].second.erase(it_edge_a_in_0);
                    // continue;
                }
                const auto it_edge_a_in_1 = std::ranges::find_if(edge_stats[idx_stats][1].second, find_a);
                if (it_edge_a_in_1 != edge_stats[idx_stats][1].second.end())
                {
                    edge_stats[idx_stats][1].second.erase(it_edge_a_in_1);
                }
            }
        }
        else
        {  // add edge_b to remove
            incorrect_crossing_edges.insert(edge_b);
            // remove edge_b from edge_stats
            for (int idx_stats = 0; idx_stats < (int)edge_stats.size(); ++idx_stats)
            {
                if (idx_stats == idx_biggest_diff)
                {
                    continue;
                }
                const auto it_edge_b_in_0 = std::ranges::find_if(edge_stats[idx_stats][0].second, find_b);
                if (it_edge_b_in_0 != edge_stats[idx_stats][0].second.end())
                {
                    edge_stats[idx_stats][0].second.erase(it_edge_b_in_0);
                    // continue;
                }
                const auto it_edge_b_in_1 = std::ranges::find_if(edge_stats[idx_stats][1].second, find_b);
                if (it_edge_b_in_1 != edge_stats[idx_stats][1].second.end())
                {
                    edge_stats[idx_stats][1].second.erase(it_edge_b_in_1);
                }
            }
        }
        edge_stats[idx_biggest_diff][0].second.clear();
        edge_stats[idx_biggest_diff][1].second.clear();
    }

    return incorrect_crossing_edges;
}

// having an angle ordering find edges which rotated 180 degrees does not have corresponding edge
std::set<Edge> get_edges_without_collinear_edges(const std::vector<base::MarkerNeighborhood> &graph)
{
    static constexpr float kAngleThresh = 5.f * (pi / 180.f);
    static constexpr float kDistDiffRatioThresh = 0.07f;

    std::set<Edge> edges_without_collinear_edges;

    for (int id_1st = 0; id_1st < (int)graph.size(); ++id_1st)
    {
        if (graph[id_1st].neighbors.size() < 3)
        {
            continue;
        }

        for (int idx_2nd = 0; idx_2nd < (int)graph[id_1st].neighbors.size(); ++idx_2nd)
        {
            const int id_2nd = graph[id_1st].neighbors[idx_2nd];
            const float angle_2nd = graph[id_1st].angles[idx_2nd];
            for (int idx_3rd = 0; idx_3rd < (int)graph[id_1st].neighbors.size(); ++idx_3rd)
            {
                if (idx_2nd == idx_3rd)
                {
                    continue;
                }
                const int id_3rd = graph[id_1st].neighbors[idx_3rd];
                const float angle_3rd = graph[id_1st].angles[idx_3rd];
                if (std::ranges::find(graph[id_2nd].neighbors, id_3rd) == graph[id_2nd].neighbors.cend() ||
                    std::ranges::find(graph[id_3rd].neighbors, id_2nd) == graph[id_3rd].neighbors.cend())
                {
                    continue;
                }
                for (int idx_4th = 0; idx_4th < (int)graph[id_1st].neighbors.size(); ++idx_4th)
                {
                    if (idx_4th == idx_2nd || idx_4th == idx_3rd)
                    {
                        continue;
                    }
                    const int id_4th = graph[id_1st].neighbors[idx_4th];
                    const float dist_4th_to_1st = graph[id_1st].distances[idx_4th];

                    if (std::ranges::find(graph[id_2nd].neighbors, id_4th) != graph[id_2nd].neighbors.cend() ||
                        std::ranges::find(graph[id_3rd].neighbors, id_4th) != graph[id_3rd].neighbors.cend())
                    {
                        continue;
                    }

                    const std::array edge_1st_4th{std::min(id_1st, id_4th), std::max(id_1st, id_4th)};
                    if (edges_without_collinear_edges.contains(edge_1st_4th))
                    {
                        continue;
                    }

                    const float angle_4th = graph[id_1st].angles[idx_4th];

                    const float angle_4th_to_2nd = unroll_angle(angle_2nd - angle_4th + pi);
                    const float angle_4th_to_3rd = unroll_angle(angle_3rd - angle_4th + pi);
                    if (std::abs(angle_4th_to_2nd) < kAngleThresh || std::abs(angle_4th_to_3rd) < kAngleThresh)
                    {
                        continue;
                    }

                    bool has_simmilar_edge_nereby = false;
                    for (int idx = 0; idx < 2; ++idx)
                    {
                        const int id_1st_or_4th = idx == 1 ? id_1st : id_4th;
                        const int id_4th_or_1st = idx == 1 ? id_4th : id_1st;
                        for (int idx_5th_in = 0; idx_5th_in < (int)graph[id_1st_or_4th].neighbors.size(); ++idx_5th_in)
                        {
                            const int id_5th = graph[id_1st_or_4th].neighbors[idx_5th_in];
                            if (id_5th == id_4th_or_1st || id_5th == id_2nd || id_5th == id_3rd)
                            {
                                continue;
                            }
                            const float angle_5th_in_4th = graph[id_1st_or_4th].angles[idx_5th_in];
                            const float angle_3th_to_5th = unroll_angle(angle_5th_in_4th - angle_4th);
                            const float angle_3th_to_5th_pi = unroll_angle(angle_5th_in_4th - angle_4th + pi);
                            if (std::abs(angle_3th_to_5th) > kAngleThresh &&
                                std::abs(angle_3th_to_5th_pi) > kAngleThresh)
                            {
                                continue;
                            }

                            const float dist_5th_to = graph[id_1st_or_4th].distances[idx_5th_in];
                            const float dist_diff = std::abs(dist_4th_to_1st - dist_5th_to);
                            const float dist_diff_ratio = dist_diff / std::min(dist_4th_to_1st, dist_5th_to);
                            if (dist_diff_ratio > kDistDiffRatioThresh)
                            {
                                if (dist_4th_to_1st < dist_5th_to)
                                {  // so much longer edge is strange, should be removed
                                    edges_without_collinear_edges.insert(
                                        {std::min(id_5th, id_1st_or_4th), std::max(id_5th, id_1st_or_4th)});
                                }
                                continue;
                            }
                            has_simmilar_edge_nereby = true;
                            break;
                        }
                        if (has_simmilar_edge_nereby)
                        {
                            break;
                        }
                    }

                    if (!has_simmilar_edge_nereby)
                    {
                        edges_without_collinear_edges.insert(edge_1st_4th);
                    }
                }
            }
        }
    }

    return edges_without_collinear_edges;
}

// having an angle ordering find opposite edges which has different length
std::set<Edge> get_opposite_edges_with_different_length(const std::vector<base::MarkerNeighborhood> &graph)
{
    static constexpr float kAngleThresh = 5.f * (pi / 180.f);
    static constexpr float kDistDiffRatioThresh = 0.05f;

    std::set<Edge> opposite_edges_with_different_length;

    for (int id_1st = 0; id_1st < (int)graph.size(); ++id_1st)
    {
        if (graph[id_1st].neighbors.size() < 2)
        {
            continue;
        }

        using EdgeWithLength = std::pair<Edge, float>;
        std::vector<EdgeWithLength> surrounding_edges_1st;
        for (size_t idx_2nd = 0; idx_2nd < graph[id_1st].neighbors.size(); ++idx_2nd)
        {
            const int id_2nd = graph[id_1st].neighbors[idx_2nd];
            for (size_t idx_3rd = idx_2nd + 1; idx_3rd < graph[id_1st].neighbors.size(); ++idx_3rd)
            {
                const int id_3rd = graph[id_1st].neighbors[idx_3rd];
                const auto it_3rd_in_2nd = std::ranges::find(graph[id_2nd].neighbors, id_3rd);
                const auto it_2nd_in_3rd = std::ranges::find(graph[id_3rd].neighbors, id_2nd);
                if (it_3rd_in_2nd == graph[id_2nd].neighbors.cend() && it_2nd_in_3rd == graph[id_3rd].neighbors.cend())
                {
                    continue;
                }
                const auto idx_3rd_in_2nd = std::distance(graph[id_2nd].neighbors.begin(), it_3rd_in_2nd);
                const auto idx_2nd_in_3rd = std::distance(graph[id_3rd].neighbors.begin(), it_2nd_in_3rd);
                const float dist_2nd_to_3rd = graph[id_3rd].distances[idx_2nd_in_3rd];
                const float dist_3rd_to_2nd = graph[id_2nd].distances[idx_3rd_in_2nd];
                surrounding_edges_1st.emplace_back(
                    EdgeWithLength{{std::min(id_2nd, id_3rd), std::max(id_2nd, id_3rd)}, dist_2nd_to_3rd});
            }
        }

        std::vector<std::array<EdgeWithLength, 2>> dbg_opposite_edges_1st, dbg_mismatching_sbg_opposite_edges_1st;
        for (auto it_edge_a = surrounding_edges_1st.begin(); it_edge_a != surrounding_edges_1st.end(); ++it_edge_a)
        {
            const auto edge_a = std::array{std::min(it_edge_a->first[0], it_edge_a->first[1]),
                                           std::max(it_edge_a->first[0], it_edge_a->first[1])};
            if (opposite_edges_with_different_length.contains(edge_a))
            {
                continue;
            }

            const int id_2nd_a = it_edge_a->first[0];
            const int id_3rd_a = it_edge_a->first[1];
            const auto idx_3rd_in_2nd_a = std::distance(graph[id_2nd_a].neighbors.begin(),
                                                        std::ranges::find(graph[id_2nd_a].neighbors, id_3rd_a));
            const float angle_a = graph[id_2nd_a].angles[idx_3rd_in_2nd_a];

            for (auto it_edge_b = std::next(it_edge_a, 1); it_edge_b != surrounding_edges_1st.end(); ++it_edge_b)
            {
                const auto edge_b = std::array{std::min(it_edge_b->first[0], it_edge_b->first[1]),
                                               std::max(it_edge_b->first[0], it_edge_b->first[1])};
                if (opposite_edges_with_different_length.contains(edge_b))
                {
                    continue;
                }

                const int id_2nd_b = it_edge_b->first[0];
                const int id_3rd_b = it_edge_b->first[1];
                if (id_2nd_a == id_2nd_b || id_3rd_a == id_2nd_b || id_2nd_a == id_3rd_b || id_3rd_a == id_3rd_b)
                {
                    continue;
                }
                const auto idx_3rd_in_2nd_b = std::distance(graph[id_2nd_b].neighbors.begin(),
                                                            std::ranges::find(graph[id_2nd_b].neighbors, id_3rd_b));
                const float angle_b = graph[id_2nd_b].angles[idx_3rd_in_2nd_b];

                const float angle_diff = std::abs(unroll_angle(angle_b - angle_a));
                const float angle_diff_pi = std::abs(unroll_angle(angle_b + pi - angle_a));
                if (angle_diff > kAngleThresh && angle_diff_pi > kAngleThresh)
                {
                    continue;
                }
                dbg_opposite_edges_1st.push_back({*it_edge_a, *it_edge_b});
                const float dist_diff = std::abs(it_edge_a->second - it_edge_b->second);
                const float dist_diff_ratio = dist_diff / std::min(it_edge_a->second, it_edge_b->second);
                if (dist_diff_ratio < kDistDiffRatioThresh)
                {
                    continue;
                }
                dbg_mismatching_sbg_opposite_edges_1st.push_back({*it_edge_a, *it_edge_b});

                if (it_edge_a->second > it_edge_b->second)
                {
                    opposite_edges_with_different_length.insert(edge_a);
                    break;
                }
                else
                {
                    opposite_edges_with_different_length.insert(edge_b);
                    break;
                }
            }
        }
    }

    return opposite_edges_with_different_length;
}

void remove_edge(std::vector<base::MarkerNeighborhood> &graph, const Edge &edge)
{
    const int id_1st = edge[0];
    const int id_2nd = edge[1];

    const auto it_2nd_in_1st = std::ranges::find(graph[id_1st].neighbors, id_2nd);
    if (it_2nd_in_1st != graph[id_1st].neighbors.end())
    {
        const auto idx_2nd_in_1st = std::distance(graph[id_1st].neighbors.begin(), it_2nd_in_1st);
        graph[id_1st].neighbors.erase(it_2nd_in_1st);
        graph[id_1st].distances.erase(std::next(graph[id_1st].distances.begin(), idx_2nd_in_1st));
        graph[id_1st].angles.erase(std::next(graph[id_1st].angles.begin(), idx_2nd_in_1st));
    }

    const auto it_1st_in_2nd = std::ranges::find(graph[id_2nd].neighbors, id_1st);
    if (it_1st_in_2nd != graph[id_2nd].neighbors.end())
    {
        const auto idx_1st_in_2nd = std::distance(graph[id_2nd].neighbors.begin(), it_1st_in_2nd);
        graph[id_2nd].neighbors.erase(it_1st_in_2nd);
        graph[id_2nd].distances.erase(std::next(graph[id_2nd].distances.begin(), idx_1st_in_2nd));
        graph[id_2nd].angles.erase(std::next(graph[id_2nd].angles.begin(), idx_1st_in_2nd));
    }
}

void remove_edges(std::vector<base::MarkerNeighborhood> &graph, const std::set<Edge> &edges_to_remove)
{
    for (const auto &edge : edges_to_remove)
    {
        remove_edge(graph, edge);
    }
}

}  // namespace

void identification::filter_out_flase_positives(std::vector<base::MarkerNeighborhood> &neighbors_graph)
{
    while (true)
    {
        const auto my_neighbors_that_dont_know_me = get_neighbors_that_dont_know_me(neighbors_graph);
        remove_edges(neighbors_graph, my_neighbors_that_dont_know_me);
        if (!my_neighbors_that_dont_know_me.empty())
        {
            continue;
        }

        // remove obsture edges
        const auto obtuse_edges = get_obtuse_edges(neighbors_graph);
        remove_edges(neighbors_graph, obtuse_edges);
        if (!obtuse_edges.empty())
        {
            continue;
        }

        // remove longest edge of 4 neighbors clickue
        const auto a_pair = get_set_of_longest_edge_of_4_neighbors(neighbors_graph);
        const auto &longest_edges_of_4_neighbors = std::get<0>(a_pair);
        remove_edges(neighbors_graph, longest_edges_of_4_neighbors);
        if (!longest_edges_of_4_neighbors.empty())
        {
            continue;
        }

        const std::set<std::array<Edge, 2>> &crossing_edge_pairs = std::get<1>(a_pair);
        const auto incorrect_crossing_edges = get_incorrect_crossing_edges(neighbors_graph, crossing_edge_pairs);
        remove_edges(neighbors_graph, incorrect_crossing_edges);

        break;
    }

    while (true)
    {
        const auto my_only_two_neighbors_dont_know_each_other = get_neighbors_dont_know_each_other(neighbors_graph);
        remove_edges(neighbors_graph, my_only_two_neighbors_dont_know_each_other);
        if (!my_only_two_neighbors_dont_know_each_other.empty())
        {
            continue;
        }

        const auto single_neighbour_edges = get_single_neighbour_vertice_edges(neighbors_graph);
        remove_edges(neighbors_graph, single_neighbour_edges);
        if (!single_neighbour_edges.empty())
        {
            continue;
        }

        sort_neighbors_by_angle(neighbors_graph);
        check_size(neighbors_graph);

        const auto edges_without_collinear_edges = get_edges_without_collinear_edges(neighbors_graph);
        remove_edges(neighbors_graph, edges_without_collinear_edges);
        if (!edges_without_collinear_edges.empty())
        {
            continue;
        }

        const auto opposite_edges_with_different_length = get_opposite_edges_with_different_length(neighbors_graph);
        remove_edges(neighbors_graph, opposite_edges_with_different_length);
        if (!opposite_edges_with_different_length.empty())
        {
            continue;
        }

        const auto no_neighbors_knowing_each_other = get_vertex_without_neighbors_knowing_each_other(neighbors_graph);
        remove_edges(neighbors_graph, no_neighbors_knowing_each_other);
        if (!no_neighbors_knowing_each_other.empty())
        {
            continue;
        }

        break;
    }
}
