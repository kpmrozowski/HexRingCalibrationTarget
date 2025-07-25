#include "../ordering.hpp"

#include <cmath>
#include <numeric>
#include <queue>
#include <vector>

#include <spdlog/spdlog.h>

#include <constants.hpp>

#include "board.hpp"
#include "calibration.hpp"

namespace
{
// the below thresh. decides about what the maximal angle between two edges can be to be considered as parallel
static constexpr float kAngleThresh = 5. * (pi / 180.f);

/**
 * @ref https://www.redblobgames.com/grids/hexagons/#coordinates
 */
struct AxialCoord
{
    int q = 0, r = 0, s = 0;
    std::optional<float> rotation_axis_q = std::nullopt;
    std::optional<float> rotation_axis_r = std::nullopt;
    std::optional<float> rotation_axis_s = std::nullopt;

    // checks if has at least two axes
    bool has_value() const { return int(!rotation_axis_q) + int(!rotation_axis_r) + int(!rotation_axis_s) < 2; }

    // checks if coords are incremented in the allowed way
    bool is_ok() const { return q + r + s == 0; }
};

struct OffsetCoord
{
    int col, row;
};

enum class HexAxis
{
    Q = 0,
    R = 1,
    S = 2
};

static constexpr std::array<std::string_view, 3> kHexAxisNames{{
    "Q",
    "R",
    "S",
}};

/**
 * @brief The values corresponds to the axis names from kHexAxisNames and holds a pair: (direction idx, direction angle)
 *        Q+(r+,s-): (1, -PI)      S-:(q-,r+): (2, 0.f)
 *                           \    /
 *                            \  /
 * R-:(q-,s-): (-3, -PI) ______\/_____ R+:(q+,s+): (0, 0.f)
 *                             /\
 *                            /  \
 *                           /    \
 *      S+: (q+,r-): (-1, PI)      Q-:(r-,s+): (-2, 0.f)
 */
static constexpr std::array<std::array<std::pair<int, float>, 2>, 3> kHexAxisDirectionOffsetsLeft{
    // {Q+/R+/S+}, {Q-/R-/S-}
    {{std::pair(1, -pi), std::pair(-2, 0.f)},
     {std::pair(0, 0.f), std::pair(-3, -pi)},
     {std::pair(-1, pi), std::pair(2, 0.f)}}};

/**
 * @brief The values corresponds to the axis names from kHexAxisNames and holds a pair: (direction idx, direction angle)
 *        Q+(r+,s-): (1, 0.f)      S-:(q-,r+): (2, PI)
 *                           \    /
 *                            \  /
 * R-:(q-,s-): (-3, 0.f) ______\/_____ R+:(q+,s+): (0, PI)
 *                             /\
 *                            /  \
 *                           /    \
 *     S+: (q+,r-): (-1, 0.f)      Q-:(r-,s+): (-2, -PI)
 */
static constexpr std::array<std::array<std::pair<int, float>, 2>, 3> kHexAxisDirectionOffsetsRight{
    // {Q+/R+/S+}, {Q-/R-/S-}
    {{std::pair(1, 0.f), std::pair(-2, -pi)},
     {std::pair(0, pi), std::pair(-3, 0.f)},
     {std::pair(-1, 0.f), std::pair(2, pi)}}};

std::pair<size_t, float> get_idx_and_offset_of_axis(const std::pair<int, float> &offset,
                                                    const size_t idx_middle_ring_in_coding,
                                                    const std::vector<float> &coding_angles)
{
    const auto [id_offset, angle_offset] = offset;
    if (id_offset != -3)
    {
        return {size_t(id_offset + int(idx_middle_ring_in_coding)), angle_offset};
    }

    const size_t r_minus_idx = (idx_middle_ring_in_coding + 3) % coding_angles.size();
    if (coding_angles.at(idx_middle_ring_in_coding) > 0.f)
    {  // R- axis corresponds to the first coding's neighbour
        return {r_minus_idx, -angle_offset};
    }
    else  // if (coding_angles[idx_middle_ring_in_coding] < 0.f)
    {     // R- axis corresponds to the last coding's neighbour
        return {r_minus_idx, angle_offset};
    }
}

template <HexAxis kHexAxis, bool kIsCodingLeft>
std::optional<float> get_axis_rotation_when_all_6_detected(const std::vector<base::MarkerNeighborhood> &graph,
                                                           const int id_coding, const int id_middle_ring)
{
    auto coding_neighbors = graph[id_coding].neighbors;
    auto coding_angles = graph[id_coding].angles;
    const auto &middle_neighbors = graph[id_middle_ring].neighbors;

    if constexpr (!kIsCodingLeft)
    {
        std::ranges::rotate(coding_neighbors, coding_neighbors.begin() + 3);
        std::ranges::rotate(coding_angles, coding_angles.begin() + 3);
    }
    const auto it_middle_ring = std::ranges::find(coding_neighbors, id_middle_ring);
    const size_t idx_middle_ring_in_coding = std::distance(coding_neighbors.begin(), it_middle_ring);
    constexpr auto kOffsets =
        kIsCodingLeft ? kHexAxisDirectionOffsetsLeft[(int)kHexAxis] : kHexAxisDirectionOffsetsRight[(int)kHexAxis];

    // is_opposite==false corresponds to the direction which absolute value of it's angle from 'u' axis is < PI/2
    // is_opposite==true corresponds to the opposite direction f.e. for R+ is R-, for Q+ is Q- and for S+ is S-
    for (bool is_opposite : {false, true})
    {
        const auto [axis_ring_idx_in_coding, angle_offset] =
            get_idx_and_offset_of_axis(kOffsets[is_opposite], idx_middle_ring_in_coding, coding_angles);

        const int axis_ring_id = coding_neighbors.at(axis_ring_idx_in_coding);
        const bool is_neighbour_of_middle = middle_neighbors.end() != std::ranges::find(middle_neighbors, axis_ring_id);
        const bool cant_be_neighbour_of_middle = std::abs(kOffsets[is_opposite].first) != 1;
        if (cant_be_neighbour_of_middle || is_neighbour_of_middle)
        {
            return coding_angles.at(axis_ring_idx_in_coding) + angle_offset;
        }
        spdlog::warn(
            "The {} marker forming the '{}' axis together with the {} marker is not a neighbour of {} (middle) marker",
            axis_ring_id, kHexAxisNames[(int)kHexAxis], id_coding, id_middle_ring);
    }
    spdlog::warn("There is no marker that forms the '{}' axis together with the {} marker.",
                 kHexAxisNames[(int)kHexAxis], id_coding);

    return std::nullopt;
}

std::vector<std::optional<size_t>> get_opposite_indices(const std::vector<int> &neighbors,
                                                        const std::vector<float> &angles)
{
    std::vector<std::optional<size_t>> closest_opposite_indices(neighbors.size(), std::nullopt);
    for (size_t idx_1st = 0; idx_1st < neighbors.size(); ++idx_1st)
    {
        for (size_t idx_2nd = 0; idx_2nd < neighbors.size(); ++idx_2nd)
        {
            if (idx_1st == idx_2nd)
            {
                continue;
            }
            const float abs_diff = std::abs(angles[idx_2nd] - angles[idx_1st]);
            const float abs_diff_pi = std::abs(abs_diff - pi);
            if (kAngleThresh > abs_diff_pi)
            {
                closest_opposite_indices[idx_1st] = idx_2nd;
            }
        }
    }
    return closest_opposite_indices;
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

size_t count_non_parallel_directions(const std::vector<std::optional<size_t>> &opposite_indices)
{
    size_t non_parallel_directions = 0;
    std::set<size_t> unique;
    for (size_t idx = 0; idx < opposite_indices.size(); ++idx)
    {
        if (!opposite_indices[idx])
        {
            unique.insert(idx);
        }
        else if (!unique.contains(opposite_indices[idx].value()))
        {
            unique.insert(idx);
        }
    }
    non_parallel_directions = unique.size();
    return non_parallel_directions;
}

template <bool kIsCodingLeft>
std::array<std::optional<float>, 3> get_axis_rotations_when_at_least_one_missing(
    const std::vector<base::MarkerNeighborhood> &graph, const int id_coding, const int id_middle_ring)
{
    auto coding_neighbors = graph[id_coding].neighbors;
    auto coding_angles = graph[id_coding].angles;
    const auto &middle_neighbors = graph[id_middle_ring].neighbors;

    const auto it_middle_ring = std::ranges::find(coding_neighbors, id_middle_ring);
    const size_t idx_middle_ring_in_coding = std::distance(coding_neighbors.begin(), it_middle_ring);

    const auto closest_opposite_indices = get_opposite_indices(coding_neighbors, coding_angles);
    const size_t non_parallel_directions = count_non_parallel_directions(closest_opposite_indices);
    if (non_parallel_directions < 2)
    {
        spdlog::warn("Need at least 2 independent directions.");
        return {std::nullopt, std::nullopt, std::nullopt};
    }

    std::vector<float> angles_to_middle;
    std::vector<float> angles_axis;
    for (size_t idx = 0; idx < coding_angles.size(); ++idx)
    {
        const float angle_middle = unroll_angle(coding_angles[idx_middle_ring_in_coding] + (kIsCodingLeft ? 0.f : pi));
        if (idx == idx_middle_ring_in_coding)
        {
            angles_to_middle.push_back(0.f);
            angles_axis.push_back(angle_middle);
            continue;
        }
        float angle = coding_angles[idx];
        float diff = angle - angle_middle;
        if (std::abs(diff) > kAngleThresh && std::abs(diff) - kAngleThresh < pi / 2.f)
        {
            angle = coding_angles[idx] + pi;
            diff = angle - angle_middle;
        }
        if (std::abs(diff) + kAngleThresh > pi)
        {
            angle = coding_angles[idx] - pi;
            diff = angle - angle_middle;
        }
        angles_to_middle.push_back(diff);
        angles_axis.push_back(angle);
    }

    std::map<HexAxis, std::vector<size_t>> found_axes{
        {HexAxis::Q, {}},
        {HexAxis::R, {}},
        {HexAxis::S, {}},
    };
    for (size_t idx = 0; idx < angles_to_middle.size(); ++idx)
    {
        if (angles_to_middle[idx] < -pi / 2.f)
        {
            found_axes[HexAxis::Q].push_back(idx);
        }
        else if (angles_to_middle[idx] > pi / 2.f)
        {
            found_axes[HexAxis::S].push_back(idx);
        }
        else
        {
            found_axes[HexAxis::R].push_back(idx);
        }
    }

    if (2 <=
        int(found_axes[HexAxis::Q].empty()) + int(found_axes[HexAxis::R].empty()) + int(found_axes[HexAxis::S].empty()))
    {
        throw std::runtime_error("Interesting... Please report it to kornel.mrozowski@gmail.com");
    }

    std::array<std::optional<float>, 3> axes_averaged;
    size_t axis_idx = 0;
    for (HexAxis axis : {HexAxis::Q, HexAxis::R, HexAxis::S})
    {
        float sum = 0.f;
        for (const size_t idx : found_axes.at(axis))
        {
            sum += angles_axis[idx];
        }
        axes_averaged[axis_idx++] = sum / float(found_axes.at(axis).size());
    }

    return axes_averaged;
}

template <bool kIsCodingLeft>
std::array<std::optional<float>, 3> get_axis_rotations(const std::vector<base::MarkerNeighborhood> &graph,
                                                       const int id_coding, const int id_middle_ring)
{
    if (!std::ranges::is_sorted(graph[id_coding].angles) && graph[id_coding].angles[0] < graph[id_coding].angles[1])
    {
        spdlog::warn("The {} marker's angles should be already sorted", id_coding);
        return {};
    }

    if (graph[id_coding].neighbors.size() == 1)
    {
        spdlog::warn("The {} coding marker has one neighbor ", id_coding);
        return {std::nullopt, std::nullopt, std::nullopt};
    }

    if (graph[id_coding].neighbors.size() == 6)
    {
        auto rotation_q =
            get_axis_rotation_when_all_6_detected<HexAxis::Q, kIsCodingLeft>(graph, id_coding, id_middle_ring);
        auto rotation_r =
            get_axis_rotation_when_all_6_detected<HexAxis::R, kIsCodingLeft>(graph, id_coding, id_middle_ring);
        auto rotation_s =
            get_axis_rotation_when_all_6_detected<HexAxis::S, kIsCodingLeft>(graph, id_coding, id_middle_ring);
        return {rotation_q, rotation_r, rotation_s};
    }
    return get_axis_rotations_when_at_least_one_missing<kIsCodingLeft>(graph, id_coding, id_middle_ring);
}

std::optional<float> get_closest_angle(const std::optional<float> &angle_1st, const std::vector<float> &angles,
                                       const float angle_thresh)
{
    std::vector<float> possible_angles;
    if (!angle_1st)
    {
        return std::nullopt;
    }
    for (const float angle_2nd : angles)
    {
        if (angle_thresh > std::abs(angle_2nd - angle_1st.value()))
        {
            possible_angles.insert(possible_angles.begin(), angle_2nd);
        }
        else if (angle_thresh > std::abs(angle_2nd + pi - angle_1st.value()))
        {
            possible_angles.push_back(angle_2nd + pi);
        }
        else if (angle_thresh > std::abs(angle_2nd - pi - angle_1st.value()))
        {
            possible_angles.push_back(angle_2nd - pi);
        }
    }
    if (possible_angles.empty())
    {
        return std::nullopt;
    }
    return possible_angles.front();
}

void print_axial_stats(const std::vector<base::MarkerNeighborhood> &graph, const std::vector<AxialCoord> &axial_coords)
{
    int num_no_neighbors = 0;
    int num_no_val = 0;
    int num_nok = 0;
    int num_zero = 0;
    for (size_t id_1st = 0; id_1st < axial_coords.size(); ++id_1st)
    {
        if (graph[id_1st].neighbors.empty())
        {
            ++num_no_neighbors;
            continue;
        }
        if (!axial_coords[id_1st].has_value())
        {
            ++num_no_val;
            continue;
        }
        if (!axial_coords[id_1st].is_ok())
        {
            ++num_nok;
            continue;
        }
        if (axial_coords[id_1st].q == 0 && axial_coords[id_1st].r == 0 && axial_coords[id_1st].s == 0)
        {
            ++num_zero;
            continue;
        }
    }
    spdlog::debug("num_no_neighbors={}, num_no_val={}, num_nok={}, num_zero={}", num_no_neighbors, num_no_val, num_nok,
                  num_zero);
}

template <bool kIsCodingLeft>
std::vector<AxialCoord> make_axial_coords(const std::vector<base::MarkerNeighborhood> &graph, const int id_coding,
                                          const int id_middle_ring)
{
    std::vector<AxialCoord> axial_coords(graph.size());
    auto [rotation_q, rotation_r, rotation_s] = get_axis_rotations<kIsCodingLeft>(graph, id_coding, id_middle_ring);
    if (2 > int(rotation_q.has_value()) + int(rotation_r.has_value()) + int(rotation_s.has_value()))
    {
        return {};
    }

    if constexpr (kIsCodingLeft)
    {
        axial_coords[id_coding] = AxialCoord(0, 0, 0, rotation_q, rotation_r, rotation_s);
    }
    else
    {
        axial_coords[id_coding] = AxialCoord(2, 0, -2, rotation_q, rotation_r, rotation_s);
    }

    const auto propagate_angle = [&graph](auto &axis_b, const int id_b, const auto &axis_a)
    {
        const auto rot_axis_b = get_closest_angle(axis_a, graph[id_b].angles, kAngleThresh);
        axis_b = rot_axis_b.has_value() ? rot_axis_b : axis_b;
        axis_b = axis_b.has_value() ? axis_b : axis_a;
    };

    std::queue<int> bfs_queue;
    bfs_queue.push(id_coding);
    while (!bfs_queue.empty())
    {
        const int id_a = bfs_queue.front();
        bfs_queue.pop();
        auto &[q_a, r_a, s_a, axis_q_a, axis_r_a, axis_s_a] = axial_coords[id_a];
        for (size_t idx_b = 0; idx_b < graph[id_a].neighbors.size(); ++idx_b)
        {
            const int id_b = graph[id_a].neighbors[idx_b];
            const float angle_b = graph[id_a].angles[idx_b];
            auto &[q_b, r_b, s_b, axis_q_b, axis_r_b, axis_s_b] = axial_coords[id_b];
            if (axial_coords[id_b].has_value())
            {
                continue;
            }
            if (axis_q_a && kAngleThresh > std::abs(angle_b - axis_q_a.value()))
            {  // Q+
                axis_q_b = angle_b;
                q_b = q_a;
                r_b = r_a - 1;
                s_b = s_a + 1;
            }
            else if (axis_q_a && (kAngleThresh > std::abs(angle_b - pi - axis_q_a.value()) ||
                                  kAngleThresh > std::abs(angle_b + pi - axis_q_a.value())))
            {  // Q-
                axis_q_b = (kAngleThresh > std::abs(angle_b - pi - axis_q_a.value())) ? angle_b - pi : angle_b + pi;
                q_b = q_a;
                r_b = r_a + 1;
                s_b = s_a - 1;
            }
            else if (axis_r_a && kAngleThresh > std::abs(angle_b - axis_r_a.value()))
            {  // R+
                axis_r_b = angle_b;
                q_b = q_a + 1;
                r_b = r_a;
                s_b = s_a - 1;
            }
            else if (axis_r_a && (kAngleThresh > std::abs(angle_b + pi - axis_r_a.value()) ||
                                  kAngleThresh > std::abs(angle_b - pi - axis_r_a.value())))
            {  // R-
                axis_r_b = (kAngleThresh > std::abs(angle_b + pi - axis_r_a.value())) ? angle_b + pi : angle_b - pi;
                q_b = q_a - 1;
                r_b = r_a;
                s_b = s_a + 1;
            }
            else if (axis_s_a && kAngleThresh > std::abs(angle_b - axis_s_a.value()))
            {  // S+
                axis_s_b = angle_b;
                q_b = q_a - 1;
                r_b = r_a + 1;
                s_b = s_a;
            }
            else if (axis_s_a && (kAngleThresh > std::abs(angle_b + pi - axis_s_a.value()) ||
                                  kAngleThresh > std::abs(angle_b - pi - axis_s_a.value())))
            {  // S-
                axis_s_b = (kAngleThresh > std::abs(angle_b + pi - axis_s_a.value())) ? angle_b + pi : angle_b - pi;
                q_b = q_a + 1;
                r_b = r_a - 1;
                s_b = s_a;
            }
            else
            {
                spdlog::debug("None of +q, +r, +s, -q, -r, -s");
                continue;
            }
            propagate_angle(axis_q_b, id_b, axis_q_a);
            propagate_angle(axis_r_b, id_b, axis_r_a);
            propagate_angle(axis_s_b, id_b, axis_s_a);
            const bool has_2_axes = axial_coords[id_b].has_value();
            if (!has_2_axes)
            {
                continue;
            }

            bfs_queue.push(id_b);
        }
    }
    print_axial_stats(graph, axial_coords);

    return axial_coords;
}

std::vector<std::optional<OffsetCoord>> make_offset_coords(const std::vector<AxialCoord> &axial_coords,
                                                           const bool is_even)
{
    std::vector<std::optional<OffsetCoord>> offset_coords(axial_coords.size());
    const auto operation = is_even ? [](const int r_coord, const bool parity) { return r_coord + parity; }
                                   : [](const int r_coord, const bool parity) { return r_coord - parity; };
    for (size_t idx = 0; idx < axial_coords.size(); ++idx)
    {
        if (!axial_coords[idx].has_value())
        {
            continue;
        }
        const int col = axial_coords[idx].q + operation(axial_coords[idx].r, (axial_coords[idx].r & 1) == 1) / 2;
        const int row = axial_coords[idx].r;
        offset_coords[idx] = OffsetCoord(col, row);
    }
    return offset_coords;
}

std::vector<int> assign_global_indices(const std::vector<std::optional<OffsetCoord>> &offset_coords,
                                       const BoardHexGrid &board)
{
    std::vector<int> markers_order(offset_coords.size());
    for (size_t idx = 0; idx < offset_coords.size(); ++idx)
    {
        if (!offset_coords[idx].has_value())
        {
            markers_order[idx] = -1;
            continue;
        }
        const int col = offset_coords[idx]->col + board.col_left_;
        const int row = offset_coords[idx]->row + board.row_left_;
        markers_order[idx] = col + row * board.cols_;
    }
    return markers_order;
}

std::optional<int> get_common_neighbour(const std::vector<base::MarkerNeighborhood> &graph,
                                        const std::pair<int, int> coding_pair)
{
    auto neighbors_left = graph[coding_pair.first].neighbors;
    auto neighbors_right = graph[coding_pair.second].neighbors;
    std::ranges::sort(neighbors_left);
    std::ranges::sort(neighbors_right);
    std::vector<int> common_neighbors;
    std::ranges::set_intersection(neighbors_left, neighbors_right, std::back_inserter(common_neighbors));
    if (common_neighbors.size() == 1)
    {
        return common_neighbors.front();
    }
    return std::nullopt;
}

}  // namespace

std::vector<int> identification::create_markers_order(const std::vector<base::MarkerNeighborhood> &graph,
                                                      const std::vector<base::MarkerRing> markers,
                                                      const std::pair<int, int> coding_pair, const BoardHexGrid &board)
{
    auto common_neighbour = get_common_neighbour(graph, coding_pair);
    if (!common_neighbour)
    {
        spdlog::warn("Coding markers does not share common neighbour.");
        return {};
    }
    std::vector<AxialCoord> axial_coords = make_axial_coords<true>(graph, coding_pair.first, common_neighbour.value());
    if (axial_coords.empty())
    {
        axial_coords = make_axial_coords<false>(graph, coding_pair.second, common_neighbour.value());
    }

    const std::vector<std::optional<OffsetCoord>> offset_coords = make_offset_coords(axial_coords, board.is_even_);
    const std::vector<int> markers_order = assign_global_indices(offset_coords, board);

    return markers_order;
}
