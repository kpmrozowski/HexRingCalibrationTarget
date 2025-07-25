#pragma once

#include <memory>

#include "board.hpp"
#include "ordering.hpp"

namespace base
{
struct MarkerRing;
struct MarkerCoding;
struct MarkerNeighborhood;
}  // namespace base

namespace marker
{
struct DetectionParameters;
}  // namespace marker

class Board;
class BoardRectGrid;
class BoardHexGrid;

namespace identification
{
struct Decoded
{
    BoardType ordering_type;
    OrderingBoardRect ordering_rect;
    OrderingBoardHex ordering_hex;
    std::vector<base::MarkerRing> markers;
    int identified_markers;

    const Eigen::Matrix<std::optional<int>, -1, -1> &ordering() const;
    Eigen::Matrix<std::optional<int>, -1, -1> &ordering();
};

/**
 * @brief Given set of coding marker (unordered) we iterate all configurations of id assignment that can be made from
 * marker set and try to derive global indexing base on it.
 *
 * @param coding set of markers that we assume can be coding (unordered, can be more that 3)
 * @param rings set of markers that we assume are rings marker (unordered, can hold arbitrary large number [if not
 * decoded, they will not be assigned])
 * @param board definition of pattern
 */
std::pair<std::optional<identification::Decoded>, std::vector<base::MarkerNeighborhood>> assign_global_IDs(
    const std::vector<base::MarkerCoding> &coding, const std::vector<base::MarkerRing> &rings,
    const std::unique_ptr<Board> &board);

std::pair<std::optional<identification::Decoded>, std::vector<base::MarkerNeighborhood>> assign_global_IDs(
    const std::vector<base::MarkerCoding> &coding, const std::vector<base::MarkerRing> &rings,
    const BoardRectGrid &board);

std::pair<std::optional<identification::Decoded>, std::vector<base::MarkerNeighborhood>> assign_global_IDs(
    const std::vector<base::MarkerCoding> &coding, const std::vector<base::MarkerRing> &rings,
    const BoardHexGrid &board);

}  // namespace identification
