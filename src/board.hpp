#pragma once

#include <filesystem>
#include <memory>
#include <variant>

#include <Eigen/Dense>

enum class BoardType
{
    RECT = 0,
    HEX = 1,
};
class Board
{
   public:
    BoardType type_;

    int cols_;
    int rows_;

    float inner_radius_;  // [mm]
    float outer_radius_;  // [mm]

    float spacing_cols_;  // [mm]
    float spacing_rows_;  // [mm]

    std::vector<Eigen::Vector3d> marker_centers_;

    Eigen::Vector3d top_left_;      // [mm]
    Eigen::Vector3d bottom_right_;  // [mm]

    virtual ~Board() = default;
    Eigen::Vector2i id_to_row_and_col(const int id) const;
    int row_and_col_to_id(const int row, const int col) const;
    void apply_scale(const float factor);

    virtual std::string_view name() { return "dummy"; }
};

/**
 * @brief Define calibration board base on concentric circles.
 *
 * 0 1 2 are big black markers, and N is called "non-unique" (becaus it's ring marker) but it forms with unique markers
 * quad in board domain.
 *
 *  board in ordering of
 *  ------------>
 * | * * * * * *
 * | * 0 * * N *
 * | * * * * * *
 * | * 1 * * 2 *
 * V * * * * * *
 *
 * board indexing
 * ------------>
 * | 0  1  2 * * 29
 * | 30 31 * * * *
 * | * * * * * * *
 * | * * * * * * *
 * V * * * * * * *
 *
 */
class BoardRectGrid : public Board
{
   public:
    struct Params
    {
        int rows = 17;
        int cols = 23;
        float inner_radius = 2.5f;     // [mm]
        float outer_radius = 5.f;      // [mm]
        float spacing_rows = 16.217f;  // [mm]
        float spacing_cols = 16.217f;  // [mm]

        int row_top = 4;
        int col_top = 4;
        int row_down = 12;
        int col_down = 4;
        int row_right = 12;
        int col_right = 18;
        int row_non_unique = 4;
        int col_non_unique = 18;
        float edge_average_difference_allowed = 0.5f;
    };

    int row_top_;
    int col_top_;

    int row_down_;
    int col_down_;

    int row_right_;
    int col_right_;

    int row_non_unique_;
    int col_non_unique_;

    float edge_average_difference_allowed_;

    BoardRectGrid();
    BoardRectGrid(const Params& params);

    int marker_distance_01() const;
    int marker_distance_12() const;

    // return row/col locations of line between X-Y coding locations (exclude coding markers)
    std::vector<std::pair<int, int>> marker_lines_01_locations() const;
    std::vector<std::pair<int, int>> marker_lines_12_locations() const;

    std::string_view name() override { return "RectRingCalibTarget"; }

   private:
    void init_params(const Params& params);
    void setup_markers();
};

/**
 * @brief Define calibration board base on concentric circles.
 *
 * 0 1 are invisible markers
 *
 *  board in ordering of
 *  ------------>
 * | * * * * * *
 * |  * * * * * *
 * | * * * * * *
 * |  * 0 * 1 * *
 * | * * * * * *
 * V  * * * * * *
 *
 * board indexing
 * ------------>
 * | 000 001 002  *   *   028
 * |   029 030  *   *   *    *
 * |  *   *   *   *   *    *
 * |    *   *   *   *   *    *
 * V  *   *   *   *   *    *
 *
 * The single marker of the board:
 * |__________________10mm___________________|
 * |       |___________5mm___________|       |
 * |       |                         |       |
 * |       |      ooooooooooooo      |       |
 * |       |  ooooooooooooooooooooo  |       |
 * |       |ooooooooooooooooooooooooo|       |
 * |     oo|ooooooooooooooooooooooooo|oo     |
 * |   oooo|ooooooo           ooooooo|oooo   |
 * |  ooooo|oooo                 oooo|oooo#  |
 * | oooooo|oo                     oo|oooooo |
 * |ooooooo|o                       o|ooooooo|
 * |ooooooo|                         |ooooooo|
 * |ooooooo|                         |ooooooo|
 * |ooooooo|                         |ooooooo|
 * ooooooooo                         ooooooooo
 *  oooooooo                         oooooooo
 *  ooooooooo                       ooooooooo
 *   ooooooooo                     ooooooooo
 *    oooooooooo                 oooooooooo
 *     oooooooooooo           oooooooooooo
 *       ooooooooooooooooooooooooooooooo
 *         ooooooooooooooooooooooooooo
 *            ooooooooooooooooooooo
 *                ooooooooooooo
 */
class BoardHexGrid : public Board
{
   public:
    struct Params
    {
        int rows = 29;
        int cols = 35;
        int row_left = 14;
        int col_left = 15;
        int row_right = 14;
        int col_right = 17;
        float spacing_cols = 11.6f;  // [mm]
        float inner_radius = 2.4f;   // [mm]
        float outer_radius = 4.8f;   // [mm]
        bool is_even = true;
    };

    int row_left_;
    int col_left_;
    int row_right_;
    int col_right_;

    /**
     *  Even:
     * | * * * * * *
     * |  * L * R * *
     * | * * * * * *
     * |  * * * * * *
     *  Odd:
     * |  * * * * * *
     * | * L * R * *
     * |  * * * * * *
     * | * * * * * *
     */
    bool is_even_;

    BoardHexGrid();
    BoardHexGrid(const Params& params);

    int marker_distance_01() const;

    // return row/col locations of line between X-Y coding locations (exclude coding markers)
    std::vector<std::pair<int, int>> marker_lines_01_locations() const;

    std::string_view name() override { return "HexRingCalibTarget"; }

   private:
    void init_params(const Params& params);
    void setup_markers();
};

namespace board
{

BoardRectGrid::Params get_rect_params(const std::vector<std::variant<int, float>>& board_params);
BoardHexGrid::Params get_hex_params(const std::vector<std::variant<int, float>>& board_params);

std::unique_ptr<Board> get_board(const int board_type, const BoardHexGrid::Params& board_params);
std::unique_ptr<Board> get_board(const int board_type, const std::filesystem::path& directory_path);
std::unique_ptr<Board> get_board(const int board_type, const std::vector<std::variant<int, float>>& board_params_vec);

};  // namespace board
