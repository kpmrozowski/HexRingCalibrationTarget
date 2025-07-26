#include "board.hpp"
#include <variant>

namespace
{
Eigen::Vector3d marker_localization(const BoardRectGrid& board, const int row, const int col)
{
    return board.top_left_ + Eigen::Vector3d(col * board.spacing_cols_, row * board.spacing_rows_, 0.0);
}

Eigen::Vector3d marker_localization(const BoardHexGrid& board, const int row, const int col)
{
    if (row % 2 == 0)
    {
        return board.top_left_ + Eigen::Vector3d(col * board.spacing_cols_, row * board.spacing_rows_, 0.0);
    }
    else
    {
        return board.top_left_ + Eigen::Vector3d((col + 0.5) * board.spacing_cols_, row * board.spacing_rows_, 0.0);
    }
}

}  // namespace

// Board

Eigen::Vector2i Board::id_to_row_and_col(const int id) const
{
    const int row = id / cols_;
    const int col = id - row * cols_;
    return {row, col};
}

int Board::row_and_col_to_id(const int row, const int col) const
{
    const int marker_id = row * cols_ + col;
    return marker_id;
}

void Board::apply_scale(const float factor)
{
    inner_radius_ *= factor;
    outer_radius_ *= factor;
    spacing_rows_ *= factor;
    spacing_cols_ *= factor;
    top_left_ *= factor;
    bottom_right_ *= factor;
    for (int row = 0; row < rows_; ++row)
    {
        for (int col = 0; col < cols_; ++col)
        {
            marker_centers_[col + row * cols_] *= factor;
        }
    }
}

// BoardRectGrid

int BoardRectGrid::marker_distance_01() const { return row_down_ - row_top_; }

int BoardRectGrid::marker_distance_12() const { return col_right_ - col_down_; }

std::vector<std::pair<int, int>> BoardRectGrid::marker_lines_01_locations() const
{
    std::vector<std::pair<int, int>> ordering_markers;
    for (int row = row_top_ + 1; row < row_down_; ++row)
    {
        ordering_markers.emplace_back(row, col_top_);
    }
    return ordering_markers;
}

std::vector<std::pair<int, int>> BoardRectGrid::marker_lines_12_locations() const
{
    std::vector<std::pair<int, int>> ordering_markers;
    for (int col = col_down_ + 1; col < col_right_; ++col)
    {
        ordering_markers.emplace_back(row_down_, col);
    }
    return ordering_markers;
}

BoardRectGrid::BoardRectGrid()
{
    type_ = BoardType::RECT;
    rows_ = 17;
    cols_ = 23;
    inner_radius_ = 2.5f;     // [mm]
    outer_radius_ = 5.f;      // [mm]
    spacing_rows_ = 16.217f;  // [mm]
    spacing_cols_ = 16.217f;  // [mm]
    top_left_ = Eigen::Vector3d(spacing_cols_ + outer_radius_, spacing_rows_ + outer_radius_, 0);
    bottom_right_ = Eigen::Vector3d((cols_ + 1.0) * spacing_cols_ + outer_radius_,
                                    (rows_ + 1.0) * spacing_rows_ + outer_radius_, 0.0);

    for (int row = 0; row < rows_; ++row)
    {
        for (int col = 0; col < cols_; ++col)
        {
            marker_centers_.emplace_back(marker_localization(*this, row, col));
        }
    }
}

// BoardHexGrid

int BoardHexGrid::marker_distance_01() const { return col_left_ - col_right_; }

std::vector<std::pair<int, int>> BoardHexGrid::marker_lines_01_locations() const
{
    std::vector<std::pair<int, int>> ordering_markers;
    for (int col = col_left_; col < col_right_; ++col)
    {
        ordering_markers.emplace_back(row_left_, col);
    }
    return ordering_markers;
}

BoardHexGrid::BoardHexGrid()
{
    init_params(Params{});
    setup_markers();
}

BoardHexGrid::BoardHexGrid(const Params& params)
{
    init_params(params);
    setup_markers();
}

void BoardHexGrid::init_params(const Params& params)
{
    type_ = BoardType::HEX;
    rows_ = params.rows;
    cols_ = params.cols;
    row_left_ = params.row_left;
    col_left_ = params.col_left;
    row_right_ = params.row_right;
    col_right_ = params.col_right;
    spacing_cols_ = params.spacing_cols;
    inner_radius_ = params.inner_radius;
    outer_radius_ = params.outer_radius;
    is_even_ = params.is_even;
}

void BoardHexGrid::setup_markers()
{
    static constexpr float kEquilateralTriangleHeight = std::numbers::sqrt3_v<float> / 2.f;

    spacing_rows_ = kEquilateralTriangleHeight * spacing_cols_;
    top_left_ = 2. * Eigen::Vector3d(outer_radius_, outer_radius_, 0);
    bottom_right_ =
        Eigen::Vector3d((cols_ + 0.5) * spacing_cols_ + outer_radius_, rows_ * spacing_rows_ + outer_radius_, 0.0);

    for (int row = 0; row < rows_; ++row)
    {
        for (int col = 0; col < cols_; ++col)
        {
            marker_centers_.emplace_back(marker_localization(*this, row, col));
        }
    }
}

BoardHexGrid::Params board::get_hex_params(const std::vector<std::variant<int, float>>& board_params)
{
    BoardHexGrid::Params params;
    switch (board_params.size())
    {
        case 10:
            params.is_even = std::get<int>(board_params.at(9)) != 0;
        case 9:
            params.outer_radius = std::get<float>(board_params.at(8));
        case 8:
            params.inner_radius = std::get<float>(board_params.at(7));
        case 7:
            params.spacing_cols = std::get<float>(board_params.at(6));
        case 6:
            params.col_right = std::get<int>(board_params.at(5));
        case 5:
            params.row_right = std::get<int>(board_params.at(4));
        case 4:
            params.col_left = std::get<int>(board_params.at(3));
        case 3:
            params.row_left = std::get<int>(board_params.at(2));
        case 2:
            params.cols = std::get<int>(board_params.at(1));
        case 1:
            params.rows = std::get<int>(board_params.at(0));
    }

    return params;
}

std::unique_ptr<Board> board::get_board(const int board_type, const std::vector<std::variant<int, float>>& board_params)
{
    std::unique_ptr<Board> calibration_board;
    switch ((BoardType)board_type)
    {
        case BoardType::RECT:
            return std::make_unique<BoardRectGrid>();
        case BoardType::HEX:
            return std::make_unique<BoardHexGrid>(get_hex_params(board_params));
    }
}
