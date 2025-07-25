#include "board.hpp"

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
    inner_radius_ = 0.25f;
    outer_radius_ = 0.5f;
    spacing_rows_ = 1.6217f;
    spacing_cols_ = 1.6217f;
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
    init_default();
    setup_markers();
}

BoardHexGrid::BoardHexGrid(const Params& params)
{
    init_default();

    rows_ = params.rows;
    cols_ = params.cols;
    row_left_ = params.row_left;
    col_left_ = params.col_left;
    row_right_ = params.row_right;
    col_right_ = params.col_right;

    setup_markers();
}

void BoardHexGrid::init_default()
{
    static constexpr float kEquilateralTriangleHeight = std::numbers::sqrt3_v<float> / 2.f;
    type_ = BoardType::HEX;
    Params default_params;
    rows_ = default_params.rows;
    cols_ = default_params.cols;
    inner_radius_ = 0.24f;
    outer_radius_ = 0.48f;
    spacing_cols_ = 1.16f;
    spacing_rows_ = kEquilateralTriangleHeight * spacing_cols_;
    top_left_ = 2. * Eigen::Vector3d(outer_radius_, outer_radius_, 0);
    bottom_right_ =
        Eigen::Vector3d((cols_ + 0.5) * spacing_cols_ + outer_radius_, rows_ * spacing_rows_ + outer_radius_, 0.0);

    row_left_ = default_params.row_left;
    col_left_ = default_params.col_left;
    row_right_ = default_params.row_right;
    col_right_ = default_params.col_right;
}

void BoardHexGrid::setup_markers()
{
    for (int row = 0; row < rows_; ++row)
    {
        for (int col = 0; col < cols_; ++col)
        {
            marker_centers_.emplace_back(marker_localization(*this, row, col));
        }
    }
}

BoardHexGrid::Params board::get_hex_params(const std::vector<int>& board_params)
{
    BoardHexGrid::Params params;
    switch (board_params.size())
    {
        case 6:
            params.col_right = board_params.at(5);
        case 5:
            params.row_right = board_params.at(4);
        case 4:
            params.col_left = board_params.at(3);
        case 3:
            params.row_left = board_params.at(2);
        case 2:
            params.cols = board_params.at(1);
        case 1:
            params.rows = board_params.at(0);
    }

    return params;
}

std::unique_ptr<Board> board::get_board(const int board_type, const std::vector<int>& board_params)
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
