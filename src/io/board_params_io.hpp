#pragma once

#include <filesystem>

#include "../board.hpp"

namespace io
{

struct BoardParams
{
    BoardType type_;
    BoardRectGrid::Params params_rect;
    BoardHexGrid::Params params_hex;
    BoardCircleGrid::Params params_circle;
};

BoardParams read_params(const std::filesystem::path& filepath);

}  // namespace io
