#pragma once

#include <filesystem>
#include <map>
// #include <nlohmann/json_fwd.hpp>

#include "../board.hpp"

namespace io
{
using BoardVariants = std::variant<BoardRectGrid::Params, BoardHexGrid::Params>;

template <typename BoardParamsType>
BoardParamsType read_params(const std::filesystem::path& directory_path);
}  // namespace io

/*



static const std::map<std::string_view, BoardType> kNameToBoardType{
    {"rect", BoardType::RECT},
    {"hex", BoardType::HEX},
};

static constexpr std::string_view kBoardParamsJsonName = "board";
static constexpr std::string_view kType = "type";

nlohmann::json read_json(const std::filesystem::path& directory_path);
BoardVariants read_params_rect(const nlohmann::json& json);
BoardVariants read_params_hex(const nlohmann::json& json);

template <typename BoardParamsType>
BoardParamsType read_params(const std::filesystem::path& directory_path)
{
    // Load dataset metadata
    std::ifstream file(directory_path / (std::string(kBoardParamsJsonName) + ".json"));
    nlohmann::json json;
    file >> json;

    const std::string board_type_str = json[kType.data()].get<std::string>();
    if (!kNameToBoardType.contains(board_type_str))
    {
        throw std::invalid_argument(
            std::format("Error loading '{}.json'. Unknown board type: {}.", kBoardParamsJsonName, board_type_str));
    }

    switch (kNameToBoardType.at(board_type_str))
    {
        case BoardType::RECT:
            return std::get<BoardParamsType>(read_params_rect(json));
        case BoardType::HEX:
            return std::get<BoardParamsType>(read_params_hex(json));
    }
}
*/