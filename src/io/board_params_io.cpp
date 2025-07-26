#include "board_params_io.hpp"

#include <format>
#include <fstream>

#include <nlohmann/json.hpp>

#include "../board.hpp"

namespace rect
{
constexpr std::string_view kRows = "rows";
constexpr std::string_view kCols = "cols";
constexpr std::string_view kInnerRadius = "inner_radius";
constexpr std::string_view kOuterRadius = "outer_radius";
constexpr std::string_view kSpacingRows = "spacing_rows";
constexpr std::string_view kSpacingCols = "spacing_cols";
constexpr std::string_view kRowTop = "row_top";
constexpr std::string_view kColTop = "col_top";
constexpr std::string_view kRowDown = "row_down";
constexpr std::string_view kColDown = "col_down";
constexpr std::string_view kRowRight = "row_right";
constexpr std::string_view kColRight = "col_right";
constexpr std::string_view kRowNonUnique = "row_non_unique";
constexpr std::string_view kColNonUnique = "col_non_unique";
constexpr std::string_view kEdgeAverageDifferenceAllowed = "edge_average_difference_allowed";
}  // namespace rect

namespace hex
{
constexpr std::string_view kRows = "rows";
constexpr std::string_view kCols = "cols";
constexpr std::string_view kRowLeft = "row_left";
constexpr std::string_view kColLeft = "col_left";
constexpr std::string_view kRowRight = "row_right";
constexpr std::string_view kColRight = "col_right";
constexpr std::string_view kSpacingCols = "spacing_cols";
constexpr std::string_view kInnerRadius = "inner_radius";
constexpr std::string_view kOuterRadius = "outer_radius";
constexpr std::string_view kIsEven = "is_even";
}  // namespace hex

namespace
{

static const std::map<std::string_view, BoardType> kNameToBoardType{
    {"rect", BoardType::RECT},
    {"hex", BoardType::HEX},
};

static constexpr std::string_view kBoardParamsJsonName = "board";
static constexpr std::string_view kType = "type";

io::BoardVariants read_params_rect(const nlohmann::json& json)
{
    return BoardRectGrid::Params{
        .rows = json[rect::kRows].get<int>(),
        .cols = json[rect::kCols].get<int>(),
        .inner_radius = json[rect::kInnerRadius].get<float>(),
        .outer_radius = json[rect::kOuterRadius].get<float>(),
        .spacing_rows = json[rect::kSpacingRows].get<float>(),
        .spacing_cols = json[rect::kSpacingCols].get<float>(),
        .row_top = json[rect::kRowTop].get<int>(),
        .col_top = json[rect::kColTop].get<int>(),
        .row_down = json[rect::kRowDown].get<int>(),
        .col_down = json[rect::kColDown].get<int>(),
        .row_right = json[rect::kRowRight].get<int>(),
        .col_right = json[rect::kColRight].get<int>(),
        .row_non_unique = json[rect::kRowNonUnique].get<int>(),
        .col_non_unique = json[rect::kColNonUnique].get<int>(),
        .edge_average_difference_allowed = json[rect::kEdgeAverageDifferenceAllowed].get<float>(),
    };
}

std::variant<BoardRectGrid::Params, BoardHexGrid::Params> read_params_hex(const nlohmann::json& json)
{
    return BoardHexGrid::Params{
        .rows = json[hex::kRows.data()].get<int>(),
        .cols = json[hex::kCols.data()].get<int>(),
        .row_left = json[hex::kRowLeft.data()].get<int>(),
        .col_left = json[hex::kColLeft.data()].get<int>(),
        .row_right = json[hex::kRowRight.data()].get<int>(),
        .col_right = json[hex::kColRight.data()].get<int>(),
        .spacing_cols = json[hex::kSpacingCols.data()].get<float>(),
        .inner_radius = json[hex::kInnerRadius.data()].get<float>(),
        .outer_radius = json[hex::kOuterRadius.data()].get<float>(),
        .is_even = json[hex::kIsEven.data()].get<bool>(),
    };
}

}  // namespace

template <typename BoardParamsType>
BoardParamsType io::read_params(const std::filesystem::path& directory_path)
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

template BoardRectGrid::Params io::read_params(const std::filesystem::path& directory_path);
template BoardHexGrid::Params io::read_params(const std::filesystem::path& directory_path);
