#include "board_params_io.hpp"
#include <spdlog/spdlog.h>

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

namespace circle
{
constexpr std::string_view kRows = "rows";
constexpr std::string_view kCols = "cols";
constexpr std::string_view kSpacing = "spacing";
constexpr std::string_view kRadius = "radius";
constexpr std::string_view kIsAsymetric = "is_asymetric";
constexpr std::string_view kPaddingMm = "padding_mm";
}  // namespace circle

namespace
{

static const std::map<std::string_view, BoardType> kNameToBoardType{
    {"rect", BoardType::RECT},
    {"hex", BoardType::HEX},
    {"circle", BoardType::CIRCLE},
};

static constexpr std::string_view kType = "type";

BoardRectGrid::Params read_params_rect(const nlohmann::json& json)
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

BoardHexGrid::Params read_params_hex(const nlohmann::json& json)
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

BoardCircleGrid::Params read_params_circle(const nlohmann::json& json)
{
    return BoardCircleGrid::Params{
        .rows = json[circle::kRows.data()].get<int>(),
        .cols = json[circle::kCols.data()].get<int>(),
        .radius = json[circle::kRadius.data()].get<float>(),
        .spacing = json[circle::kSpacing.data()].get<float>(),
        .is_asymetric = json[circle::kIsAsymetric.data()].get<bool>(),
        .padding_mm = Eigen::Map<Eigen::Vector2i>{json[circle::kPaddingMm.data()].get<std::array<int, 2>>().data()}};
}

}  // namespace

io::BoardParams io::read_params(const std::filesystem::path& filepath)
{
    const auto board_params_filepath = filepath;
    if (!std::filesystem::exists(board_params_filepath))
    {
        throw std::invalid_argument(
            std::format("Error loading '{}'. File does not exist.", board_params_filepath.string()));
    }

    // Load dataset metadata
    std::ifstream file(board_params_filepath);
    nlohmann::json json;
    file >> json;

    const std::string board_type_str = json[kType.data()].get<std::string>();
    if (!kNameToBoardType.contains(board_type_str))
    {
        throw std::invalid_argument(
            std::format("Error loading '{}.json'. Unknown board type: {}.", filepath.string(), board_type_str));
    }

    BoardParams params;
    params.type_ = kNameToBoardType.at(board_type_str);
    spdlog::info("board_type: {}", board_type_str);

    switch (params.type_)
    {
        case BoardType::RECT:
            params.params_rect = read_params_rect(json);
            break;
        case BoardType::HEX:
            params.params_hex = read_params_hex(json);
            break;
        case BoardType::CIRCLE:
            params.params_circle = read_params_circle(json);
            break;
    }
    return params;
}
