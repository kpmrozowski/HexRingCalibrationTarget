#include "draw_board.hpp"

#include <opencv2/imgcodecs.hpp>

#include <spdlog/spdlog.h>
#include <cmd_launcher/subcommand.hpp>
#include <io/save_path.hpp>
#include "board_drawing.hpp"

namespace
{

constexpr float kInchesPerMeter = 39.36;
constexpr float kInchesPerCentymeter = kInchesPerMeter / 100.;
constexpr float kPointsPerInch = 72.00;
constexpr float kPointsPerCentymeter = kPointsPerInch * kInchesPerCentymeter;
constexpr float kMagicDpiScale = 2710 / 420.;
const std::map<BoardType, std::string_view> kTypeToName{};

/** @brief Point is a smallest unit of measurement in typography
 * @ref https://www.a4-size.com/a4-size-in-point
 */
cv::Size_<float> get_board_dimention_cm(const int a_x)
{
    cv::Size resolution;
    switch (a_x)
    {
        case -2:
            return cv::Size_<float>(6742, 4768) / kPointsPerCentymeter;
        case -1:
            return cv::Size_<float>(4768, 3371) / kPointsPerCentymeter;
        case 0:
            return cv::Size_<float>(3371, 2384) / kPointsPerCentymeter;
        case 1:
            return cv::Size_<float>(2384, 1684) / kPointsPerCentymeter;
        case 2:
            return cv::Size_<float>(1684, 1191) / kPointsPerCentymeter;
        case 3:
            return cv::Size_<float>(1191, 842) / kPointsPerCentymeter;
        case 4:
            return cv::Size_<float>(842, 595) / kPointsPerCentymeter;
        case 5:
            return cv::Size_<float>(595, 420) / kPointsPerCentymeter;
        case 6:
            return cv::Size_<float>(420, 298) / kPointsPerCentymeter;
        case 7:
            return cv::Size_<float>(298, 210) / kPointsPerCentymeter;
        default:
            throw std::invalid_argument("Unsupported AX size.");
    }
    return resolution;
}

void save_board_tiff_real_scale(const cv::Mat1b &image, const float cm_per_pixel, const int dpi,
                                const std::string program_name, const std::string board_name)
{
    const float tiff_dpi = dpi * kMagicDpiScale;
    const int tiff_dpi_i32 = std::ceil(tiff_dpi);
    const std::filesystem::path dir_path = io::save_path() / program_name;
    std::filesystem::create_directories(dir_path);

    const std::filesystem::path full_filepath = dir_path / std::format("{}.tiff", board_name);
    spdlog::info("cm_per_pixel={}, tiff_dpi={}, saving to {}", cm_per_pixel, tiff_dpi, full_filepath.string());

    cv::imwrite(full_filepath, image,
                {cv::ImwriteFlags::IMWRITE_TIFF_XDPI, tiff_dpi_i32, cv::ImwriteFlags::IMWRITE_TIFF_YDPI, tiff_dpi_i32});
}

cv::Mat1b draw_canonical_board(const std::unique_ptr<Board> &calibration_board, const float cm_per_pixel,
                               const int resolution)
{
    const BoardRectGrid *const board_grid = dynamic_cast<const BoardRectGrid *const>(calibration_board.get());
    if (board_grid)
    {
        return board::draw_canonical_board(*board_grid, cm_per_pixel);
    }

    BoardHexGrid *const board_hex = dynamic_cast<BoardHexGrid *const>(calibration_board.get());
    if (board_hex)
    {
        const cv::Size_<float> board_dimention_cm = get_board_dimention_cm(resolution);
        const float board_scale = board_dimention_cm.width / get_board_dimention_cm(3).width;
        board_hex->apply_scale(board_scale);
        return board::draw_canonical_board(*board_hex, cm_per_pixel, board_dimention_cm);
    }

    throw std::invalid_argument("Wrong board type");
}

}  // namespace

void DrawBoard::execute()
{
    if (board_type_ > 1)
    {
        throw std::invalid_argument("Invalid board type.");
    }

    if (board_type_ == 0 && !board_params_.empty())
    {
        throw std::invalid_argument("RectRingCalibTarget board does not take parameters.");
    }
    else if (board_type_ == 0 && resolution_ != 3)
    {
        throw std::invalid_argument("RectRingCalibTarget board supports only A3 size.");
    }

    const float cm_per_pixel = kInchesPerCentymeter / dpi_;

    const std::unique_ptr<Board> calibration_board = board::get_board(board_type_, board_params_);

    const cv::Mat1b image = draw_canonical_board(calibration_board, cm_per_pixel, resolution_);

    const std::string filename = std::format("{}_A{}", calibration_board->name(), board_type_ == 1 ? resolution_ : 3);
    save_board_tiff_real_scale(image, cm_per_pixel, dpi_, name(), filename);
}
