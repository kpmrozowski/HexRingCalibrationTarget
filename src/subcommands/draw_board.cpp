#include "draw_board.hpp"

#include <opencv2/imgcodecs.hpp>

#include <spdlog/spdlog.h>
#include <cmd_launcher/subcommand.hpp>
#include <io/save_path.hpp>
#include "board_drawing.hpp"

static constexpr float kInchesPerMeter = 39.36;
static constexpr float kInchesPerCentymeter = kInchesPerMeter / 100.;
static constexpr float kPointsPerInch = 72.00;
static constexpr float kPointsPerCentymeter = kPointsPerInch * kInchesPerCentymeter;
static constexpr float kMagicDpiScale = 2710 / 420.;

namespace
{

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

void save_board_tiff_real_scale(const cv::Mat1b& image, const float cm_per_pixel, const int dpi,
                                const std::string program_name)
{
    const float tiff_dpi = dpi * kMagicDpiScale;
    const int tiff_dpi_i32 = std::ceil(tiff_dpi);
    const std::filesystem::path dir_path = io::save_path() / program_name;
    std::filesystem::create_directories(dir_path);

    spdlog::info("cm_per_pixel={}, tiff_dpi={}, saving to {}", cm_per_pixel, tiff_dpi,
                 (dir_path / "hex_board.tiff").string());

    cv::imwrite(dir_path / "hex_board.tiff", image,
                {cv::ImwriteFlags::IMWRITE_TIFF_XDPI, tiff_dpi_i32, cv::ImwriteFlags::IMWRITE_TIFF_YDPI, tiff_dpi_i32});
}

}  // namespace

void DrawBoard::execute()
{
    const float cm_per_pixel = kInchesPerCentymeter / dpi_;

    cv::Mat1b image;
    if (board_type_ == 0)
    {
        image = board::draw_canonical_board(BoardRectGrid(), cm_per_pixel);
    }
    else if (board_type_ == 1)
    {
        const cv::Size_<float> board_dimention_cm = get_board_dimention_cm(resolution_);
        const float board_scale = board_dimention_cm.width / get_board_dimention_cm(3).width;
        BoardHexGrid board;
        board.apply_scale(board_scale);
        image = board::draw_canonical_board(board, cm_per_pixel, board_dimention_cm);
    }
    else
    {
        throw std::invalid_argument("Wrong board type");
    }

    save_board_tiff_real_scale(image, cm_per_pixel, dpi_, name());
}
