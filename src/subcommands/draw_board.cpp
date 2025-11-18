#include "draw_board.hpp"

#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgcodecs.hpp>

#include <spdlog/spdlog.h>
#include <cmd_launcher/subcommand.hpp>
#include <io/save_path.hpp>
#include "board_drawing.hpp"

namespace
{

constexpr int kDefaultPaperDimension = 3;  // A3
constexpr float kInchesPerMeter = 39.36f;
constexpr float kInchesPerMillimeter = kInchesPerMeter / 1000.f;
constexpr float kPointsPerInch = 72.00f;
constexpr float kPointsPerMillimeter = kPointsPerInch * kInchesPerMillimeter;

/** @brief Point is a smallest unit of measurement in typography
 * @ref https://www.a4-size.com/a4-size-in-point
 */
cv::Size_<float> get_board_dimension_mm(const int a_x)
{
    cv::Size resolution;
    switch (a_x)
    {
        case -2:
            return cv::Size_<float>(6742, 4768) / kPointsPerMillimeter;
        case -1:
            return cv::Size_<float>(4768, 3371) / kPointsPerMillimeter;
        case 0:
            return cv::Size_<float>(3371, 2384) / kPointsPerMillimeter;
        case 1:
            return cv::Size_<float>(2384, 1684) / kPointsPerMillimeter;
        case 2:
            return cv::Size_<float>(1684, 1191) / kPointsPerMillimeter;
        case 3:
            return cv::Size_<float>(1191, 842) / kPointsPerMillimeter;
        case 4:
            return cv::Size_<float>(842, 595) / kPointsPerMillimeter;
        case 5:
            return cv::Size_<float>(595, 420) / kPointsPerMillimeter;
        case 6:
            return cv::Size_<float>(420, 298) / kPointsPerMillimeter;
        case 7:
            return cv::Size_<float>(298, 210) / kPointsPerMillimeter;
        default:
            throw std::invalid_argument("Unsupported AX size.");
    }
    return resolution;
}

void save_board_tiff_real_scale(const cv::Mat1b &image, const float mm_per_pixel, const int tiff_dpi,
                                const std::string program_name, const std::string board_name)
{
    const int tiff_dpi_i32 = std::ceil(tiff_dpi);
    const std::filesystem::path dir_path = io::save_path() / program_name;
    std::filesystem::create_directories(dir_path);

    const std::filesystem::path full_filepath = dir_path / std::format("{}.tiff", board_name);
    spdlog::info("mm_per_pixel={}, tiff_dpi={}, saving to {}", mm_per_pixel, tiff_dpi, full_filepath.string());

    cv::imwrite(full_filepath, image,
                {cv::ImwriteFlags::IMWRITE_TIFF_XDPI, tiff_dpi_i32, cv::ImwriteFlags::IMWRITE_TIFF_YDPI, tiff_dpi_i32});
}

void test_detection(const cv::Mat1b &image, const BoardCircleGrid &board)
{
    const cv::Size pattern_size(board.cols_, board.rows_);
    std::vector<cv::Point2f> centers(board.cols_ * board.rows_);

    spdlog::info("findCirclesGrid");
    const auto simetricity = board.is_asymetric_ ? cv::CALIB_CB_ASYMMETRIC_GRID : cv::CALIB_CB_SYMMETRIC_GRID;
    if (const bool success = cv::findCirclesGrid(image, pattern_size, centers, simetricity); success)
    {
        for (size_t idx = 0; idx < centers.size(); ++idx)
        {
            spdlog::info("pt[{:02}]: ({:7.3f}, {:7.3f})", idx, centers[idx].x, centers[idx].y);
        }
    }
    else
    {
        spdlog::warn("BoardCircleGrid detection failure! Lower the DPI to max 50.");
    }
}

cv::Mat1b draw_canonical_board(const std::unique_ptr<Board> &calibration_board, const float mm_per_pixel,
                               const int resolution)
{
    const BoardRectGrid *const board_grid = dynamic_cast<const BoardRectGrid *const>(calibration_board.get());
    if (board_grid)
    {
        return board::draw_canonical_board(*board_grid, mm_per_pixel);
    }

    BoardHexGrid *const board_hex = dynamic_cast<BoardHexGrid *const>(calibration_board.get());
    if (board_hex)
    {
        const cv::Size_<float> default_board_dimension_mm = get_board_dimension_mm(kDefaultPaperDimension);
        const cv::Size_<float> board_dimension_mm = get_board_dimension_mm(resolution);
        const float board_scale = board_dimension_mm.width / default_board_dimension_mm.width;
        board_hex->apply_scale(board_scale);
        return board::draw_canonical_board(*board_hex, mm_per_pixel, board_dimension_mm);
    }

    BoardCircleGrid *const board_circle = dynamic_cast<BoardCircleGrid *const>(calibration_board.get());
    if (board_circle)
    {
        const cv::Size_<float> board_dimension_mm = get_board_dimension_mm(resolution);
        board_circle->apply_scale(1);
        cv::Mat1b image = board::draw_canonical_board(*board_circle, mm_per_pixel, board_dimension_mm);
        test_detection(image, *board_circle);
        return image;
    }

    throw std::invalid_argument("Wrong board type");
}

}  // namespace

void DrawBoard::execute()
{
    if (board_type_ < 0 || board_type_ > 2)
    {
        throw std::invalid_argument("Invalid board type.");
    }
    else if (board_type_ == 0 && !board_params_.empty())
    {
        throw std::invalid_argument("RectRingCalibTarget board does not take parameters.");
    }
    else if (board_type_ == 0 && resolution_ != 3)
    {
        throw std::invalid_argument("RectRingCalibTarget board supports only A3 size.");
    }

    const float mm_per_pixel = 1.f / (dpi_ * kInchesPerMillimeter);

    std::unique_ptr<Board> calibration_board;
    if (!board_params_.empty())
    {
        calibration_board = board::get_board(board_type_, board_params_);
    }
    else if (!board_params_path_.empty())
    {
        calibration_board = board::get_board(board_params_path_);
    }
    else
    {
        throw std::invalid_argument("Use --board-params or --board-params-path");
    }

    const cv::Mat1b image = draw_canonical_board(calibration_board, mm_per_pixel, resolution_);

    const std::string filename = std::format("{}_A{}", calibration_board->name(), board_type_ == 1 ? resolution_ : 3);
    save_board_tiff_real_scale(image, mm_per_pixel, dpi_, name(), filename);
}
