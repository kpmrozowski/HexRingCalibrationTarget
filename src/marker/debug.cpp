#include "debug.hpp"

#include <filesystem>
#include <format>

#include <opencv2/imgproc.hpp>

#include <Eigen/Dense>
#include <nlohmann/json.hpp>

#include "board_drawing.hpp"
#include "io/debug.hpp"
#include "io/save_path.hpp"

namespace io::debug
{
}  // namespace io::debug

namespace
{
void draw_dot(cv::Mat3b &image, const float col_center, const float row_center, const float radius_in_pixel,
              const int channel, const float in_value = 255.f)
{
    const int start_row = std::clamp(int(row_center) - int(radius_in_pixel) - 2, 0, image.rows);
    const int end_row = std::clamp(int(row_center) + 1 + int(radius_in_pixel) + 3, 0, image.rows);

    const int start_col = std::clamp(int(col_center) - int(radius_in_pixel) - 2, 0, image.cols);
    const int end_col = std::clamp(int(col_center) + 1 + int(radius_in_pixel) + 3, 0, image.cols);

    for (int row = start_row; row < end_row; ++row)
    {
        for (int col = start_col; col < end_col; ++col)
        {
            // NOTE THAT WE DO NOT ALLOW ALIASING OF CIRCLES!!!
            const auto [val, inside] = board::accurate_intensity(row, col, row_center, col_center, radius_in_pixel,
                                                                 in_value, image(row, col)[0]);
            if (inside)
            {
                image(row, col)[channel] = val;
            }
        }
    }
}

}  // namespace

namespace marker::debug
{
void save_inner_markers_and_unique(const cv::Mat1b &image, const std::vector<base::MarkerUnidentified> &markers_core,
                                   const std::vector<base::MarkerUnidentified> &ring_and_coding, const int image_idx,
                                   const size_t scale_idx)
{
    cv::Mat3b painted;
    cv::cvtColor(image, painted, cv::COLOR_GRAY2BGR);

    int inner_marker_id = 0;
    int ring_marker_id = 0;
    int unique_marker_id = 0;

    int unrecognized = 0;

    for (const auto &marker : markers_core)
    {
        if (marker.type_ == base::Type::INNER_CORE)
        {
            cv::putText(painted, std::to_string(inner_marker_id), cv::Point(marker.col_, marker.row_),
                        cv::FONT_HERSHEY_PLAIN, 1.0, cv::Scalar(255, 0, 0));
            ++inner_marker_id;
        }
        else
        {
            cv::putText(painted, std::to_string(unrecognized), cv::Point(marker.col_, marker.row_),
                        cv::FONT_HERSHEY_PLAIN, 1.0, cv::Scalar(100, 0, 100));
            ++unrecognized;
        }
    }

    for (const auto &marker : ring_and_coding)
    {
        if (marker.type_ == base::Type::RING)
        {
            cv::putText(painted, std::to_string(ring_marker_id),
                        cv::Point(marker.col_ - marker.width_ / 2, marker.row_ - marker.height_ / 2),
                        cv::FONT_HERSHEY_PLAIN, 1.0, cv::Scalar(0, 0, 255));
            draw_dot(painted, marker.col_ - marker.width_ / 2, marker.row_ - marker.height_ / 2, 1.f, 2);
            ++ring_marker_id;
        }
        else if (marker.type_ == base::Type::CODING)
        {
            cv::putText(painted, std::to_string(unique_marker_id), cv::Point(marker.col_, marker.row_),
                        cv::FONT_HERSHEY_PLAIN, 1.0, cv::Scalar(0, 255, 0));
            draw_dot(painted, marker.col_, marker.row_, 1.f, 1);
            ++unique_marker_id;
        }
    }

    io::debug::save_image(painted, std::format("soup_{}_{}", image_idx, scale_idx), kMarkersSubdir);
}

void save_inner_markers_and_unique(const cv::Mat1b &image, const std::vector<base::MarkerCoding> &coding,
                                   const std::vector<base::MarkerRing> &ring, const int image_idx,
                                   const size_t scale_idx)
{
    cv::Mat3b painted;
    cv::cvtColor(image, painted, cv::COLOR_GRAY2BGR);

    for (int idx = 0; idx < coding.size(); idx++)
    {
        cv::putText(painted, std::to_string(idx), cv::Point(coding[idx].col_, coding[idx].row_), cv::FONT_HERSHEY_PLAIN,
                    1.0, cv::Scalar(0, 255, 0));
        draw_dot(painted, coding[idx].col_, coding[idx].row_, 1.f, 1);
    }

    for (int ring_idx = 0; ring_idx < ring.size(); ring_idx++)
    {
        cv::putText(painted, std::to_string(ring_idx + coding.size()),
                    cv::Point(ring[ring_idx].col_, ring[ring_idx].row_), cv::FONT_HERSHEY_PLAIN, 1.0,
                    cv::Scalar(0, 0, 255));
        draw_dot(painted, ring[ring_idx].col_, ring[ring_idx].row_, 1.f, 2);
    }

    io::debug::save_image(painted, std::format("final_{}_{}", image_idx, scale_idx), kMarkersSubdir);
}

void save_marker_identification(const cv::Mat1b &image, const Eigen::Matrix<std::optional<int>, -1, -1> &ordering,
                                const std::vector<base::MarkerRing> &markers, const int image_idx)
{
    cv::Mat3b painted;
    cv::cvtColor(image, painted, cv::COLOR_GRAY2BGR);

    for (int row = 0; row < ordering.rows(); ++row)
    {
        for (int col = 0; col < ordering.cols(); ++col)
        {
            if (!ordering(row, col).has_value())
            {
                continue;
            }
            const int ring_idx = ordering(row, col).value();
            const int gloabal_id = col + row * (int)ordering.cols();
            cv::putText(painted, std::to_string(gloabal_id), cv::Point(markers[ring_idx].col_, markers[ring_idx].row_),
                        cv::FONT_HERSHEY_PLAIN, 1.0, cv::Scalar(0, 255, 0));
            draw_dot(painted, markers[ring_idx].col_, markers[ring_idx].row_, 1.f, 1);
        }
    }

    io::debug::save_image(painted, std::format("identified_{}", image_idx), kMarkersSubdir);
}

void save_neighbors_edges(const cv::Mat1b &image, const std::vector<base::MarkerNeighborhood> &neighbors,
                          const std::vector<base::MarkerCoding> &coding, const std::vector<base::MarkerRing> &ring,
                          const int image_idx)
{
    if (neighbors.empty())
    {
        return;
    }

    cv::Mat3b painted;
    cv::cvtColor(image, painted, cv::COLOR_GRAY2BGR);

    const int coding_size = coding.size();
    for (int id_1st = 0; id_1st < neighbors.size(); id_1st++)
    {
        const int ring_id_1st = id_1st - coding_size;
        const float col_1st = ring_id_1st < 0 ? coding[id_1st].col_ : ring[ring_id_1st].col_;
        const float row_1st = ring_id_1st < 0 ? coding[id_1st].row_ : ring[ring_id_1st].row_;
        for (int idx_2nd = 0; idx_2nd < neighbors[id_1st].neighbors.size(); idx_2nd++)
        {
            const int id_2nd = neighbors[id_1st].neighbors[idx_2nd];
            const int ring_id_2nd = id_2nd - coding_size;
            const float col_2nd = ring_id_2nd < 0 ? coding[id_2nd].col_ : ring[ring_id_2nd].col_;
            const float row_2nd = ring_id_2nd < 0 ? coding[id_2nd].row_ : ring[ring_id_2nd].row_;

            cv::line(painted, cv::Point(col_1st, row_1st), cv::Point(col_2nd, row_2nd), cv::Scalar(0, 255, 0));

            cv::putText(painted, std::to_string(idx_2nd),
                        cv::Point(col_1st + 0.33f * (col_2nd - col_1st), row_1st + +0.33f * (row_2nd - row_1st)),
                        cv::FONT_HERSHEY_PLAIN, 1.0, cv::Scalar(255, 0, 255));
        }
    }

    for (int coding_idx = 0; coding_idx < coding.size(); coding_idx++)
    {
        cv::putText(painted, std::to_string(coding_idx), cv::Point(coding[coding_idx].col_, coding[coding_idx].row_),
                    cv::FONT_HERSHEY_PLAIN, 1.0, cv::Scalar(0, 255, 0));
        draw_dot(painted, coding[coding_idx].col_, coding[coding_idx].row_, 1.f, 1);
    }
    for (int ring_idx = 0; ring_idx < ring.size(); ring_idx++)
    {
        cv::putText(painted, std::to_string(ring_idx + coding.size()),
                    cv::Point(ring[ring_idx].col_, ring[ring_idx].row_), cv::FONT_HERSHEY_PLAIN, 1.0,
                    cv::Scalar(0, 0, 255));
        draw_dot(painted, ring[ring_idx].col_, ring[ring_idx].row_, 1.f, 2);
    }

    io::debug::save_image(painted, std::format("neighbors_{}", image_idx), kMarkersSubdir);
}
}  // namespace marker::debug
