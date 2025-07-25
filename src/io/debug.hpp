#pragma once

/* @brief This file provides functions meant for debugging only. All files saved using these are saved in
 * `save_path()/debug`.
 */

#include <Eigen/Dense>
#include <nlohmann/json.hpp>

#include <spdlog/spdlog.h>
#include <opencv2/core/mat.hpp>

#include "save_path.hpp"

namespace happly
{
class PLYData;
}

namespace io
{
static constexpr std::string_view kVertexName("vertex");
static constexpr std::string_view kVertexX("x");
static constexpr std::string_view kVertexY("y");
static constexpr std::string_view kVertexZ("z");
static constexpr std::string_view kEdgeName("edge");
static constexpr std::string_view kEdgeVertexIdx1("vertex1");
static constexpr std::string_view kEdgeVertexIdx2("vertex2");

}  // namespace io

namespace io::debug
{
/* @brief Saves an image of `cv::Mat` types fulfilling `SavableMat` concept.
 * @param mat cv::Mat to save.
 * @param name Name of the file, without extension (it is decided automatically by our code).
 * @param subdir Optional subdirectory for the created file.
 */
void save_image(const cv::Mat& image, const std::filesystem::path& name,
                const std::filesystem::path& subdir = std::filesystem::path());

/* @brief Saves a json to a file.
 * @param mat cv::Mat to save.
 * @param name Name of the file, without extension.
 * @param subdir Optional subdirectory for the created file.
 */
void save_json(const nlohmann::json& json, std::filesystem::path name,
               const std::filesystem::path& subdir = std::filesystem::path());

/**
 * @brief Utility wrapper to add point that later be saved as PLY file to view in other SW.
 */
class HapplyWrapper
{
   private:
    std::vector<double> x_, y_, z_;
    std::vector<unsigned char> red_, green_, blue_;
    std::vector<std::array<Eigen::Vector3d, 2>> edges_;

    bool has_colors_ = false;

   public:
    /**
     * @brief Add point to internal structure, later will be saved.
     *
     * @param pts point that should be saved
     * @param color point color in RGB
     */
    void add_point(const Eigen::Vector3d& pts,
                   const Eigen::Matrix<unsigned char, 3, 1>& color = Eigen::Matrix<unsigned char, 3, 1>::Zero());

    /**
     * @brief Add line to internal structure, later will be saved.
     *
     * @param vertex1 start point of the edge
     * @param vertex2 end point of the edge
     */
    void add_edge(const Eigen::Vector3d& vertex1, const Eigen::Vector3d& vertex2);

    /**
     * @brief Add coordinate axis at position indicated by given transform. Usually used to visualize camera
     * localization.
     */
    void add_position(const Eigen::Quaterniond& rotation, const Eigen::Vector3d& translation);

    /**
     * @brief Add multiple coordinate axis at positions indicated by given transforms. Usually used to visualize camera
     * localizations.
     */
    void add_trajectory(const std::vector<Eigen::Isometry3d>& trajectory, const size_t draw_position_every_n = 1);

    /**
     * @brief Add coordinate axis at position indicated by given transform. Usually used to visualize camera
     * localizations.
     *
     * @param name Name of the file, without extension (.ply is added by our code).
     * @param subdir Optional subdirectory for the created file.
     * @param root_dir Optional directory for the created subdirectory.
     */
    void save(std::filesystem::path name, const std::filesystem::path& subdir = "",
              const std::filesystem::path& root_dir = debug_save_path());

   private:
    void create_edge_vertices();
    void add_vertices(happly::PLYData& ply_out);
    void add_edges(happly::PLYData& ply_out);
};
}  // namespace io::debug
