#include "debug.hpp"

#include <happly.h>
#include <opencv2/imgcodecs.hpp>

namespace
{
constexpr int kDebugJsonIndent = 4;

static constexpr std::string_view kRed("red");
static constexpr std::string_view kGreen("green");
static constexpr std::string_view kBlue("blue");

const std::unordered_map<int, std::string_view> kMatTypeToExtension{
    {CV_8UC1, ".png"},   {CV_8UC3, ".png"},   {CV_8UC4, ".png"},   {CV_16UC1, ".tiff"},
    {CV_16UC3, ".tiff"}, {CV_16SC1, ".tiff"}, {CV_32SC1, ".tiff"}, {CV_32SC3, ".tiff"},
    {CV_32FC1, ".tiff"}, {CV_32FC3, ".tiff"}, {CV_64FC1, ".tiff"}, {CV_64FC3, ".tiff"},
};

}  // namespace

namespace io
{
void debug::save_image(const cv::Mat& image, const std::filesystem::path& name, const std::filesystem::path& subdir)
{
    const std::filesystem::path dir_path = debug_save_path() / subdir;
    std::filesystem::create_directories(dir_path);

    std::filesystem::path full_filepath = (dir_path / name);
    full_filepath.replace_extension(kMatTypeToExtension.at(image.type()));
    spdlog::info("Saving image to {}", full_filepath.string());

    cv::imwrite(full_filepath.string(), image);
}

void debug::save_image_to(const cv::Mat& image, const std::filesystem::path& name, const std::filesystem::path& output_path)
{
    std::filesystem::create_directories(output_path);

    std::filesystem::path full_filepath = (output_path / name);
    full_filepath.replace_extension(kMatTypeToExtension.at(image.type()));

    cv::imwrite(full_filepath.string(), image);
}

void debug::save_json(const nlohmann::json& json, std::filesystem::path name, const std::filesystem::path& subdir)
{
    const std::filesystem::path dir_path = debug_save_path() / subdir;
    std::filesystem::create_directories(dir_path);
    name.replace_extension(".json");
    spdlog::info("Saving json to '{}'", (dir_path / name).string());
    std::ofstream file(dir_path / name);
    file << json.dump(kDebugJsonIndent);
}

namespace debug
{
void HapplyWrapper::add_point(const Eigen::Vector3d& pts, const Eigen::Matrix<unsigned char, 3, 1>& color)
{
    if (!has_colors_ && x_.empty())
    {
        has_colors_ = !color.isZero();
    }

    x_.emplace_back(pts(0));
    y_.emplace_back(pts(1));
    z_.emplace_back(pts(2));

    if (has_colors_)
    {
        red_.emplace_back(color(0));
        green_.emplace_back(color(1));
        blue_.emplace_back(color(2));
    }
}

void HapplyWrapper::add_edge(const Eigen::Vector3d& vertex1, const Eigen::Vector3d& vertex2)
{
    edges_.push_back({vertex1, vertex2});
}

void HapplyWrapper::add_position(const Eigen::Quaterniond& rotation, const Eigen::Vector3d& translation)
{
    constexpr int kSteps = 50;
    constexpr float kLength = 0.1;
    constexpr float kDistInc = kLength / kSteps;

    for (float pt_dist = 0.f; pt_dist < kLength; pt_dist += kDistInc)
    {
        const Eigen::Vector3d pts_0 = translation + rotation * Eigen::Vector3d::UnitX() * pt_dist;
        add_point(pts_0, {255, 0, 0});
        const Eigen::Vector3d pts_1 = translation + rotation * Eigen::Vector3d::UnitY() * pt_dist;
        add_point(pts_1, {0, 255, 0});
        const Eigen::Vector3d pts_2 = translation + rotation * Eigen::Vector3d::UnitZ() * pt_dist;
        add_point(pts_2, {0, 0, 255});
    }
}

void HapplyWrapper::add_trajectory(const std::vector<Eigen::Isometry3d>& trajectory, const size_t draw_position_every_n)
{
    if (trajectory.empty())
    {
        return;
    }

    add_position(Eigen::Quaterniond(trajectory.front().linear()), trajectory.front().translation());

    Eigen::Vector3d last = trajectory.front().translation();
    for (size_t idx = 1; idx < trajectory.size(); ++idx)
    {
        const Eigen::Quaterniond rotation(trajectory[idx].linear());
        const Eigen::Vector3d translation(trajectory[idx].translation());
        add_edge(last, translation);
        last = translation;

        if (idx % draw_position_every_n == 0)
        {
            add_position(rotation, translation);
        }
    }
}

void HapplyWrapper::create_edge_vertices()
{
    for (size_t idx = 0; idx < edges_.size(); ++idx)
    {
        const auto& [vertex1, vertex2] = edges_[idx];
        add_point(vertex1, {255, 255, 255});
        add_point(vertex2, {255, 255, 255});
    }
}

void HapplyWrapper::add_vertices(happly::PLYData& ply_out)
{
    ply_out.addElement(std::string(kVertexName), x_.size());

    ply_out.getElement(std::string(kVertexName)).addProperty<double>(std::string(kVertexX), x_);
    ply_out.getElement(std::string(kVertexName)).addProperty<double>(std::string(kVertexY), y_);
    ply_out.getElement(std::string(kVertexName)).addProperty<double>(std::string(kVertexZ), z_);

    if (!red_.empty() || !green_.empty() || !blue_.empty())
    {
        ply_out.getElement(std::string(kVertexName)).addProperty<unsigned char>(std::string(kRed), red_);
        ply_out.getElement(std::string(kVertexName)).addProperty<unsigned char>(std::string(kGreen), green_);
        ply_out.getElement(std::string(kVertexName)).addProperty<unsigned char>(std::string(kBlue), blue_);
    }
}

void HapplyWrapper::add_edges(happly::PLYData& ply_out)
{
    const int edge_vertices_offset = (int)(x_.size() - 2 * edges_.size());
    std::vector<int> vertices_1(edges_.size());
    std::vector<int> vertices_2(edges_.size());
    for (int idx = 0; idx < (int)edges_.size(); ++idx)
    {
        vertices_1[idx] = edge_vertices_offset + 2 * idx + 0;
        vertices_2[idx] = edge_vertices_offset + 2 * idx + 1;
    }
    ply_out.addElement(std::string(kEdgeName), edges_.size());
    ply_out.getElement(std::string(kEdgeName)).addProperty<int>(std::string(kEdgeVertexIdx1), vertices_1);
    ply_out.getElement(std::string(kEdgeName)).addProperty<int>(std::string(kEdgeVertexIdx2), vertices_2);
}

void HapplyWrapper::save(std::filesystem::path name, const std::filesystem::path& subdir,
                         const std::filesystem::path& root_dir)
{
    happly::PLYData ply_out;

    create_edge_vertices();
    add_vertices(ply_out);
    add_edges(ply_out);

    const std::filesystem::path dir_path = root_dir / subdir;
    std::filesystem::create_directories(dir_path);
    name.replace_extension(".ply");

    spdlog::debug("Saving cloud: {}", (dir_path / name).string());
    ply_out.write(dir_path / name, happly::DataFormat::ASCII);
}

}  // namespace debug

}  // namespace io
