#include "debug.hpp"

#include "initial_calibration/precalibration.hpp"
#include "io/debug.hpp"

namespace
{
constexpr std::string_view kCameraMatrix = "camera_matrix";
constexpr std::string_view kDistortions = "distortions";
constexpr std::string_view kSensorRows = "sensor_rows";
constexpr std::string_view kSensorCols = "sensor_cols";
constexpr std::string_view kDeviceType = "device_type";
constexpr std::string_view kReprojRMSE = "reprojection_RMSE";

template <typename Floating>
struct PinholeCalibData
{
    std::vector<Floating> camera_data_;
    std::vector<Floating> distortions_;
    int image_rows_, image_cols_;
    std::string device_type_name_;
    double reproj_rmse_;
};

template <typename Output, typename Input>
Output vector_to_vector(const Input& input)
{
    Output out(input.size());

    for (size_t idx = 0; idx < static_cast<size_t>(out.size()); ++idx)
    {
        out[idx] = input[idx];
    }
    return out;
}

template <typename Floating>
PinholeCalibData<Floating> to_json_data(const precalibration::PinholeCalibration& calib)
{
    PinholeCalibData<Floating> data;

    data.camera_data_ = {calib.camera_matrix_(0, 0), calib.camera_matrix_(0, 1), calib.camera_matrix_(0, 2),
                         calib.camera_matrix_(1, 0), calib.camera_matrix_(1, 1), calib.camera_matrix_(1, 2),
                         calib.camera_matrix_(2, 0), calib.camera_matrix_(2, 1), calib.camera_matrix_(2, 2)};

    data.distortions_ = vector_to_vector<std::vector<Floating>>(calib.distortions_);

    data.image_cols_ = calib.image_cols_;
    data.image_rows_ = calib.image_rows_;

    data.device_type_name_ = "Brown-Conrady";
    data.reproj_rmse_ = calib.reproj_rmse_;

    return data;
}
}  // namespace

void precalibration::debug::save_calibration(const PinholeCalibration& calib, const std::string& camera_name,
                                             const std::filesystem::path& subdir)
{
    nlohmann::json calibration_nodes;

    const auto calib_data = to_json_data<double>(calib);
    calibration_nodes[camera_name] = {
        {std::string(kCameraMatrix), calib_data.camera_data_},    {std::string(kDistortions), calib_data.distortions_},
        {std::string(kSensorRows), calib_data.image_rows_},       {std::string(kSensorCols), calib_data.image_cols_},
        {std::string(kDeviceType), calib_data.device_type_name_}, {std::string(kReprojRMSE), calib_data.reproj_rmse_}};

    io::debug::save_json(calibration_nodes, "calibration_" + camera_name, subdir);
}
