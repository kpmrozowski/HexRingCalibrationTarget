#pragma once

#include <filesystem>

namespace precalibration
{
struct PinholeCalibration;

namespace debug
{
void save_calibration(const PinholeCalibration& calib, const std::string& camera_name,
                      const std::filesystem::path& subdir = std::filesystem::path());
}

}  // namespace precalibration
