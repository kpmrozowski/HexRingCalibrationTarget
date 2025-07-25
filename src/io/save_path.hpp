#pragma once

/* @brief Functions for getting paths to places where datasets and debugging data are saved.
 */

#include <filesystem>

namespace io
{
/* @brief Returns a path where datasets are saved and additionally creates this location if it doesn't exist yet.
 */
std::filesystem::path save_path();

/* @brief Returns a path where debug data is saved and additionally creates this location if it doesn't exist yet.
 * Can be a subdirectory of `save_path()`.
 */
std::filesystem::path debug_save_path();

}  // namespace io
