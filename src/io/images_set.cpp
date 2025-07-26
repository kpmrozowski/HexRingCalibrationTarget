#include "images_set.hpp"

#include <filesystem>
#include <fstream>
#include <regex>
#include <vector>

#include <spdlog/spdlog.h>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

const std::regex ImageFilesDataset::kRegexDataset = std::regex("^(\\w+)_(\\d+)$", std::regex_constants::ECMAScript);

bool ImageFilesDataset::read_images_filenames(std::vector<ImageFileDescriptor>& result, const std::regex& regex) const
{
    for (const std::filesystem::path& entry : std::filesystem::directory_iterator(path_))
    {
        if (ImageFileDescriptor::is_valid(entry))
        {
            const auto match = ImageFileDescriptor::match_filepath(entry, regex);
            if (match.first.size() == 3 && camera_key_ == match.first[1])
            {
                result.emplace_back(entry, std::stoi(match.first[2]), view_current_type());
            }
            else if (match.first.size() == 3 && camera_key_ == match.first[2])
            {
                result.emplace_back(entry, std::stoi(match.first[1]), view_current_type());
            }
        }
    }

    std::sort(result.begin(), result.end(),
              [](const ImageFileDescriptor& lhs, const ImageFileDescriptor& rhs) { return lhs.id() < rhs.id(); });

    return !result.empty();
}

std::vector<ImageFileDescriptor> ImageFilesDataset::operator()() const
{
    //// TODO: It should be checked on cli input, not here
    if (!std::filesystem::is_directory(path_))
    {
        spdlog::error("{} is not a directory", path_);
        throw std::runtime_error("not a directory");
    }

    std::vector<ImageFileDescriptor> result;
    if (read_images_filenames(result, kRegexDataset))
    {
        spdlog::info("Found dataset in correct format");
    }
    else
    {
        spdlog::error("Dataset in correct format (<Camera_ID>_<IDX>) was not found at at dir {} ", path_);
        throw std::runtime_error("Not enoughs images");
    }
    if (result.size() < 3)
    {
        spdlog::error("At dir {} we found {} images, at least 3 is needed. Searched type was {} ", path_, result.size(),
                      camera_key_);
        throw std::runtime_error("Not enoughs images");
    }
    return result;
}

std::tuple<cv::Mat, int> ImageFileDescriptor::read_image() const
{
    std::tuple<cv::Mat, int> return_image;
    cv::Mat img = cv::imread(path(), cv::IMREAD_UNCHANGED);
    std::get<0>(return_image) = img;
    std::get<1>(return_image) = id();

    return return_image;
}
