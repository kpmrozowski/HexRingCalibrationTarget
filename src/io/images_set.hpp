#pragma once

#include <filesystem>
#include <regex>
#include <string>

#include <spdlog/spdlog.h>
#include <opencv2/core/mat.hpp>

#include <map>
#include <mutex>
#include <string>

/*
 * @brief class stores absolute path to images that where red by system during IO stage. As number of images can be big,
 * we created lazy reading where we can ask to IO from disc image for given device so at no point we have all loaded
 * images to memory.
 */
class ImageDatabase
{
   public:
    static void push_absolute_mapping(const std::string& camera_type, const int image_id,
                                      const std::string& absolute_path);

    static std::string get_absolute_path(const std::string& camera_type, const int image_id);

   private:
    static std::mutex mtx_global;

    // Mapping [ Camera / [image id / absolute path]]
    static std::map<std::string, std::map<int, std::string>> absolute_mapping_;
};

// This should be client code
class ImageFileDescriptor
{
    static constexpr std::array<std::string_view, 5> kPossibleExtensions = {".bmp", ".png", ".tiff", ".jpg", ".raw"};

   public:
    ImageFileDescriptor(const std::filesystem::path& entry, int id, std::string type)
        : path_(entry), id_(id), type_(type) {};

    std::tuple<cv::Mat, int> read_image() const;

    static bool is_valid(const std::filesystem::path& entry)
    {
        return std::filesystem::is_regular_file(entry) && is_valid_image(entry);  // std::make_pair(is_valid, match);
    }

    /**
     * @brief Provide regex matching.
     *
     * @param entry It's stem will be checked
     *
     * @returns SMatch of image regex AND ITS STRING. String CAN NOT BE DISCARDED because it invalidate iterators in
     * smatch
     */
    [[nodiscard]] static std::pair<std::smatch, std::string> match_filepath(const std::filesystem::path& entry,
                                                                            const std::regex& regex)
    {
        std::pair<std::smatch, std::string> results;
        results.second = entry.stem().string();
        std::regex_match(results.second, results.first, regex);
        return results;
    }

    std::string path() const { return path_.string(); }

    int id() const { return id_; }

    std::string type() const { return type_; }

   private:
    static bool is_valid_image(const std::filesystem::path& entry)
    {
        return std::any_of(kPossibleExtensions.cbegin(), kPossibleExtensions.cend(), [&entry](const auto& extension)
                           { return entry.extension().string().compare(extension) == 0; });
    }
    std::filesystem::path path_;
    int id_;
    std::string type_;
};

// This should be client code
class ImageFilesDataset
{
    // regex for images with format <camera_key>_<image_number>
    static const std::regex kRegexDataset;

   public:
    ImageFilesDataset(const std::string& path, const std::string_view& camera_id, const int start_idx = 0)
        : path_(path), camera_key_(std::string(camera_id)), start_idx_(start_idx) {};
    std::vector<ImageFileDescriptor> operator()() const;

    std::string view_current_type() const { return camera_key_; }

   private:
    const std::string path_;
    const std::string camera_key_;
    const int start_idx_;

    bool read_images_filenames(std::vector<ImageFileDescriptor>& result, const std::regex& regex) const;
};

std::tuple<cv::Mat, int> read_any_image_data(const ImageFileDescriptor& desc);
