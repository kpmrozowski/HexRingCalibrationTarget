#pragma once

#include <filesystem>
#include <string>

#include <CLI/CLI.hpp>
#include <variant>

namespace utils
{
class Subcommand
{
   protected:
    virtual std::string name() const = 0;
    virtual std::string description() const = 0;

    virtual void set_options(CLI::App& cmd) = 0;
    virtual void execute() = 0;

    CLI::Option* add_dataset_path(CLI::App& cmd, std::filesystem::path& dataset_folder);
    CLI::Option* add_path_to_save(CLI::App& cmd, std::filesystem::path& output_folder);
    CLI::Option* add_camera(CLI::App& cmd, std::string& camera_id);
    CLI::Option* add_board(CLI::App& cmd, int& board_type);
    CLI::Option* add_board_params(CLI::App& cmd, std::vector<float>& board_params);

   public:
    CLI::App& set_subcommand(CLI::App& app);
    void set_main_command(CLI::App& app);

    virtual ~Subcommand() = default;
};

}  // namespace utils
