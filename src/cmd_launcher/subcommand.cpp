#include "subcommand.hpp"

#include <spdlog/spdlog.h>

namespace utils
{
CLI::App& Subcommand::set_subcommand(CLI::App& app)
{
    CLI::App* subcommand = app.add_subcommand(name(), description());
    set_options(*subcommand);
    subcommand->callback(
        [this]()
        {
            spdlog::info("Launching {}", name());
            execute();
        });
    return *subcommand;
}

void Subcommand::set_main_command(CLI::App& app)
{
    set_options(app);
    app.callback([this]() { execute(); });
}

CLI::Option* Subcommand::add_dataset_path(CLI::App& cmd, std::filesystem::path& dataset_folder)
{
    return cmd.add_option("--dataset-dir", dataset_folder, "Path to dataset directory")->check(CLI::ExistingDirectory);
}

CLI::Option* Subcommand::add_path_to_save(CLI::App& cmd, std::filesystem::path& output_folder)
{
    return cmd.add_option("-o, --output-dir", output_folder, "Path to save directory");
}

CLI::Option* Subcommand::add_camera(CLI::App& cmd, std::string& camera_id)
{
    return cmd.add_option("-C, --cam", camera_id, "Name of a camera");
}

CLI::Option* Subcommand::add_board(CLI::App& cmd, int& board_type)
{
    return cmd.add_option("-b, --board", board_type, "Board type: 0->RectGrid, 1->HexGrid");
}

CLI::Option* Subcommand::add_board_params(CLI::App& cmd, std::vector<int>& board_params)
{
    return cmd.add_option("-p, --board-params", board_params,
                          "Params for HexRingCalibTarget: [rows, cols, row_left, col_left, row_right, col_right], "
                          "Params for RectRingCalibTarget: unavailable");
}

}  // namespace utils
