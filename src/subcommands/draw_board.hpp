#pragma once

#include <cmd_launcher/subcommand.hpp>

class DrawBoard : public utils::Subcommand
{
   private:
    int board_type_;

    std::vector<float> board_params_;
    std::filesystem::path board_params_path_;
    int resolution_;
    int dpi_;

   public:
    std::string name() const override { return "DrawBoard"; }

    std::string description() const override { return "Draw a board (Rect or Hex)"; }

    void set_options(CLI::App& cmd) override
    {
        add_board(cmd, board_type_)->required();
        add_board_params(cmd, board_params_);

        cmd.add_option("--board-params-path", board_params_path_, "Path to board parameters json.")
            ->check(CLI::ExistingFile);
        cmd.add_option("-r, --resolution", resolution_, "AX, where X={-2, -1, 0, 1, 2, 3, 4, 5}.")->default_val(3);
        cmd.add_option("-d, --dpi", dpi_, "Dots per inch.")->default_val(300);
    }

    void execute() override;
};
