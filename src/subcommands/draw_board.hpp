#pragma once

#include <cmd_launcher/subcommand.hpp>

class DrawBoard : public utils::Subcommand
{
   private:
    int board_type_ = 1;
    int dpi_ = 300;
    int resolution_ = 3;

   public:
    std::string name() const override { return "DrawBoard"; }

    std::string description() const override { return "Draw a board (Quad or Hex)"; }

    void set_options(CLI::App& cmd) override
    {
        add_board(cmd, board_type_)->required();
        cmd.add_option("-d, --dpi", dpi_, "Dots per inch.");
        cmd.add_option("-r, --resolution", resolution_, "AX, where X={0, 1, 2, 3, 4, 5}.");
    }

    void execute() override;
};
