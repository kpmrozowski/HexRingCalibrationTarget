#include <cmd_launcher/cmd_launcher.hpp>

#include "subcommands/calibrate_mono.hpp"
#include "subcommands/draw_board.hpp"

static constexpr std::string_view kAbout = "calibration";

int main(int argc, char* argv[])
{
    utils::CmdLauncher<CalibrateMono, DrawBoard> launcher(kAbout);
    launcher.launch(argc, argv);

    return EXIT_SUCCESS;
}
