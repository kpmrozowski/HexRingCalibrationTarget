#pragma once

#include <string_view>

#include <CLI/CLI.hpp>

#include <spdlog/cfg/env.h>
#include <spdlog/spdlog.h>

#include "subcommand.hpp"

namespace utils
{
template <std::derived_from<Subcommand>... Subcommands>
class CmdLauncher
{
    CLI::App app_;
    std::tuple<Subcommands...> subcommands_;

   public:
    CmdLauncher(const std::string_view& about) : app_{std::string(about)}
    {
        spdlog::cfg::load_env_levels();

        subcommands_ = std::make_tuple<Subcommands...>(Subcommands()...);
        std::apply([this](auto&&... args) { (args.set_subcommand(app_), ...); }, subcommands_);

        app_.require_subcommand(1);
    }

    void launch(const int argc, const char* const* const argv)
    {
        try
        {
            app_.parse(argc, argv);
        }
        catch (CLI::ParseError& e)
        {
            std::exit(app_.exit(e));
        }

        spdlog::info("Subcommand {} success. Before you leave don't forget to pick up the results! :)", argv[0]);
    }
};

}  // namespace utils
