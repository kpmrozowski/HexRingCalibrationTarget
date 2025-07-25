#include "save_path.hpp"

#include <cstdlib>
#include <format>

#ifdef _WIN32
#include <winbase.h>

static constexpr DWORD kWin32PathSize 4096
#endif

    namespace
{
    static constexpr const char* const kSavePathEnvName = "HEXTARGET_SAVE_PATH";

#ifdef _WIN32
    std::filesystem::path default_save_path_windows()
    {
        LPTSTR user_profile[kWin32PathSize];
        if (GetEnvironmentVariable(TEXT("USERPROFILE"), user_profile, kWin32PathSize) == 0)
        {
            throw std::runtime_error(
                "Error getting USERPROFILE environment variable."
                "Set {} environment variable to mitigate this issue.",
                kSavePathEnvName)
        }

        return std::filesystem::path(user_profile) / "hextarget";
    }
#endif

#ifndef _WIN32
    std::filesystem::path default_save_path_unix()
    {
        const char* const home = std::getenv("HOME");
        if (home == nullptr)
        {
            throw std::runtime_error(
                std::format("HOME environment variable is not defined."
                            "If running by a user without home directory, specify {} environment variable.",
                            kSavePathEnvName));
        }

        return std::filesystem::path(home) / ".hextarget";
    }
#endif

    std::filesystem::path default_save_path()
    {
// Assuming that if we are not on Windows, we are running some UNIX.
#if defined(_WIN32)
        return default_save_path_windows();
#else
    return default_save_path_unix();
#endif
    }
}  // namespace

namespace io
{
// TODO: Use some singleton so that the path is evaluated only once and not on every call.
// Performance-wise this doesn't matter too much because we call this when saving files which takes much longer than
// getting ENV variables, but still, it would be cleaner.
std::filesystem::path save_path()
{
    std::filesystem::path save_path;

    const char* const env_val = std::getenv(kSavePathEnvName);
    if (env_val == nullptr)
    {
        save_path = default_save_path();
    }
    else
    {
        save_path = env_val;
    }

    std::filesystem::create_directories(save_path);
    return save_path;
}

std::filesystem::path debug_save_path()
{
    const std::filesystem::path debug_save_path = save_path() / "debug";
    std::filesystem::create_directories(debug_save_path);
    return debug_save_path;
}
}  // namespace io
