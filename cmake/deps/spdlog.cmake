include_guard()

FetchContent_Declare_Local(spdlog spdlog v1.14.1)
set(SPDLOG_FMT_EXTERNAL_HO ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
fetchContent_MakeAvailable(spdlog)
