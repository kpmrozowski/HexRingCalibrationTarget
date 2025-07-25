include_guard()

FetchContent_Declare_Local(ceres ceres 2.2.0-fix)

set(BUILD_TESTING OFF)
set(BUILD_EXAMPLES OFF)
set(BUILD_BENCHMARKS OFF)
set(MINIGLOG ON)
set(GFLAGS OFF)
set(USE_CUDA OFF)

fetchContent_MakeAvailable(ceres)
