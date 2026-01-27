cmake_minimum_required(VERSION 3.28)

project(HexRingCalibrationTarget)

option(HEXRINGTARGET_BUILD_TESTS "Build unit tests" OFF)

include(cmake/options.cmake)

disable_all_warnings()

include("${deps_dir}/opencv.cmake")
include("${deps_dir}/cli11.cmake")
include("${deps_dir}/eigen.cmake")
include("${deps_dir}/ceres.cmake")
include("${deps_dir}/json.cmake")
include("${deps_dir}/happly.cmake")
include("${deps_dir}/fmtlib.cmake")
include("${deps_dir}/spdlog.cmake")

reenable_warnings()

add_subdirectory(src)

if(HEXRINGTARGET_BUILD_TESTS)
    add_subdirectory(tests)
endif()
