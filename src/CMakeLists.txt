set(MODULE_NAME calibration)
set(LIB_NAME ${PROJECT_NAME}_${MODULE_NAME})
set(EXPORT_LIB_NAME ${PROJECT_NAME}::${MODULE_NAME})

add_subdirectory(cmd_launcher)
add_subdirectory(io)

set(LIB_SRCS
    board.cpp
    board_drawing.cpp
    calibration.cpp
    marker/debug.cpp
    marker/knn.cpp
    marker/detection.cpp
    marker/detection_parameters.cpp
    marker/symmetries.cpp
    marker/thresholds.cpp
    identification/board_circle/identification_circle.cpp
    identification/board_hex/filtering.cpp
    identification/board_hex/identification_hex.cpp
    identification/board_hex/ordering_hex.cpp
    identification/board_rect/filtering_rect.cpp
    identification/board_rect/identification_rect.cpp
    identification/board_rect/ordering_rect.cpp
    initial_calibration/debug.cpp
    initial_calibration/precalibration.cpp
)

add_library(${LIB_NAME} ${LIB_SRCS})
add_library(${EXPORT_LIB_NAME} ALIAS ${LIB_NAME})

target_include_directories(
  ${LIB_NAME}
    PUBLIC
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>)

target_link_libraries(${LIB_NAME}
  PRIVATE
    opencv::core
    opencv::calib3d
    opencv::imgproc
    opencv::features2d
    Eigen3::Eigen
    ceres
    happly
    spdlog
    fmt
  PUBLIC
    nlohmann_json
    HexRingCalibrationTarget::io
)

add_executable(
  ${MODULE_NAME}
    subcommands/calibrate_mono.cpp
    subcommands/draw_board.cpp
    main.cpp
)

target_link_libraries(
  ${MODULE_NAME}
    PRIVATE
      ${LIB_NAME}
      opencv::imgcodecs
      opencv::calib3d
      HexRingCalibrationTarget::cmd_launcher
)
