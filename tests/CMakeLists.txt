include(FetchContent)

FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG v3.5.0
)
FetchContent_MakeAvailable(Catch2)

enable_testing()

add_executable(HexRingCalibrationTarget_tests
    identification/test_identify_markers_by_row_lines.cpp
)

target_link_libraries(HexRingCalibrationTarget_tests
    PRIVATE
        HexRingCalibrationTarget_calibration
        Catch2::Catch2WithMain
        Eigen3::Eigen
)

add_test(NAME HexRingCalibrationTarget_tests COMMAND HexRingCalibrationTarget_tests)
