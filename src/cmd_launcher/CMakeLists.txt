set(MODULE_NAME cmd_launcher)
set(LIB_NAME ${PROJECT_NAME}_${MODULE_NAME})
set(EXPORT_LIB_NAME ${PROJECT_NAME}::${MODULE_NAME})

add_library(${LIB_NAME} subcommand.cpp)
add_library(${EXPORT_LIB_NAME} ALIAS ${LIB_NAME})

target_link_libraries(${LIB_NAME}
  PUBLIC
    CLI11
    spdlog
)

target_include_directories(${LIB_NAME}
  INTERFACE
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src>
)
