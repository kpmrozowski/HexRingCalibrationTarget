set(MODULE_NAME io)
set(LIB_NAME ${PROJECT_NAME}_${MODULE_NAME})
set(EXPORT_LIB_NAME ${PROJECT_NAME}::${MODULE_NAME})

set(LIB_SRCS
    debug.cpp
    images_set.cpp
    save_path.cpp
)

add_library(${LIB_NAME} ${LIB_SRCS})
add_library(${EXPORT_LIB_NAME} ALIAS ${LIB_NAME})

target_include_directories(${LIB_NAME}
  PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>/..
)

target_link_libraries(${LIB_NAME}
  PRIVATE
    opencv::imgcodecs
    happly
    spdlog
  PUBLIC
    Eigen3::Eigen
    opencv::core
    nlohmann_json
)
