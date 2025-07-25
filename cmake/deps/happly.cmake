include_guard()

FetchContent_Declare_Local(happly happly 8a60630)

fetchContent_MakeAvailable(happly)

add_library(happly INTERFACE)
target_include_directories(happly INTERFACE ${happly_SOURCE_DIR})
