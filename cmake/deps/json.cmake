include_guard()

FetchContent_Declare_Local(nlohmann_json json v3.11.3)

set(JSON_BuildTests OFF)

fetchContent_MakeAvailable(nlohmann_json)
