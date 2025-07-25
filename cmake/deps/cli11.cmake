include_guard()

FetchContent_Declare_Local(cli11 cli11 v2.4.2)

set(cli11_BUILD_DOCS OFF)
set(cli11_BUILD_TESTS OFF)
set(cli11_BUILD_EXAMPLES OFF)

fetchContent_MakeAvailable(cli11)
