include_guard()

FetchContent_Declare_Local(eigen eigen 3.4.0)

set(BUILD_TESTING OFF)
set(EIGEN_BUILD_DOC OFF)
set(EIGEN_BUILD_BTL OFF)

fetchContent_MakeAvailable(eigen)
