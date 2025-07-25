include_guard()
include(FetchContent)

function(FetchContent_Declare_Local name deps version)
    FetchContent_Declare(
        ${name}
        URL ${CMAKE_SOURCE_DIR}/deps/${deps}
        SYSTEM
        OVERRIDE_FIND_PACKAGE
    )
endfunction()
