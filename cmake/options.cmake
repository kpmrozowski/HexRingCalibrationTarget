set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin)

set(deps_dir ${CMAKE_SOURCE_DIR}/cmake/deps)
set(FETCHCONTENT_QUIET OFF)

option(ENABLE_ASAN "ASAN" OFF)
option(ENABLE_UBSAN "UBSAN" OFF)
option(X86_OPTIMIZATIONS "enable additional x86 optimization (x86-64-v2, AVX)" ON)

include("${CMAKE_SOURCE_DIR}/cmake/openmp.cmake")
include("${CMAKE_SOURCE_DIR}/cmake/sanitizers.cmake")
include("${CMAKE_SOURCE_DIR}/cmake/fetchcontent.cmake")

option(BUILD_TOOLS "Add to build optional targets used for debugging" ON)

set(BUILD_SHARED_LIBS OFF CACHE BOOL "Prefer using shared libraries")

set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)

add_custom_target(uninstall)
function(add_custom_target target_name)
    if(${target_name} STREQUAL "uninstall")
        return()
    endif()
    #this invokes original function
    _add_custom_target(${ARGV})
endfunction()

if(MSVC)
    add_compile_options(/W4)
    add_definitions(/bigobj)
else()
    add_compile_options(-Wall -Wextra)
    if(X86_OPTIMIZATIONS)
        add_compile_options(-march=x86-64-v2 -mavx)
    endif()
endif()

macro(disable_all_warnings)
    if (MSVC)
        add_compile_options(/w)
    else()
        add_compile_options(-w)
    endif()
endmacro()

macro(reenable_warnings)
    get_directory_property(compile_options COMPILE_OPTIONS)
    list(REMOVE_ITEM compile_options "/w")
    list(REMOVE_ITEM compile_options "-w")
    set_directory_properties(PROPERTIES COMPILE_OPTIONS "${compile_options}")
endmacro()
