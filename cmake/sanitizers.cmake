if(ENABLE_ASAN)
    list(APPEND SANITIZERS address)
endif()
if(ENABLE_UBSAN)
    list(APPEND SANITIZERS undefined)
endif()

list(JOIN SANITIZERS "," SANITIZE_FLAG)
string(PREPEND SANITIZE_FLAG "-fsanitize=")

include(CheckCXXCompilerFlag)
include(CMakePushCheckState)

CMAKE_PUSH_CHECK_STATE(RESET)

set(CMAKE_REQUIRED_LINK_OPTIONS "${SANITIZE_FLAG}")
CHECK_CXX_COMPILER_FLAG("${SANITIZE_FLAG}" SANITIZERS_SUPPORTED)

if(SANITIZERS_SUPPORTED AND SANITIZERS)
    add_compile_options(${SANITIZE_FLAG})
    add_link_options(${SANITIZE_FLAG})
    if(NOT MSVC)
        add_compile_options(-g -fno-omit-frame-pointer)
        add_link_options(-g)
    endif()
elseif(SANITIZERS)
    message(FATAL_ERROR "${SANITIZE_FLAG} not supported!")
endif()

CMAKE_POP_CHECK_STATE()
