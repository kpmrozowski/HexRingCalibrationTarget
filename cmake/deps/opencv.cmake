include_guard()

FetchContent_Declare_Local(opencv opencv 4.10.0-stripped)

set(WITH_EIGEN OFF)
set(WITH_OPENEXR OFF)
set(WITH_ITT OFF)
set(WITH_FFMPEG OFF)
set(WITH_GSTREAMER OFF)
set(WITH_OPENCL OFF)
set(WITH_1394 OFF)
set(WITH_GTK OFF)
set(WITH_ADE OFF)
set(WITH_VA OFF)
set(WITH_VTK OFF)
set(WITH_WEBP OFF)
set(WITH_PROTOBUF OFF)
set(WITH_FLATBUFFERS OFF)
set(BUILD_TESTS OFF)
set(BUILD_PERF_TESTS OFF)
set(BUILD_EXAMPLES OFF)
set(BUILD_TIFF OFF)
set(BUILD_PNG OFF)
set(BUILD_ZLIB OFF)
set(BUILD_JAVA OFF)
set(BUILD_opencv_apps OFF)
set(BUILD_SHARED_LIBS OFF)
set(BUILD_WITH_STATIC_CRT ${STATIC_CRT})

set(BUILD_LIST core,imgproc,imgcodecs,calib3d CACHE BOOL "" FORCE)

fetchContent_MakeAvailable(opencv)

foreach(the_module ${OPENCV_MODULES_BUILD})
    if(TARGET ${the_module})
        set_target_properties(
            ${the_module}
            PROPERTIES
                INTERFACE_INCLUDE_DIRECTORIES
                    $<TARGET_PROPERTY:${the_module},INCLUDE_DIRECTORIES>
        )
        string(REPLACE "opencv_" "" module_name ${the_module})
        add_library(opencv::${module_name} ALIAS ${the_module})
    endif()
endforeach()

