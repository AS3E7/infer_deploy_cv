cmake_minimum_required(VERSION 3.14.0)
project(programs)

find_package(Boost COMPONENTS program_options QUIET)

function(add_program)
    get_filename_component(ProgramName ${ARGV0} NAME_WE)
    add_executable(${ProgramName} ${ARGV})
    target_link_libraries(${ProgramName} ffmpegwrapper ${LinkLibraries} gtest)
endfunction(add_program)

if(FFMPEG_FOUND)
    file(GLOB test_src_files
        "${CMAKE_CURRENT_SOURCE_DIR}/ffmpeg_samples/*.c??"
        "${CMAKE_CURRENT_SOURCE_DIR}/*.c??")
else()
    file(GLOB test_src_files
        "${CMAKE_CURRENT_SOURCE_DIR}/*.c??")
endif()

foreach(file IN LISTS test_src_files)
    add_program(${file})
endforeach(file)

if(OpenCV_FOUND)
    add_subdirectory(CV_Samples)
endif()

set(CMAKE_INSTALL_PREFIX "${CMAKE_SOURCE_DIR}/release/${CMAKE_SYSTEM_PROCESSOR}/${TARGET_CHIP}/${CHIP_NAME}")

if(CMAKE_BUILD_TYPE STREQUAL "Release")
    if(NOT (${TARGET_CHIP} STREQUAL "bm1684" OR ${TARGET_CHIP} STREQUAL "jetson"))
        install(DIRECTORY ${FFMPEG_LIBRARY_DIRS}
            DESTINATION ${CMAKE_INSTALL_PREFIX}
            FILES_MATCHING PATTERN "libav*.so.*" PATTERN "libsw*.so.*")
    endif()

    if(${TARGET_CHIP} STREQUAL "nvidia"
        OR ${TARGET_CHIP} STREQUAL "intel"
        OR ${TARGET_CHIP} STREQUAL "mlu220"
        OR ${TARGET_CHIP} STREQUAL "mlu270"
        OR ${TARGET_CHIP} STREQUAL "mlu370"
        OR ${TARGET_CHIP} STREQUAL "rv1126")
        install(DIRECTORY ${OpenCV_INSTALL_PATH}/lib
            DESTINATION ${CMAKE_INSTALL_PREFIX}
            FILES_MATCHING PATTERN "*.so.*")
    endif()

    install(TARGETS node_srv RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin)
    install(DIRECTORY ${CMAKE_BINARY_DIR}/_deps/inference-sdk-src/lib/
        DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
        FILES_MATCHING PATTERN "*.so*")
endif()

# add_library(ai_solution_static STATIC ${CMAKE_SOURCE_DIR}/programs/modules/test_ai_solution.cpp)
# add_executable(test_ai_solution ${CMAKE_SOURCE_DIR}/programs/modules/test_ai_solution.cpp)
# target_link_libraries(test_ai_solution PRIVATE ai_solution ${OpenCV_LIBRARIES} ${FFMPEG_LIBRARIES} pthread dl)

# if (${TARGET_CHIP} STREQUAL "mlu220")
# target_link_libraries(test_ai_solution PRIVATE gddsdk ${FFMPEG_LIBRARIES} cncodec cnrt ion jpu easydk postproc glog cncv crypto ssl pthread dl)
# endif()