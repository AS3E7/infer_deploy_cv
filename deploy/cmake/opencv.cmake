# find_package(OpenCV QUIET)
# if (OpenCV_FOUND)
#     message(STATUS "Found OpenCV: ${OpenCV_CONFIG} (found version \"${OpenCV_VERSION}\")")
#     file(GLOB OpenCvFiles
#             deps/yuv_rgb/yuv_rgb.c
#             src/OpenCV/cv_tools.cpp
#             src/OpenCV/cv_common.cpp
#             src/modules/tracking/*.c??
#             src/modules/report_result.cpp
#             src/modules/cvrelate/*.cpp
#             )
#     add_compile_definitions(WITH_OPENCV2)
#     include_directories(${OpenCV_INCLUDE_DIRS})
#     set(LibFiles "${LibFiles};${OpenCvFiles}")
#     message(STATUS "OpenCV_INCLUDE_DIRS: ${OpenCV_INCLUDE_DIRS}")
#     set(LibLibraries "${LibLibraries};${OpenCV_LIBRARIES}")
# endif ()

# if (${CMAKE_SYSTEM_PROCESSOR} STREQUAL "x86_64")

# find_library(Opencv_librar_core.so)

if(DEFINED BUILD_TARGET_CHIP)
    if(${BUILD_TARGET_CHIP} STREQUAL "bmnn")
        # find_package(OpenCV REQUIRED PATHS "/usr/local/bmnn/lib/opencv/soc/cmake" NO_DEFAULT_PATH)
        # find_package(OpenCV REQUIRED PATHS "/usr/local/aarch64-bm1684-soc/lib/cmake/" NO_DEFAULT_PATH)
        find_package(OpenCV REQUIRED PATHS "/usr/local/sophgo_sdk_v0.4.4/lib/cmake/" NO_DEFAULT_PATH)

        if (OPENCV_FOUND)
            message("Opencv will be use bmnn!!!")
        endif()
    elseif (${BUILD_TARGET_CHIP} STREQUAL "ts")
        find_package(OpenCV REQUIRED PATHS "/usr/local/ts_tx5368/thirdparty/opencv-4.5.5_aarch64/lib/cmake/" NO_DEFAULT_PATH)

        if (OPENCV_FOUND)
            message("Opencv will be use tsingmicro!!!")
        endif()

        install(DIRECTORY /usr/local/ts_tx5368/thirdparty/opencv-4.5.5_aarch64/ DESTINATION thirdparty)
    elseif (${BUILD_TARGET_CHIP} STREQUAL "nvidia")
        if (TARGET_ARCH STREQUAL "x86")
            find_package(OpenCV REQUIRED PATHS "/usr/local/opencv_x86_cuda//lib/cmake/" NO_DEFAULT_PATH)
        endif()
    else()
        message("!!!!!!!!!!!!!!!!!!!!!${TARGET_ARCH}")
        if (TARGET_ARCH STREQUAL "x86")
            set(OpenCV_Path "/usr/local/opencv-4.5.5_x86_ubuntu18_gcc7.5")
            # find_package(OpenCV REQUIRED PATHS "/usr/local/opencv-4.5.5/lib/cmake" NO_DEFAULT_PATH)
            find_package(OpenCV REQUIRED PATHS "${OpenCV_Path}/lib/cmake" NO_DEFAULT_PATH)
            # find_package(OpenCV REQUIRED PATHS "/volume1/gddi-data/lgy/common/opencv/lib/cmake" NO_DEFAULT_PATH)
            
            include_directories(${OpenCV_INCLUDE_DIRS})
            link_directories(${OpenCV_LIBS})
            install(DIRECTORY ${OpenCV_Path} DESTINATION thirdparty)
        else()
            find_package(OpenCV REQUIRED PATHS "/usr/local/opencv-4.5.5_aarch64/lib/cmake" NO_DEFAULT_PATH)
            # find_package(OpenCV REQUIRED PATHS "/usr/local/opencv-4.5.5_aarch64_ts/lib/cmake" NO_DEFAULT_PATH)

            include_directories(${OpenCV_INCLUDE_DIRS})
            link_directories(${OpenCV_LIBS})

            install(DIRECTORY /usr/local/opencv-4.5.5_aarch64/ DESTINATION thirdparty)
        endif()

    endif()

    if (OPENCV_FOUND)
        include_directories(${OpenCV_INCLUDE_DIRS})
        link_directories(${OpenCV_LIBS})
        message(${OpenCV_INCLUDE_DIRS})
    endif()

    
endif()