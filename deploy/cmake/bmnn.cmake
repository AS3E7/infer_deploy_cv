
if(BUILD_TARGET_CHIP STREQUAL "bmnn")

find_package(PkgConfig QUIET)

# set(SOPHGO_SDK_VERSION v0.4.5)
# set(SOPHGO_SDK_VERSION v0.4.4)
if (SOPHGO_SDK_VERSION STREQUAL "v0.4.5")
    set(SOPHGO_SDK_PATH /usr/local/sophgo_sdk_v0.4.5)
    include_directories("${SOPHGO_SDK_PATH}/include")
    link_directories(${SOPHGO_SDK_PATH}/lib)

    set(ENV{PKG_CONFIG_PATH} ${SOPHGO_SDK_PATH}/lib/pkgconfig/:$ENV{PKG_CONFIG_PATH})
elseif (SOPHGO_SDK_VERSION STREQUAL "v0.4.4")
    set(SOPHGO_SDK_PATH /usr/local/sophgo_sdk_v0.4.4)
    include_directories(${SOPHGO_SDK_PATH}/include)
    link_directories(${SOPHGO_SDK_PATH}/lib)

    set(ENV{PKG_CONFIG_PATH} ${SOPHGO_SDK_PATH}/lib/pkgconfig/:$ENV{PKG_CONFIG_PATH})
else()
    set(SOPHGO_SDK_PATH /usr/local/bmnn)
    message("!!!!!!!!!!!!not 0.4.5")
    # 设置编辑的头文件引用路径
    include_directories("${SOPHGO_SDK_PATH}/include")
    include_directories("${SOPHGO_SDK_PATH}/include/ffmpeg")
    include_directories("${SOPHGO_SDK_PATH}/include/third_party/boost/include")
    include_directories("${SOPHGO_SDK_PATH}/include/bmcpu")
    include_directories("${SOPHGO_SDK_PATH}/include/bmlib")
    include_directories("${SOPHGO_SDK_PATH}/include/bmruntime")
    include_directories("${SOPHGO_SDK_PATH}/include/opencv/opencv4")

    if (TARGET_ARCH STREQUAL "aarch64")
        link_directories(${SOPHGO_SDK_PATH}/lib/thirdparty/soc)
        link_directories(${SOPHGO_SDK_PATH}/lib/bmnn/soc)
        # link_directories(${SOPHGO_SDK_PATH}/lib/opencv/soc)
        link_directories(${SOPHGO_SDK_PATH}/lib/ffmpeg/soc)
        link_directories(${SOPHGO_SDK_PATH}/lib/decode/soc)
        link_directories(${SOPHGO_SDK_PATH}/lib/boost_lib)

        set(ENV{PKG_CONFIG_PATH} ${SOPHGO_SDK_PATH}/lib/ffmpeg/soc/pkgconfig/:$ENV{PKG_CONFIG_PATH})
    else()
        link_directories(${SOPHGO_SDK_PATH}/lib/thirdparty/x86)
        link_directories(${SOPHGO_SDK_PATH}/lib/bmnn/x86)
        # link_directories(${SOPHGO_SDK_PATH}/lib/opencv/x86)
        link_directories(${SOPHGO_SDK_PATH}/lib/ffmpeg/x86)
        link_directories(${SOPHGO_SDK_PATH}/lib/decode/x86)

        set(ENV{PKG_CONFIG_PATH} ${SOPHGO_SDK_PATH}/lib/ffmpeg/x86/pkgconfig/:$ENV{PKG_CONFIG_PATH})
    endif()
endif()

pkg_check_modules(FFMPEG REQUIRED IMPORTED_TARGET libavcodec libavfilter libavformat libavutil libswscale libswresample)

if(FFMPEG_FOUND)
    message(STATUS "Found FFmpeg: ${FFMPEG_CONFIG} (found version \"${FFMPEG_VERSION}\")")
    add_compile_definitions(WITH_FFMPEG)
    include_directories(${FFMPEG_INCLUDE_DIRS})
    link_directories(${FFMPEG_LIBRARY_DIRS})
endif()

add_compile_definitions(WITH_BM1684)

set(REGISTER_LIBS "${REGISTER_LIBS};bmrt;bmcv;bmlib;bmvideo;bmvpuapi;bmion;${FFMPEG_LIBRARIES}") #bmrt bmcv bmlib bmvideo bmvpuapi bmvpulite bmjpuapi bmion

file(GLOB REGISTER_SRC_FILES ${CMAKE_SOURCE_DIR}/src/register/devices/bmnn/*.cpp)
endif()