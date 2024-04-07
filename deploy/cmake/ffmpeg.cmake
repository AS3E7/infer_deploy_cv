find_package(PkgConfig QUIET)

if(PkgConfig_FOUND)
    if (TARGET_ARCH STREQUAL "x86")
        if(${BUILD_TARGET_CHIP} STREQUAL "bmnn")
            set(ENV{PKG_CONFIG_PATH} /usr/local/ffmpeg_x86_cpu/lib/pkgconfig/:$ENV{PKG_CONFIG_PATH})
        elseif(${BUILD_TARGET_CHIP} STREQUAL "ts")
            set(ENV{PKG_CONFIG_PATH} /usr/local/ts_tx5368/thirdparty/ffmpeg/lib/pkgconfig/:$ENV{PKG_CONFIG_PATH})
            install(DIRECTORY /usr/local/ts_tx5368/thirdparty/ffmpeg/lib/ DESTINATION thirdparty)
        elseif(${BUILD_TARGET_CHIP} STREQUAL "nvidia")
            message(STATUS "Found FFmpeg....")
            set(ENV{PKG_CONFIG_PATH} /usr/local/ffmpeg_x86_cuda/lib/pkgconfig/:$ENV{PKG_CONFIG_PATH})
        else()
            # set(ENV{PKG_CONFIG_PATH} /usr/local/ffmpeg_x86_cuda/lib/pkgconfig/:$ENV{PKG_CONFIG_PATH})
            set(ENV{PKG_CONFIG_PATH} /usr/local/ffmpeg_4.2_x86_ubuntu18_gcc7.5/lib/pkgconfig/:$ENV{PKG_CONFIG_PATH})
            install(DIRECTORY /usr/local/ffmpeg_4.2_x86_ubuntu18_gcc7.5 DESTINATION thirdparty)
        endif()
        pkg_check_modules(FFMPEG REQUIRED IMPORTED_TARGET libavcodec libavfilter libavformat libavutil libswscale libswresample)
    else()
        if(${BUILD_TARGET_CHIP} STREQUAL "ts")
            message("!#!#!#!##tsing")
            set(ENV{PKG_CONFIG_PATH} /usr/local/ts_tx5368/thirdparty/ffmpeg/lib/pkgconfig/:$ENV{PKG_CONFIG_PATH})
            pkg_check_modules(FFMPEG REQUIRED IMPORTED_TARGET libavcodec libavformat libavutil)
            install(DIRECTORY /usr/local/ts_tx5368/thirdparty/ffmpeg/ DESTINATION thirdparty/lib)
        else()
            set(ENV{PKG_CONFIG_PATH} /usr/local/ffmpeg_aarch64_cpu/lib/pkgconfig/:$ENV{PKG_CONFIG_PATH})
            install(DIRECTORY /usr/local/ffmpeg_aarch64_cpu/ DESTINATION thirdparty)
            pkg_check_modules(FFMPEG REQUIRED IMPORTED_TARGET libavcodec libavfilter libavformat libavutil libswscale libswresample)
        endif()
    endif()
    
else()
    find_package(FFMPEG QUIET)
endif()

if(FFMPEG_FOUND)
    message(STATUS "Found FFmpeg: ${FFMPEG_CONFIG} (found version \"${FFMPEG_VERSION}\")")
    add_compile_definitions(WITH_FFMPEG)
    include_directories(${FFMPEG_INCLUDE_DIRS})
    link_directories(${FFMPEG_LIBRARY_DIRS})
    message("!!!!!!!!!!!!!!!!!!FFMPEG lib:${FFMPEG_INCLUDE_DIRS}")

    set(APP_LIBS ${APP_LIBS};${FFMPEG_LIBRARIES})
endif()