# spdlog 相关

if (TARGET_ARCH STREQUAL "x86")
    if(BUILD_TARGET_CHIP STREQUAL "nvidia")
        set(Spdlog_DIR /usr/local/x86-common/spdlog_1.8.5_gcc7.5)
    else()
        set(Spdlog_DIR /usr/local/x86-common)
    endif()
else() # 如果BUILD_TARGET_CHIP不是ts
    if(BUILD_TARGET_CHIP STREQUAL "ts")
        set(Spdlog_DIR /usr/local/ts_tx5368/thirdparty/spdlog)
    else()
        set(Spdlog_DIR /usr/local/aarch64-common)
    endif()
endif()
message(STATUS "Spdlog_DIR: ${Spdlog_DIR}")
include_directories(${Spdlog_DIR}/include)
link_directories(${Spdlog_DIR}/lib)
# install(DIRECTORY ${Spdlog_DIR} DESTINATION thirdparty) 
