# clipper 相关

if (TARGET_ARCH STREQUAL "x86")
    if(BUILD_TARGET_CHIP STREQUAL "nvidia")
        set(Clipper_DIR /usr/local/x86-common/Clipper2)
    endif()
endif()
include_directories(${Clipper_DIR}/include)
link_directories(${Clipper_DIR}/lib)
# install(DIRECTORY ${Spdlog_DIR} DESTINATION thirdparty) 
