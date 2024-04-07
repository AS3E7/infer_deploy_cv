
if("${BUILD_TARGET_CHIP}" MATCHES "cambricon")
    # 设置编辑的头文件引用路径
    if (BUILD_TARGET_CHIP STREQUAL "cambricon_mlu220" AND TARGET_ARCH STREQUAL "aarch64")
        set(NEUWARE_HOME /usr/local/aarch64-mlu220-soc)
        add_compile_definitions(WITH_MLU220)

        set(CNLIBS cncv cnrt cndrv cncodec cndev ion)
    elseif (BUILD_TARGET_CHIP STREQUAL "cambricon_mlu270" AND TARGET_ARCH STREQUAL "x86")
        set(NEUWARE_HOME /usr/local/aarch64-mlu270-x86)
        add_compile_definitions(WITH_MLU270)

        set(CNLIBS cncv cnrt cndrv cncodec cndev)
    endif()

    include_directories(${NEUWARE_HOME}/include)
    link_directories(${NEUWARE_HOME}/lib)
    link_directories(${NEUWARE_HOME}/lib64)

    set(REGISTER_LIBS "${REGISTER_LIBS};${CNLIBS}") #bmrt bmcv bmlib bmvideo bmvpuapi bmvpulite bmjpuapi bmion

    file(GLOB REGISTER_SRC_FILES ${CMAKE_SOURCE_DIR}/src/register/devices/cambricon/*.cpp)
    # file(GLOB REGISTER_SRC_FILES ${CMAKE_SOURCE_DIR}/src/register/devices/cambricon/transform_cncv/*.cpp)
    message(STATUS "REGISTER_SRC_FILES: ${REGISTER_SRC_FILES}")

    install(DIRECTORY ${NEUWARE_HOME}/lib64 DESTINATION thirdparty)
    install(DIRECTORY ${NEUWARE_HOME}/include DESTINATION thirdparty)
endif()