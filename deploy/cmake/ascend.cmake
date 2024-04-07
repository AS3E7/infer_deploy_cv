if (BUILD_TARGET_CHIP STREQUAL "ascend")

    if (TARGET_ARCH STREQUAL "x86")
        # set(Ascend_Path /usr/local/Ascend/latest)
        set(Ascend_Path /usr/local/Ascend)
    endif ()

    message(${Ascend_Path}/include)
    # link_directories(${Ascend_Path}/x86_64-linux/lib64)
    link_directories(${Ascend_Path}/acllib/lib64)
    link_directories(${Ascend_Path}/acllib/lib64/stub)
    include_directories(${Ascend_Path}/acllib/include)

    link_directories(/usr/local/Ascend/driver/lib64)
    include_directories(/usr/local/Ascend/driver/include)

    add_compile_definitions(WITH_ASCEND)

    set(REGISTER_LIBS "${REGISTER_LIBS};acl_dvpp;ascendcl") #bmrt bmcv bmlib bmvideo bmvpuapi bmvpulite bmjpuapi bmion

    file(GLOB REGISTER_SRC_FILES ${CMAKE_SOURCE_DIR}/src/register/devices/ascend/*.cpp)

    # install(FILES ${CMAKE_SOURCE_DIR}/docker/dockerfile/ascend.Dockerfile DESTINATION dockerfile)

endif()