if (BUILD_TARGET_CHIP MATCHES "rk")

    set(RKNPU_BASE_PATH /usr/local/rknpu2_1.4)
    if (BUILD_TARGET_CHIP STREQUAL "rk3568")
        set(Rk_SDK_Paths ${RKNPU_BASE_PATH}/rknpu2_1.4_rk356x)
        # rga lib
        set(RGA_LIBS ${RKNPU_BASE_PATH}/librga_1.3/)
        add_compile_definitions(WITH_RK3568)
        
        add_compile_definitions(RKNN_RGA_VEWSION_1_3) 
        install(DIRECTORY ${Rk_SDK_Paths} DESTINATION thirdparty)
    elseif (BUILD_TARGET_CHIP STREQUAL "rk3588")
        set(Rk_SDK_Paths ${RKNPU_BASE_PATH}/rknpu2_1.4_rk3588)
        set(RGA_LIBS ${RKNPU_BASE_PATH}/librga_1.9/)

        add_compile_definitions(RKNN_RGA_VEWSION_1_9) 
        add_compile_definitions(WITH_RK3588)
        install(DIRECTORY ${Rk_SDK_Paths} DESTINATION thirdparty)
        install(DIRECTORY ${RGA_LIBS} DESTINATION thirdparty)
    endif()
    include_directories(${Rk_SDK_Paths}/include)
    link_directories(${Rk_SDK_Paths}/lib)

    include_directories(${RGA_LIBS}/include)
    link_directories(${RGA_LIBS}/lib)

    message("Rk_SDK_Paths: ${Rk_SDK_Paths}")

    set(REGISTER_LIBS "${REGISTER_LIBS};rknnrt;rknn_api;rga")

    file(GLOB REGISTER_SRC_FILES ${CMAKE_SOURCE_DIR}/src/register/devices/rk/*.cpp)
    
endif()