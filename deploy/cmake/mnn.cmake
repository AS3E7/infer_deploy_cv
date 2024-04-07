if (BUILD_TARGET_CHIP STREQUAL "mnn")
    message("Build mnn code")
    aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/devices/mnn        REGISTER_SRC_FILES)

    include_directories(/usr/local/mnn_x86/include)
    link_directories(/usr/local/mnn_x86/lib)
    set(MNN_LIBS MNN)

    list(APPEND REGISTER_LIBS ${MNN_LIBS})

    add_compile_definitions(WITH_MNN)

    file(GLOB REGISTER_SRC_FILES ${CMAKE_SOURCE_DIR}/src/register/devices/mnn/*.cpp)

    install(DIRECTORY /usr/local/mnn/ DESTINATION thirdparty)
endif()
