
if (BUILD_TARGET_CHIP STREQUAL "onnx")
    message("Build onnxruntime code")

    include_directories(/usr/local/onnxruntime/include)
    link_directories(/usr/local/onnxruntime/lib)

    set(REGISTER_LIBS "${REGISTER_LIBS};onnxruntime")

    add_compile_definitions(WITH_ORT)

    file(GLOB REGISTER_SRC_FILES ${CMAKE_SOURCE_DIR}/src/register/devices/onnx/*.cpp)

    install(DIRECTORY /usr/local/onnxruntime/ DESTINATION thirdparty)
endif()