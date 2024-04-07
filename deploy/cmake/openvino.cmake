
if (BUILD_TARGET_CHIP STREQUAL "intel")
    message("Build openvino code")

    # set(OpenVINO_Runtime_PATH /opt/openvino_2022/)
    set(OpenVINO_Runtime_PATH /opt/intel/openvino_2023.0.1)
    message(${OpenVINO_Runtime_PATH}/cmake)
    find_package(OpenVINO REQUIRED  PATHS ${OpenVINO_Runtime_PATH}/runtime/cmake NO_DEFAULT_PATH)
    find_package(OpenVINO REQUIRED COMPONENTS Runtime)

    include_directories(${OpenVINO_Runtime_PATH}/runtime/include/)
    link_directories(${OpenVINO_Runtime_PATH}/runtime/lib/intel64/)

    set(REGISTER_LIBS "${REGISTER_LIBS};openvino::runtime")

    add_compile_definitions(WITH_INTEL)

    file(GLOB REGISTER_SRC_FILES ${CMAKE_SOURCE_DIR}/src/register/devices/intel/*.cpp)

    install(DIRECTORY ${OpenVINO_Runtime_PATH} DESTINATION thirdparty)
endif()