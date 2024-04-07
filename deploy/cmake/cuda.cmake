if(BUILD_TARGET_CHIP STREQUAL "nvidia")
    set(CMAKE_CXX_STANDARD 17)
    # 设置编辑的头文件引用路径
    include_directories(/usr/local/cuda-11.4/targets/x86_64-linux/include/)
    include_directories(/usr/local/cuda_common/include/)
    
    
if (TARGET_ARCH STREQUAL "aarch64")
    add_compile_definitions(AARCH64)
    include_directories(${CMAKE_SOURCE_DIR}/thirdparty/libmodcry_aarch64/include/openssl/)
else()
    include_directories(${CMAKE_SOURCE_DIR}/thirdparty/libmodcry_x86/include/openssl/)
    link_directories(/usr/local/cuda-11.4/targets/x86_64-linux/lib)
endif()

add_compile_definitions(WITH_NVIDIA)

# set(REGISTER_LIBS "${REGISTER_LIBS};bmrt;bmcv;bmlib;bmvideo;bmvpuapi;bmion")
set(REGISTER_LIBS "${REGISTER_LIBS};nvinfer;nvonnxparser;cudart_static;nppig;npps;nppicc;nppidei;nppisu;nvjpeg;nvidia-ml")
set(APP_LIBS "cudart_static;nppig;npps;nppicc;nppidei;nppisu;nvjpeg")

file(GLOB REGISTER_SRC_FILES ${CMAKE_SOURCE_DIR}/src/register/devices/nvidia/*.cpp)

endif()