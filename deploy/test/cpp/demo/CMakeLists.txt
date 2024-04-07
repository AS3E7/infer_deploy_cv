# set(OpenCV_DIR /usr/local/opencv)
# find_package(OpenCV REQUIRED)

# include_directories(${OpenCV_INCLUDE_DIRS})


# onnx
# add_executable(demo_onnx onnx_infer.cpp yolov5_common.cpp)
# target_link_libraries(demo_onnx onnxruntime ${OpenCV_LIBS})


# # mnn
# link_directories(/usr/local/mnn/lib)
# include_directories(/usr/local/mnn/include)

# add_executable(demo_mnn mnn_infer.cpp yolov5_common.cpp)
# target_link_libraries(demo_mnn MNN ${OpenCV_LIBS})

# bmnn

# set(BMNN_SDK_PATH /usr/local/aarch64-bm1684-soc)

# include_directories("${BMNN_SDK_PATH}/include")
# include_directories("${BMNN_SDK_PATH}/include/bmruntime")
# include_directories("${BMNN_SDK_PATH}/include/bmlib")
# include_directories("${BMNN_SDK_PATH}/include/ffmpeg")
# include_directories("${BMNN_SDK_PATH}/include/third_party/boost/include")
# include_directories("${BMNN_SDK_PATH}/include/bmcpu")
# include_directories("${BMNN_SDK_PATH}/include/opencv/opencv4")


# link_directories(${BMNN_SDK_PATH}/lib/thirdparty/soc)
# link_directories(${BMNN_SDK_PATH}/lib/bmnn/soc)
# link_directories(${BMNN_SDK_PATH}/lib/opencv/soc)
# link_directories(${BMNN_SDK_PATH}/lib/ffmpeg/soc)
# link_directories(${BMNN_SDK_PATH}/lib/decode/soc)

# include_directories(${BMNN_SDK_PATH}/include)
# link_directories(${BMNN_SDK_PATH}/lib/)

# set(BmnnOpencvLibs opencv_core opencv_imgproc opencv_imgcodecs opencv_videoio)
# set(BmnnFFmpegLibs avcodec bmvideo swresample swscale avformat avutil bmjpuapi bmjpulite crypto ssl)

# add_executable(demo_bmnn bmnn_infer.cpp yolov5_common.cpp)
# target_link_libraries(demo_bmnn ${BMNN_Library} ${OpenCV_LIBS})

# tensorrt