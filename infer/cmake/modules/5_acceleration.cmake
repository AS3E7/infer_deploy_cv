if(DEFINED TARGET_CHIP)
    if(${TARGET_CHIP} STREQUAL "nvidia")
        add_compile_definitions(WITH_NVIDIA)
        file(GLOB ModuleFiles
            src/modules/algorithm/nv_*.cpp
            src/modules/wrapper/nv_*.cpp)
        set(LinkLibraries "${LinkLibraries};cudart;nvjpeg")

        if(NOT DEFINED CUDA_TOOLKIT_ROOT_DIR)
            find_package(CUDA 11.4 EXACT REQUIRED)
        endif()

        enable_language(CUDA)
        file(GLOB cuda_files src/modules/cuda/*.cu)
        add_library(gddi_cuda STATIC ${cuda_files})

        if(MSVC)
            link_directories(${CUDA_TOOLKIT_ROOT_DIR}/lib/x64) # windows
            set(LinkLibraries "${LinkLibraries};gddi_tensorrt_sdk;modcrypt;nvml;bcrypt")
            set(SDK_DOWNLOAD_URL "http://cacher.devops.io/api/cacher/files/7869439088bbfe504fa3d38cf2ebf58f43839213c55345b3312fcdfe521c419e")
            set(SDK_URL_HASH "7869439088bbfe504fa3d38cf2ebf58f43839213c55345b3312fcdfe521c419e")
        else()
            include_directories(${CUDA_TOOLKIT_ROOT_DIR}/include)
            link_directories(${CUDA_TOOLKIT_ROOT_DIR}/lib64/stubs)
            set(LinkLibraries "${LinkLibraries};gddi_tensorrt_sdk;nvonnxparser;nvinfer;nvidia-ml;nppicc")
            set(SDK_DOWNLOAD_URL "http://cacher.devops.io/api/cacher/files/71dfbbd96c8c283b364d2bea388dc71224c5be97072b84f05de85df3a0a63c2a")
            set(SDK_URL_HASH "71dfbbd96c8c283b364d2bea388dc71224c5be97072b84f05de85df3a0a63c2a")
        endif()

        include_directories(${CUDA_INCLUDE_DIRS})
        set(LinkLibraries "${LinkLibraries};${CUDA_LIBRARIES};gddi_cuda")
        message(STATUS "Found CUDA: ${CUDA_TOOLKIT_ROOT_DIR} (found version \"${CUDA_VERSION}\")")
    elseif(${TARGET_CHIP} STREQUAL "bm1684") # 比特 bm1684
        add_compile_definitions(WITH_BM1684)
        file(GLOB ModuleFiles src/modules/algorithm/bm_*.cpp)
        set(LinkLibraries "${LinkLibraries};gddeploy_app;gddeploy_api;gddeploy_core;gddeploy_register;gddeploy_common;bmrt")
        set(SDK_DOWNLOAD_URL "http://cacher.devops.io/api/cacher/files/e874f6bf54aed6a2749225b295f4b1db1f9ad75455c13733a8beec24652aa979")
        set(SDK_URL_HASH "e874f6bf54aed6a2749225b295f4b1db1f9ad75455c13733a8beec24652aa979")
    elseif(${TARGET_CHIP} STREQUAL "mlu220") # 寒武纪 mlu220
        add_compile_definitions(WITH_MLU220)
        file(GLOB ModuleFiles src/modules/algorithm/mlu_*.cpp src/modules/wrapper/cn_*.cpp)
        set(LinkLibraries "${LinkLibraries};gddsdk;cncodec;cnrt;ion;jpu;easydk;postproc;glog;cncv")
        set(SDK_DOWNLOAD_URL "http://cacher.devops.io/api/cacher/files/b0eafb19d77736c4299ab834d8b019cd4a5016440c5826ee63776a234591bd7a")
        set(SDK_URL_HASH "b0eafb19d77736c4299ab834d8b019cd4a5016440c5826ee63776a234591bd7a")
    elseif(${TARGET_CHIP} STREQUAL "jetson") # jetson xavier nx
        add_compile_definitions(WITH_JETSON)
        include_directories(${CUDA_INCLUDE_DIRS})
        link_directories(${CUDA_TOOLKIT_ROOT_DIR}/lib/stubs)
        file(GLOB ModuleFiles src/modules/algorithm/jet*.cpp)
        set(LinkLibraries "${LinkLibraries};gddi_tensorrt_sdk;cublas;cublasLt")
        set(SDK_DOWNLOAD_URL "http://cacher.devops.io/api/cacher/files/21ee2aa8012fe98e14b9e924f123c317454bac724081d6269ca832c80b6de718")
        set(SDK_URL_HASH "21ee2aa8012fe98e14b9e924f123c317454bac724081d6269ca832c80b6de718")
    elseif(${TARGET_CHIP} STREQUAL "intel") # 英特尔
        find_package(InferenceEngine REQUIRED)
        message(STATUS "Found InferenceEngine: ${InferenceEngine_CONFIG} (found version \"${InferenceEngine_VERSION}\")")
        add_compile_definitions(WITH_INTEL)
        file(GLOB ModuleFiles src/modules/algorithm/intel_*.cpp)
        set(LinkLibraries "${LinkLibraries};gddsdk;IE::inference_engine")
        set(SDK_DOWNLOAD_URL "http://cacher.devops.io/api/cacher/files/e50b025be5c70356a976ee05e46ef8bb428067d3dc62a6cc96d5927457c219f7")
        set(SDK_URL_HASH "e50b025be5c70356a976ee05e46ef8bb428067d3dc62a6cc96d5927457c219f7")
    elseif(${TARGET_CHIP} STREQUAL "mlu270") # 寒武纪 mlu270
        add_compile_definitions(WITH_MLU270)
        file(GLOB ModuleFiles src/modules/algorithm/mlu_*.cpp)
        set(LinkLibraries "${LinkLibraries};gddsdk;easydk;cncodec;cnrt")
        set(SDK_DOWNLOAD_URL "http://cacher.devops.io/api/cacher/files/ee92137b37db22e68535f92d5bdb1f65a029da65903d73800140f2c6143dd423")
        set(SDK_URL_HASH "ee92137b37db22e68535f92d5bdb1f65a029da65903d73800140f2c6143dd423")
    elseif(${TARGET_CHIP} STREQUAL "rv1126") # 瑞芯微 rv1126
        add_compile_definitions(WITH_RV1126)
        file(GLOB ModuleFiles src/modules/algorithm/rv_*.cpp src/modules/wrapper/mpp_*.cpp src/modules/wrapper/rkmedia_*.cpp)
        set(LinkLibraries "${LinkLibraries};gddi_rockchip_sdk;rknn_api;easymedia;rga;rockchip_mpp")
        set(SDK_DOWNLOAD_URL "http://cacher.devops.io/api/cacher/files/7f3c1047086ac0fec83de3cc2c527afcd70af6cd0f742b62e021437f4ae43bf7")
        set(SDK_URL_HASH "7f3c1047086ac0fec83de3cc2c527afcd70af6cd0f742b62e021437f4ae43bf7")
    elseif(${TARGET_CHIP} STREQUAL "mlu370") # 寒武纪 mlu370
        add_compile_definitions(WITH_MLU370)
        file(GLOB ModuleFiles src/modules/algorithm/mlu370_*.cpp)
        set(LinkLibraries "${LinkLibraries};gddsdk;easydk;cncodec_v3;cnrt")
        set(SDK_DOWNLOAD_URL "http://cacher.devops.io/api/cacher/files/45aff1278e55d9d59037770fb6e6614db626c29959ee1c81251f46ff3cd707a9")
        set(SDK_URL_HASH "45aff1278e55d9d59037770fb6e6614db626c29959ee1c81251f46ff3cd707a9")
    elseif(${TARGET_CHIP} STREQUAL "tx5368") # 清微 tx5368
        add_compile_definitions(WITH_TX5368)
        file(GLOB ModuleFiles src/modules/algorithm/tsing_*.cpp)
        set(LinkLibraries "${LinkLibraries};gddeploy_app;gddeploy_api;gddeploy_core;gddeploy_register;gddeploy_common;mpi")
        set(SDK_DOWNLOAD_URL "http://cacher.devops.io/api/cacher/files/86709875d83868e48ba415719d4b17d9cc1f15cf537b8f229a67432967ddad0e")
        set(SDK_URL_HASH "86709875d83868e48ba415719d4b17d9cc1f15cf537b8f229a67432967ddad0e")
    else()
        message(FATAL_ERROR "Unsupported target chip: ${TARGET_CHIP}")
    endif()

    include(FetchContent)
    FetchContent_Declare(
        inference-sdk
        URL ${SDK_DOWNLOAD_URL}
        URL_HASH SHA256=${SDK_URL_HASH}
    )

    FetchContent_MakeAvailable(inference-sdk)
    include_directories(${inference-sdk_SOURCE_DIR}/include)
    link_directories(${inference-sdk_SOURCE_DIR}/lib)

    file(GLOB NodeFiles src/nodes/algorithm/*.cpp
                        src/modules/wrapper/draw_image.cpp
                        src/modules/wrapper/export_video.cpp)
    set(LibFiles "${LibFiles};${ModuleFiles};${NodeFiles};${CuFiles}")
else()
    message(STATUS "Acc disabled, does not compile algorithm modules")
endif()
