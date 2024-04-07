# file(GLOB_RECURSE SRC_FILES "api/*.c??" "app/*.c??" "common/*.c??" "core/*.c??" "model/*.c??" "tracker/*.c??")

# message(${SRC_FILES})

# aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/src/core SRC_CORE)
# # message(${SRC_CORE})

# add_executable(test_core ${SRC_FILES} ${SRC_CORE})
# target_link_libraries(test_core hwsdk)

# include_directories(${CMAKE_SOURCE_DIR}/src)

# # test model manager
# set(3rdpart_path ${CMAKE_SOURCE_DIR}/thirdparty)
# link_directories(${3rdpart_path}/libmodcry_${TARGET_ARCH}/lib)
# include_directories(${3rdpart_path}/libmodcry_${TARGET_ARCH}/include)

# # add_executable(test_core_model_manager ${CMAKE_CURRENT_SOURCE_DIR}/core/test_model_manager.cpp)
# # target_link_libraries(test_core_model_manager gddeploy_core)
# # install(TARGETS test_core_model_manager LIBRARY DESTINATION bin)

# # test preprocess
# # aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/processor CORE_SRC_FILES)

# add_executable(test_processor_preproc ${CMAKE_CURRENT_SOURCE_DIR}/processor/test_preprocess.cpp)
# target_link_libraries(test_processor_preproc gddeploy_core gddeploy_register)

# add_subdirectory(demo)

# add_executable(test_processor_api ${CMAKE_CURRENT_SOURCE_DIR}/processor/test_preprocess.cpp)
# target_link_libraries(test_processor_preproc gddeploy_core gddeploy_register)

# include_directories("/usr/local/ts_tx5368/include")
# link_directories("/usr/local/ts_tx5368/libs/64bit")
# add_executable(sample_ts sample_ts.cpp)
# target_link_libraries(sample_ts PRIVATE chip_sdk;plat_rne_linux_a53 ${OpenCV_LIBS})

# install(TARGETS sample_ts LIBRARY DESTINATION bin)

# link_directories(${CMAKE_BINARY_DIR}/src/core)
# link_directories(${CMAKE_BINARY_DIR}/src/common)
# link_directories(${CMAKE_BINARY_DIR}/src/register)
# message(${CMAKE_BINARY_DIR}/src/register)

# include_directories(${CMAKE_SOURCE_DIR})
# add_executable(test_nvidia_nvjpeg ${CMAKE_CURRENT_SOURCE_DIR}/register/nvidia/test_nvjpeg.cpp)
# target_link_libraries(test_nvidia_nvjpeg PRIVATE gddeploy_core gddeploy_common gddeploy_register gddeploy_core)