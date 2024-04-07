set(3rdpart_path ${CMAKE_SOURCE_DIR}/thirdparty)
link_directories(${3rdpart_path}/libmodcry_${TARGET_ARCH}/lib)
include_directories(${3rdpart_path}/libmodcry_${TARGET_ARCH}/include)

include_directories(${CMAKE_SOURCE_DIR}/src)

aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR} CORE_SRC_FILES)
aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/util CORE_SRC_FILES)
aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/model CORE_SRC_FILES)
aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/processor CORE_SRC_FILES)
aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/mem CORE_SRC_FILES)
aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/alg CORE_SRC_FILES)

include_directories(${CMAKE_SOURCE_DIR}/src/register/devices)
include_directories(${CMAKE_SOURCE_DIR}/src/register/alg)
# aux_source_directory(${CMAKE_SOURCE_DIR}/src/register/devices/processor/preprocess/cpu REGISTER_SRC_FILES)

# link_directories(/usr/local/ts_tx5368/thirdparty/spdlog/lib/)
add_library(gddeploy_core SHARED ${CORE_SRC_FILES}) # ${REGISTER_SRC_FILES})
target_link_libraries(gddeploy_core PUBLIC  gddeploy_common gddeploy_register spdlog)
if (TARGET_ARCH STREQUAL "x86")
    target_link_libraries(gddeploy_core PRIVATE  libmodcrypt.a libssl.a libcrypto.a ${CONAN_LIBS} pthread dl) #hv_static
else()
    target_link_libraries(gddeploy_core PRIVATE  libmodcrypt.a hv_static libssl.a libcrypto.a ${CONAN_LIBS} pthread dl) #hv_static
endif()

install(TARGETS gddeploy_core LIBRARY DESTINATION lib)