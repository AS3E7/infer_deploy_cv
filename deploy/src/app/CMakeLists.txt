include_directories(${FFMPEG_INCLUDE_DIRS})

link_directories(${CMAKE_BINARY_DIR}/src/api)
link_directories(${CMAKE_BINARY_DIR}/src/core)
link_directories(${CMAKE_BINARY_DIR}/src/common)
link_directories(${CMAKE_BINARY_DIR}/src/register)

if(BUILD_TARGET_CHIP STREQUAL "ts")
link_directories("/usr/local/ts_tx5368/libs/")
link_directories("/usr/local/ts_tx5368/thirdparty/libs/")
include_directories("/usr/local/ts_tx5368/thirdparty/include/")
endif()

aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/video APP_SRC_FILES)
aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/pic APP_SRC_FILES)
aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR} APP_SRC_FILES)

add_library(gddeploy_app SHARED ${APP_SRC_FILES})
target_link_libraries(gddeploy_app PUBLIC gddeploy_api gddeploy_core gddeploy_register gddeploy_common ${APP_LIBS} ${CONAN_LIBS} pthread)

install(TARGETS gddeploy_app LIBRARY DESTINATION lib)
install(DIRECTORY ${CMAKE_SOURCE_DIR}/src/app DESTINATION sample)
install(DIRECTORY ${CMAKE_SOURCE_DIR}/include/app DESTINATION include)