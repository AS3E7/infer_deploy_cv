aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR} COMMON_SRC_FILES)

add_library(gddeploy_common SHARED ${COMMON_SRC_FILES})
target_link_libraries(gddeploy_common spdlog ${FFMPEG_LIBRARIES})

install(TARGETS gddeploy_common LIBRARY DESTINATION lib)
install(DIRECTORY ${CMAKE_SOURCE_DIR}/src/common DESTINATION sample)
install(DIRECTORY ${CMAKE_SOURCE_DIR}/include/common DESTINATION sample)