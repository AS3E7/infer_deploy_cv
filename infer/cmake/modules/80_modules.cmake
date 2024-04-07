file(GLOB ModuleLibFiles
    src/modules/bytetrack/*.c??
    src/modules/postprocess/*.c??)

if(${TARGET_CHIP} STREQUAL "tx5368")
    list(APPEND ModuleLibFiles "src/modules/wrapper/tsing_jpeg_encode.cpp;src/modules/wrapper/tscv_operator.cpp")
endif()

message(${ModuleLibFiles})

add_library(gddi_post SHARED ${ModuleLibFiles})
set_target_properties(gddi_post PROPERTIES
    PUBLIC_HEADER "src/modules/types.hpp;src/modules/bytetrack/target_tracker.h;src/modules/postprocess/cross_border.h;")
set_target_properties(gddi_post PROPERTIES VERSION ${PROJECT_VERSION} SOVERSION ${PROJECT_VERSION_MAJOR})
if(${TARGET_CHIP} STREQUAL "nvidia")
    target_link_libraries(gddi_post PRIVATE ${OpenCV_LIBRARIES} Eigen3::Eigen ${FFMPEG_LIBRARIES} cudart nppicc spdlog::spdlog)
elseif(${TARGET_CHIP} STREQUAL "tx5368")
    target_link_libraries(gddi_post PRIVATE ${OpenCV_LIBRARIES} Eigen3::Eigen spdlog::spdlog tscv mpi)
else()
    target_link_libraries(gddi_post PRIVATE ${OpenCV_LIBRARIES} Eigen3::Eigen spdlog::spdlog)
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Release")
    set(CMAKE_INSTALL_PREFIX "${CMAKE_SOURCE_DIR}/release/${CMAKE_SYSTEM_PROCESSOR}/${TARGET_CHIP}/${CHIP_NAME}")
    install(TARGETS gddi_post
        ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
        PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_PREFIX}/include)
endif()

# add_library(ai_solution SHARED src/modules/dahuatech/ai_solution.cpp)
# set_target_properties(ai_solution PROPERTIES VERSION ${PROJECT_VERSION} SOVERSION ${PROJECT_VERSION_MAJOR})
# target_link_libraries(ai_solution PRIVATE)

set(LinkLibraries "${LinkLibraries};gddi_post")