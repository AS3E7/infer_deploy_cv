

include_directories(${CMAKE_SOURCE_DIR}/include)

# set(REGISTER_LIBS)
# set(REGISTER_LIBS "123" CACHE STRING "" FORCE)

#--------------------clipper2--------------------------
if (${BUILD_TARGET_CHIP} STREQUAL "ts")
    set(Clipper_Path /usr/local/ts_tx5368/thirdparty/clipper2)
    link_directories("/usr/local/ts_tx5368/libs/")
    link_directories("/usr/local/ts_tx5368/thirdparty/libs/")
    message("!!!!!!!!!!!!!!!gddeploy_register: ${REGISTER_LIBS}")
    set(COMMON_LIBS /usr/local/ts_tx5368/thirdparty/clipper2/lib/libClipper2.a)
else ()
    set(Clipper_Path /usr/local/${TARGET_ARCH}-common)
    set(COMMON_LIBS Clipper2)
endif ()
include_directories(${Clipper_Path}/include)
link_directories(${Clipper_Path}/lib)


# alg
aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/alg/classify/ REGISTER_SRC_FILES)
aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/alg/detect/ REGISTER_SRC_FILES)
aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/alg/pose/ REGISTER_SRC_FILES)
aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/alg/seg/ REGISTER_SRC_FILES)
aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/alg/action/ REGISTER_SRC_FILES)
aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/alg/image_retrieval/ REGISTER_SRC_FILES)
aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/alg/face_retrieval/ REGISTER_SRC_FILES)
aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/alg/ocr_detect/ REGISTER_SRC_FILES)
aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/alg/ocr_retrieval/ REGISTER_SRC_FILES)
aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/alg/action/ REGISTER_SRC_FILES)

# add cpu process code
aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/devices/cpu/preproc REGISTER_SRC_FILES)
aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/devices/cpu/preproc/transform REGISTER_SRC_FILES)
aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/devices/cpu/preproc/action REGISTER_SRC_FILES)
aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/devices/cpu/postproc REGISTER_SRC_FILES)
aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/devices/cpu/postproc/util REGISTER_SRC_FILES)
aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/devices/cpu/ REGISTER_SRC_FILES)
message(${REGISTER_SRC_FILES})
add_library(gddeploy_register SHARED ${REGISTER_SRC_FILES})

target_link_libraries(gddeploy_register PUBLIC gddeploy_common ${OpenCV_LIBS})
target_link_libraries(gddeploy_register PUBLIC ${REGISTER_LIBS} ${COMMON_LIBS} ${CONAN_LIBS})

install(TARGETS gddeploy_register LIBRARY DESTINATION lib)