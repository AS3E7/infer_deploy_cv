set(3rdpart_path ${CMAKE_SOURCE_DIR}/thirdparty)
link_directories(${3rdpart_path}/libmodcry_${TARGET_ARCH}/lib)
include_directories(${3rdpart_path}/libmodcry_${TARGET_ARCH}/include)
include_directories(${CMAKE_SOURCE_DIR}/include)

add_executable(gxt_maker gtx_maker.cpp)
target_link_libraries(gxt_maker PUBLIC gddeploy_api gddeploy_core gddeploy_common gddeploy_register ${3rdpart_path}/libmodcry_${TARGET_ARCH}/lib/libmodcrypt.a ${3rdpart_path}/libmodcry_${TARGET_ARCH}/lib/libssl.a ${3rdpart_path}/libmodcry_${TARGET_ARCH}/lib/libcrypto.a dl pthread)
install(TARGETS gxt_maker RUNTIME DESTINATION tools)