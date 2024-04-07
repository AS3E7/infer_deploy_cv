link_directories(${CMAKE_BINARY_DIR}/src/core)
link_directories(${CMAKE_BINARY_DIR}/src/register)
link_directories(/usr/local/Ascend/lib64)

if(BUILD_TARGET_CHIP STREQUAL "ts")
    link_directories("/usr/local/ts_tx5368/libs/")
    link_directories("/usr/local/ts_tx5368/thirdparty/libs/")
    include_directories("/usr/local/ts_tx5368/thirdparty/include/")
    # message("##################boost: ")
    #------build sample_runner_pic--------
    link_directories(/usr/local/ts_tx5368/thirdparty/spdlog/lib/)
endif()

add_executable(sample_runner_pic sample_runner_pic.cpp)
if(BUILD_TARGET_CHIP STREQUAL "ts")
    target_link_libraries(sample_runner_pic gddeploy_app gddeploy_api gddeploy_core gddeploy_register spdlog ${Boost_LIBRARIES} boost_atomic mpi drm omx omxc_gpu vpu_jsw vpu_vsw vpu_comm 3a actuator_ms41929 buf_mgr calibration_imx327 cjson cmd_if csi_ar0239 csi_imx327 actuator_ms41929_aux csi_ov5647 frm_mgr isp isp_dpc isp_hal isp_lsh isp_mvd isp_osif isp_top isp_vin json_convert niss pcap sensor sensor_imx327 sensor_imx415 cmdr isp_tpg isp_submod_common sen_modules csi2d_imx327 aecd awbd afcd tuning_imx327_normal aec1d awb1d afc1d ui_uart csi2d_imx415 tuning_imx415_normal calibration_imx415 tuning_imx415_dol2 sensor_os04a10 csi2d_os04a10 tuning_os04a10_normal calibration_os04a10 omxc_gpu_stitch omx GAL)
else()
    target_link_libraries(sample_runner_pic PUBLIC gddeploy_app gddeploy_api gddeploy_core gddeploy_register ${Boost_LIBRARIES})
endif()

install(TARGETS sample_runner_pic LIBRARY DESTINATION bin)
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/sample_runner_pic.cpp DESTINATION sample)

#------build sample_runner_video--------
add_executable(sample_runner_video sample_runner_video.cpp)

if(BUILD_TARGET_CHIP STREQUAL "ts")
    target_link_libraries(sample_runner_video gddeploy_app gddeploy_api gddeploy_core gddeploy_register spdlog ${Boost_LIBRARIES} boost_atomic mpi drm omx omxc_gpu vpu_jsw vpu_vsw vpu_comm 3a actuator_ms41929 buf_mgr calibration_imx327 cjson cmd_if csi_ar0239 csi_imx327 actuator_ms41929_aux csi_ov5647 frm_mgr isp isp_dpc isp_hal isp_lsh isp_mvd isp_osif isp_top isp_vin json_convert niss pcap sensor sensor_imx327 sensor_imx415 cmdr isp_tpg isp_submod_common sen_modules csi2d_imx327 aecd awbd afcd tuning_imx327_normal aec1d awb1d afc1d ui_uart csi2d_imx415 tuning_imx415_normal calibration_imx415 tuning_imx415_dol2 sensor_os04a10 csi2d_os04a10 tuning_os04a10_normal calibration_os04a10 omxc_gpu_stitch omx GAL)
else()
    target_link_libraries(sample_runner_video PRIVATE gddeploy_app ${Boost_LIBRARIES})
endif()


install(TARGETS sample_runner_video LIBRARY DESTINATION bin)
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/sample_runner_video.cpp DESTINATION sample)


# #build sample_procesor_api
# # add_executable(sample_procesor_api sample_processor_api.cpp)
# # target_link_libraries(sample_procesor_api gddeploy_app gddeploy_api gddeploy_core gddeploy_register)

# # install(TARGETS sample_procesor_api LIBRARY DESTINATION bin)
# # install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/sample_procesor_api.cpp DESTINATION sample)

#------build sample_runner_dataset--------
add_executable(sample_runner_dataset sample_runner_dataset.cpp)
if(BUILD_TARGET_CHIP STREQUAL "ts")
    target_link_libraries(sample_runner_dataset gddeploy_app gddeploy_api gddeploy_core gddeploy_register spdlog ${Boost_LIBRARIES} boost_atomic mpi drm omx omxc_gpu vpu_jsw vpu_vsw vpu_comm 3a actuator_ms41929 buf_mgr calibration_imx327 cjson cmd_if csi_ar0239 csi_imx327 actuator_ms41929_aux csi_ov5647 frm_mgr isp isp_dpc isp_hal isp_lsh isp_mvd isp_osif isp_top isp_vin json_convert niss pcap sensor sensor_imx327 sensor_imx415 cmdr isp_tpg isp_submod_common sen_modules csi2d_imx327 aecd awbd afcd tuning_imx327_normal aec1d awb1d afc1d ui_uart csi2d_imx415 tuning_imx415_normal calibration_imx415 tuning_imx415_dol2 sensor_os04a10 csi2d_os04a10 tuning_os04a10_normal calibration_os04a10 omxc_gpu_stitch omx GAL)
else()
    target_link_libraries(sample_runner_dataset PRIVATE gddeploy_app ${Boost_LIBRARIES})
endif()

install(TARGETS sample_runner_dataset LIBRARY DESTINATION bin)
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/sample_runner_dataset.cpp DESTINATION sample)

# install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/CMakeLists.txt DESTINATION sample)
# install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/README.txt DESTINATION docs)

# add_executable(test_only_infer test_only_infer.cpp)
# target_link_libraries(test_only_infer gddeploy_api gddeploy_core gddeploy_register)
# install(TARGETS test_only_infer LIBRARY DESTINATION bin)

# link_directories(${CMAKE_SOURCE_DIR})
# add_executable(test_nvjpeg test_nvjpeg.cpp)
# target_link_libraries(test_nvjpeg PRIVATE gddeploy_app ${Boost_LIBRARIES})
# install(TARGETS test_nvjpeg LIBRARY DESTINATION bin)