
if(BUILD_TARGET_CHIP STREQUAL "ts")
    # 设置编辑的头文件引用路径
    include_directories("/usr/local/ts_tx5368/thirdparty/include/")
    include_directories("/usr/local/ts_tx5368/include")

    if (TARGET_ARCH STREQUAL "aarch64")
        link_directories("/usr/local/ts_tx5368/libs")
    else()
        link_directories("/usr/local/ts_tx5368/libs")
    endif()
    link_directories("/usr/local/ts_tx5368/libs/")
    # link_directories("/usr/local/ts_tx5368/thirdparty/libs/")
    

    add_compile_definitions(WITH_TS)
    # /usr/local/ts_tx5368/libs/64bit/librne_rt_g3.a /usr/local/ts_tx5368/libs/64bit/librne_pal_linux_a53.a
    set(REGISTER_LIBS "${REGISTER_LIBS};rne_rt_g3.a;rne_pal_linux_a53.a;mpi;drm;omx;omxc_gpu;vpu_vsw;vpu_comm;3a;actuator_ms41929;buf_mgr;calibration_imx327;cjson;cmd_if;csi_ar0239;csi_imx327;actuator_ms41929_aux;csi_ov5647;frm_mgr;isp;isp_dpc;isp_hal;isp_lsh;isp_mvd;isp_osif;isp_top;isp_vin;json_convert;niss;pcap;sensor;sensor_imx327;sensor_imx415;cmdr;isp_tpg;isp_submod_common;sen_modules;csi2d_imx327;aecd;awbd;afcd;tuning_imx327_normal;aec1d;awb1d;afc1d;ui_uart;csi2d_imx415;tuning_imx415_normal;calibration_imx415;tuning_imx415_dol2;sensor_os04a10;csi2d_os04a10;tuning_os04a10_normal;calibration_os04a10;omxc_gpu_stitch;omx;GAL")

    file(GLOB REGISTER_SRC_FILES ${CMAKE_SOURCE_DIR}/src/register/devices/ts/*.cpp)
    install(DIRECTORY /usr/local/ts_tx5368/libs/ DESTINATION thirdparty/lib)

endif()