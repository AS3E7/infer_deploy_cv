#!/bin/bash

#source $PWD/install/*.sh

# BUILD_TARGET = ()
# 解析需要编译哪个硬件平台
if [ $# -eq 0 ];then    #mean build all
    BUILD_TARGET=("onnx" "bmnn")
else
    BUILD_TARGET=$*
fi

echo "param num:$#"
echo "Build target: ${BUILD_TARGET[*]}"

function install_convert() {
    echo "===========Install convert begin============="

    source /opt/install/install_common.sh
    install_common_build
    if [ $? -ne 0 ] ; then
        echo "install cmmon FAILED!"
        exit 1
    fi

    for i in ${BUILD_TARGET[*]};do
        echo "Build target:$i"
        if [ $i = "opencv" ];then
            echo "build opencv simple"
            bash /opt/install/opencv.sh
        # elif [ $i = "onnx" ];then
        #     echo "build onnx"
        #     source /opt/install/install_onnx.sh

        #     install_onnx_build
        #     if [ $? -ne 0 ] ; then
        #         echo "install onnx FAILED!"
        #         exit 1
        #     fi
        # elif [ $i = "bmnn" ];then
        #     echo "build bmnn"
        #     source /opt/install/install_bmnn.sh

        #     install_bmnn_build
        #     if [ $? -ne 0 ] ; then
        #         echo "install bmnn FAILED!"
        #         exit 1
        #     fi
        # elif [ $i = "mnn" ];then
        #     echo "build bmnn"
        #     source /opt/install/install_mnn.sh
        #     install_mnn_build x86
        #     if [ $? -ne 0 ] ; then
        #         echo "install mnn FAILED!"
        #         exit 1
        #     fi
        elif [ $i = "openvino" ];then
            echo "build bmnn"
            source /opt/install/install_openvino.sh

            install_openvino_convert
            if [ $? -ne 0 ] ; then
                echo "install openvino FAILED!"
                exit 1
            fi
        fi

    done

    echo "===========Install convert end==============="
}

install_convert