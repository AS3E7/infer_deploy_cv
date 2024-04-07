FROM nvidia/cuda:11.4.0-cudnn8-devel-ubuntu20.04

RUN sed -i 's/archive.ubuntu.com/mirrors.ustc.edu.cn/g' /etc/apt/sources.list

ENV DEBIAN_FRONTEND=noninteractive

RUN apt update && apt install -y git pkgconf ninja-build libssl-dev wget

# CMake
RUN cd /tmp && \
    wget https://github.com/Kitware/CMake/releases/download/v3.24.0/cmake-3.24.0-linux-x86_64.sh && \
    chmod +x cmake-3.24.0-linux-x86_64.sh && \
    ./cmake-3.24.0-linux-x86_64.sh --prefix=/usr/local --exclude-subdir --skip-license

# Nvidia 编译环境
RUN cd /tmp && \
    git clone -b n10.0.26.2 --depth=1 http://git.mirror.gddi.io/mirror/nv-codec-headers && \
    cd nv-codec-headers && \
    make && make install
RUN apt install -y nasm libx264-dev libx265-dev libvpx-dev
RUN cd /tmp && \
    git clone -b n4.4.2 --depth=1 http://git.mirror.gddi.io/mirror/FFmpeg.git && cd FFmpeg && \
    ./configure --enable-shared --disable-debug  --disable-doc --disable-ffplay --enable-openssl \
        --extra-cflags="-I/usr/local/cuda-11.4/include" --extra-ldflags="-L/usr/local/cuda-11.4/lib64" \
        --extra-libs="-lpthread -lm"  --enable-gpl --enable-libx264 --enable-libx265 \
        --enable-cuda --enable-cuvid --enable-nvenc --enable-nonfree --enable-libnpp && \
    make -j && make install
RUN apt update && \
    apt install -y libjpeg-dev libpng-dev libtiff-dev libopencore-amrnb-dev libopencore-amrwb-dev \
        libtbb-dev libatlas-base-dev
RUN cd /tmp && \
    git clone -b 4.5.5 --depth=1 http://git.mirror.gddi.io/mirror/opencv.git && \
    git clone -b 4.5.5 --depth=1 http://git.mirror.gddi.io/mirror/opencv_contrib.git && cd opencv && \
    cmake -S . -Bbuild -G Ninja \
        -DCMAKE_BUILD_TYPE=Release -DWITH_TBB=ON -DENABLE_FAST_MATH=1 -DCUDA_FAST_MATH=1 \
        -DWITH_CUBLAS=1 -DWITH_CUDA=ON -DBUILD_opencv_cudacodec=OFF -DWITH_CUDNN=ON -DOPENCV_DNN_CUDA=ON \
        -DCUDA_ARCH_BIN=7.5 -DWITH_QT=OFF -DWITH_OPENGL=ON -DWITH_GSTREAMER=OFF -DWITH_GTK=OFF \
        -DOPENCV_EXTRA_MODULES_PATH=/tmp/opencv_contrib/modules -DINSTALL_PYTHON_EXAMPLES=OFF \
        -DINSTALL_C_EXAMPLES=OFF -DBUILD_EXAMPLES=OFF -DWITH_FFMPEG=OFF && \
    cmake --build build/ --target install
RUN apt install -y libnvinfer8=8.2.5-1+cuda11.4 libnvonnxparsers8=8.2.5-1+cuda11.4 \
        libnvparsers8=8.2.5-1+cuda11.4 libnvinfer-plugin8=8.2.5-1+cuda11.4 \
        libnvinfer-dev=8.2.5-1+cuda11.4 libnvonnxparsers-dev=8.2.5-1+cuda11.4 \
        libnvparsers-dev=8.2.5-1+cuda11.4 libnvinfer-plugin-dev=8.2.5-1+cuda11.4 && \
    apt-mark hold libnvinfer8 libnvonnxparsers8 libnvparsers8 libnvinfer-plugin8 \
        libnvinfer-dev libnvonnxparsers-dev libnvparsers-dev libnvinfer-plugin-dev

ENV TRT_LIBPATH=/usr/lib/x86_64-linux-gnu
# COPY docker/base-devel/yuv2bgr.h /usr/local/include/
# COPY docker/base-devel/yuv2bgr.cu /tmp/
# RUN cd /tmp && \
#     nvcc --ptxas-options=-v --compiler-options '-fPIC' --shared -o libyuv2bgr.so yuv2bgr.cu && \
#     mv libyuv2bgr.so /usr/local/lib/

# # Armv8 交叉编译链
# RUN apt update && apt install -y gcc-aarch64-linux-gnu g++-aarch64-linux-gnu gdb-multiarch

# # Jetson 交叉编译环境
# RUN cd /tmp && \
#     wget --content-disposition \
#         http://cacher.devops.io/api/cacher/files/2964ee1da3f9949939b18bcb49396a5ff56d480c76a8df617f6417c4981d7e30 && \
#     tar -zxvf jetpack_files.tar.gz
# RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-key F60F4B3D7FA2AF80
# RUN dpkg -i /tmp/jetpack_files/cuda-repo-cross-aarch64*.deb /tmp/jetpack_files/cuda-repo-ubuntu*_amd64.deb && \
#     apt-get update && \
#     apt-get install -y cuda-cross-aarch64
# RUN mkdir -p /usr/local/aarch64-jetson-soc/include && mkdir -p /usr/local/aarch64-jetson-soc/lib
# RUN dpkg -x /tmp/jetpack_files/libcudnn[7-8]_*-1+cuda10.[0-9]_arm64.deb /tmp/cudnn && \
#     dpkg -x /tmp/jetpack_files/libcudnn[7-8]-dev_*-1+cuda10.[0-9]_arm64.deb /tmp/cudnn && \
#     mv /tmp/cudnn/usr/include/aarch64-linux-gnu/* /usr/local/aarch64-jetson-soc/include && \
#     mv /tmp/cudnn/usr/lib/aarch64-linux-gnu/* /usr/local/aarch64-jetson-soc/lib
# RUN dpkg -x /tmp/jetpack_files/libnvinfer[0-8]_*-1+cuda10.[0-9]_arm64.deb /tmp/tensorrt && \
#     dpkg -x /tmp/jetpack_files/libnvinfer-dev_*-1+cuda10.[0-9]_arm64.deb /tmp/tensorrt && \
#     dpkg -x /tmp/jetpack_files/libnvparsers[6-8]_*-1+cuda10.[0-9]_arm64.deb /tmp/tensorrt && \
#     dpkg -x /tmp/jetpack_files/libnvparsers-dev_*-1+cuda10.[0-9]_arm64.deb /tmp/tensorrt && \
#     dpkg -x /tmp/jetpack_files/libnvinfer-plugin[6-8]_*-1+cuda10.[0-9]_arm64.deb /tmp/tensorrt && \
#     dpkg -x /tmp/jetpack_files/libnvinfer-plugin-dev_*-1+cuda10.[0-9]_arm64.deb /tmp/tensorrt && \
#     dpkg -x /tmp/jetpack_files/libnvonnxparsers[6-8]_*-1+cuda10.[0-9]_arm64.deb /tmp/tensorrt && \
#     dpkg -x /tmp/jetpack_files/libnvonnxparsers-dev_*-1+cuda10.[0-9]_arm64.deb /tmp/tensorrt && \
#     mv /tmp/tensorrt/usr/include/aarch64-linux-gnu/* /usr/local/aarch64-jetson-soc/include && \
#     mv /tmp/tensorrt/usr/lib/aarch64-linux-gnu/* /usr/local/aarch64-jetson-soc/lib
# RUN dpkg -x /tmp/jetpack_files/OpenCV-4.1.1-2-gd5a58aa75-aarch64-dev.deb /tmp/opencv && \
#     dpkg -x /tmp/jetpack_files/OpenCV-4.1.1-2-gd5a58aa75-aarch64-libs.deb /tmp/opencv && \
#     mv /tmp/opencv/usr/include/* /usr/local/aarch64-jetson-soc/include && \
#     mv /tmp/opencv/usr/lib/aarch64-linux-gnu/* /usr/local/aarch64-jetson-soc/lib
# RUN sed -i '55d' /usr/local/aarch64-jetson-soc/lib/cmake/opencv4/OpenCVModules.cmake
# RUN sed -i 's/aarch64-linux-gnu\///g' /usr/local/aarch64-jetson-soc/lib/cmake/opencv4/OpenCVModules-release.cmake
# RUN tar -zxvf /tmp/jetpack_files/ffmpeg.tar.gz -C /usr/local/aarch64-jetson-soc

# # BM1684 交叉编译环境
# RUN apt install unzip
# RUN mkdir -p /usr/local/aarch64-bm1684-soc/include && mkdir -p /usr/local/aarch64-bm1684-soc/lib
# RUN cd /tmp && \
#     wget --content-disposition \
#         http://cacher.devops.io/api/cacher/files/4dab8ddafc5d14f90f4c212c7886d5defb704034cb43e80ccb49ef73dfde3b06 && \
#     unzip bmnnsdk2_bm1684_v2.6.0.zip && cd bmnnsdk2_bm1684_v2.6.0 && \
#     tar -zxvf bmnnsdk2-bm1684_v2.6.0.tar.gz && \
#     cp -r bmnnsdk2-bm1684_v2.6.0/include/ffmpeg/* /usr/local/aarch64-bm1684-soc/include && \
#     cp -r bmnnsdk2-bm1684_v2.6.0/lib/ffmpeg/soc/* /usr/local/aarch64-bm1684-soc/lib && \
#     sed -i 's/\/usr\/local/\/usr\/local\/aarch64-bm1684-soc/g' /usr/local/aarch64-bm1684-soc/lib/pkgconfig/* && \
#     cp -r bmnnsdk2-bm1684_v2.6.0/include/opencv/* /usr/local/aarch64-bm1684-soc/include && \
#     cp -r bmnnsdk2-bm1684_v2.6.0/lib/opencv/soc/* /usr/local/aarch64-bm1684-soc/lib && \
#     cp bmnnsdk2-bm1684_v2.6.0/include/bmruntime/* /usr/local/aarch64-bm1684-soc/include && \
#     cp bmnnsdk2-bm1684_v2.6.0/include/bmlib/* /usr/local/aarch64-bm1684-soc/include && \
#     cp bmnnsdk2-bm1684_v2.6.0/lib/bmnn/soc/*.so /usr/local/aarch64-bm1684-soc/lib && \
#     cp bmnnsdk2-bm1684_v2.6.0/lib/decode/soc/*.so /usr/local/aarch64-bm1684-soc/lib

# # Conan 包管理工具
# RUN apt install -y python3 python3-pip && \
#     python3 -m pip install --upgrade pip && \
#     pip3 config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
# RUN cd /tmp && \
#     wget https://github.com/conan-io/conan/releases/latest/download/conan-ubuntu-64.deb && \
#     dpkg -i conan-ubuntu-64.deb && \
#     pip3 install conan

# # build-essential
# RUN apt install -y ccache sudo

# RUN apt clean autoclean && apt autoremove --yes && rm -rf /var/lib/{apt,dpkg,cache,log}/
# RUN rm -r /tmp/*

# ENV PATH=/usr/local/cuda-11.4/bin:/usr/local/bin:$PATH
# ENV LD_LIBRARY_PATH=/usr/local/cuda-11.4/lib64:/usr/local/lib:$LD_LIBRARY_PATH
# ENV C_INCLUDE_PATH=/usr/local/include:$C_INCLUDE_PATH
# ENV CPLUS_INCLUDE_PATH=/usr/local/include:$CPLUS_INCLUDE_PATH

# ARG uid=1000
# ARG gid=1000
# RUN groupadd -r -f -g ${gid} gddi && useradd -o -r -l -u ${uid} -g ${gid} -ms /bin/bash gddi
# RUN usermod -aG sudo gddi
# RUN echo 'gddi:gddi1234' | chpasswd
# RUN mkdir -p /workspace && chown gddi /workspace

# USER gddi

# RUN conan remote add gddi-conan-local http://devops.io:8081/artifactory/api/conan/gddi-conan-local && \
#     conan user -p Gddi@1234 -r gddi-conan-local gddi

WORKDIR /workspace

RUN ["/bin/bash"]

# docker build . -f docker/base-devel/Dockerfile -t hub.gddi.com/devel/inference-engine-devel:1.0.0