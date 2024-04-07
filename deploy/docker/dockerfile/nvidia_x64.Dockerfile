FROM nvidia/cuda:11.4.3-cudnn8-devel-ubuntu20.04

RUN sed -i 's/archive.ubuntu.com/mirrors.ustc.edu.cn/g' /etc/apt/sources.list

ENV DEBIAN_FRONTEND=noninteractive

RUN apt update && apt install -y git pkgconf ninja-build libssl-dev wget vim nfs-kernel-server nfs-common

# Install prerequisites
RUN apt update && apt install -y gcc-7 g++-7 git pkgconf ninja-build libssl-dev wget && \
    update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-7 100 --slave /usr/bin/g++ g++ /usr/bin/g++-7

# CMake
RUN cd /tmp && \
    wget https://github.com/Kitware/CMake/releases/download/v3.24.0/cmake-3.24.0-linux-x86_64.sh && \
    chmod +x cmake-3.24.0-linux-x86_64.sh && \
    ./cmake-3.24.0-linux-x86_64.sh --prefix=/usr/local --exclude-subdir --skip-license

# Boost
RUN cd /tmp && \
    wget https://boostorg.jfrog.io/artifactory/main/release/1.81.0/source/boost_1_81_0.tar.bz2 && \
    tar -xf boost_1_81_0.tar.bz2 && \
    cd boost_1_81_0 && \
    ./bootstrap.sh && \
    ./b2 install --prefix=/usr/local/x86-common/boost_1.81_gcc7.5

# Clipper2
RUN cd /tmp && \
    git clone --recurse-submodules https://github.com/AngusJohnson/Clipper2.git && \
    cd Clipper2/ && \
    git checkout 49b97f74d063b9f5487475689fca669b23117408
RUN cd /tmp/Clipper2/CPP && \
    cmake -S . -Bbuild -DCMAKE_INSTALL_PREFIX=/usr/local/x86-common/Clipper2 && \
    cmake --build build/ --target install

# spdlog
RUN cd /tmp && \
    git clone https://github.com/gabime/spdlog.git && \
    cd spdlog && \
    git checkout v1.8.5 && \
    cmake -S . -Bbuild -DCMAKE_INSTALL_PREFIX=/usr/local/x86-common/spdlog_1.8.5_gcc7.5 && \
    cmake --build build/ --target install

# Nvidia 编译环境
RUN cd /tmp && \
    git clone -b n10.0.26.2 --depth=1 http://git.mirror.gddi.io/mirror/nv-codec-headers && \
    cd nv-codec-headers && \
    make && make install
RUN apt install -y nasm libx264-dev libx265-dev libvpx-dev 
RUN cd /tmp && \
    git clone -b n4.4.2 --depth=1 http://git.mirror.gddi.io/mirror/FFmpeg.git && cd FFmpeg && \
    ./configure --enable-shared --disable-debug  --disable-doc --disable-ffplay --enable-openssl \
        --prefix=/usr/local/ffmpeg_x86_cuda \
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
        -DCMAKE_INSTALL_PREFIX=/usr/local/opencv_x86_cuda \
        -DCUDA_ARCH_BIN=7.5 -DWITH_QT=OFF -DWITH_OPENGL=ON -DWITH_GSTREAMER=OFF -DWITH_GTK=OFF \
        -DOPENCV_EXTRA_MODULES_PATH=/tmp/opencv_contrib/modules -DINSTALL_PYTHON_EXAMPLES=OFF \
        -DINSTALL_C_EXAMPLES=OFF -DBUILD_EXAMPLES=OFF -DWITH_FFMPEG=OFF && \
    cmake --build build/ --target install
RUN apt install -y libnvinfer8=8.2.5-1+cuda11.4 libnvonnxparsers8=8.2.5-1+cuda11.4 \
        libnvparsers8=8.2.5-1+cuda11.4 libnvinfer-plugin8=8.2.5-1+cuda11.4 \
        libnvinfer-dev=8.2.5-1+cuda11.4 libnvonnxparsers-dev=8.2.5-1+cuda11.4 \
        libnvparsers-dev=8.2.5-1+cuda11.4 libnvinfer-plugin-dev=8.2.5-1+cuda11.4

ENV TRT_LIBPATH=/usr/lib/x86_64-linux-gnu

WORKDIR /workspace

RUN ["/bin/bash"]
