FROM nvidia/cuda:11.4.0-cudnn8-devel-ubuntu20.04

RUN sed -i 's/archive.ubuntu.com/mirrors.ustc.edu.cn/g' /etc/apt/sources.list

ENV DEBIAN_FRONTEND=noninteractive

RUN apt update && \
    apt install -y git pkgconf ninja-build ccache libssl-dev wget && \
    rm -rf /var/lib/apt/lists/*

# CMake
RUN cd /tmp && \
    wget https://github.com/Kitware/CMake/releases/download/v3.24.0/cmake-3.24.0-linux-x86_64.sh && \
    chmod +x cmake-3.24.0-linux-x86_64.sh && \
    ./cmake-3.24.0-linux-x86_64.sh --prefix=/usr/local --exclude-subdir --skip-license && \
    rm -rf /tmp/*

# Nvidia 编译环境
RUN cd /tmp && \
    git clone -b n10.0.26.2 --depth=1 http://git.mirror.gddi.io/mirror/nv-codec-headers && \
    cd nv-codec-headers && \
    make && make install && \
    rm -rf /tmp/*
RUN apt update && \
    apt install -y nasm libvpx-dev && \
    rm -rf /var/lib/apt/lists/*
RUN cd /tmp && \
    git clone -b n4.4.2 --depth=1 http://git.mirror.gddi.io/mirror/FFmpeg.git && cd FFmpeg && \
    ./configure \
        --prefix=/usr/local/x86_64-nvidia-gnu \
        --disable-debug \
        --disable-programs \
        --enable-shared \
        --enable-openssl \
        --enable-cuda \
        --enable-cuvid \
        --enable-nvenc \
        --enable-libnpp \
        --enable-nonfree \
        --enable-optimizations \
        --extra-libs="-lpthread -lm" \
        --extra-cflags="-I/usr/local/cuda-11.4/include" \
        --extra-ldflags="-L/usr/local/cuda-11.4/lib64" && \
    make -j && make install && \
    rm -rf /tmp/*
RUN apt update && \
    apt install -y libjpeg-dev libpng-dev libtiff-dev libopencore-amrnb-dev libopencore-amrwb-dev \
        libtbb-dev libatlas-base-dev && \
    rm -rf /var/lib/apt/lists/*
RUN cd /tmp && \
    git clone -b 4.5.5 --depth=1 http://git.mirror.gddi.io/mirror/opencv_contrib.git && \
    git clone -b 4.5.5 --depth=1 http://git.mirror.gddi.io/mirror/opencv.git && cd opencv && \
    cmake -S . -Bbuild -G Ninja -DCMAKE_INSTALL_PREFIX=/usr/local/x86_64-nvidia-gnu \
        -DCMAKE_BUILD_TYPE=Release -DWITH_TBB=ON -DENABLE_FAST_MATH=1 -DCUDA_FAST_MATH=1 \
        -DWITH_CUBLAS=1 -DWITH_CUDA=ON -DBUILD_opencv_cudacodec=OFF -DWITH_CUDNN=ON -DOPENCV_DNN_CUDA=ON \
        -DCUDA_ARCH_BIN=7.5 -DWITH_QT=OFF -DWITH_OPENGL=ON -DWITH_GSTREAMER=OFF -DWITH_GTK=OFF \
        -DOPENCV_EXTRA_MODULES_PATH=/tmp/opencv_contrib/modules -DINSTALL_PYTHON_EXAMPLES=OFF \
        -DINSTALL_C_EXAMPLES=OFF -DBUILD_EXAMPLES=OFF -DWITH_FFMPEG=OFF && \
    cmake --build build/ --target install && \
    rm -rf /tmp/*
RUN apt update && \
    apt install -y libnvinfer8=8.2.5-1+cuda11.4 libnvonnxparsers8=8.2.5-1+cuda11.4 \
        libnvparsers8=8.2.5-1+cuda11.4 libnvinfer-plugin8=8.2.5-1+cuda11.4 \
        libnvinfer-dev=8.2.5-1+cuda11.4 libnvonnxparsers-dev=8.2.5-1+cuda11.4 \
        libnvparsers-dev=8.2.5-1+cuda11.4 libnvinfer-plugin-dev=8.2.5-1+cuda11.4 && \
    apt-mark hold libnvinfer8 libnvonnxparsers8 libnvparsers8 libnvinfer-plugin8 \
        libnvinfer-dev libnvonnxparsers-dev libnvparsers-dev libnvinfer-plugin-dev && \
    rm -rf /var/lib/apt/lists/*

# Intel 编译环境
RUN cd /tmp && \
    wget https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB && \
    apt-key add GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB && \
    rm -rf /tmp/*
RUN echo "deb https://apt.repos.intel.com/openvino/2022 bionic main" | tee /etc/apt/sources.list.d/intel-openvino-2022.list && \
    apt update && \
    apt install -y openvino-2022.1.0 openvino-opencv && \
    rm -rf /var/lib/apt/lists/*
RUN apt update && \
    apt install -y libdrm-dev autoconf libtool pkg-config && \
    rm -rf /var/lib/apt/lists/*
RUN cd /tmp && \
    git clone -b 2.15.0 --depth=1 https://github.com/intel/libva.git && cd libva && \
    ./autogen.sh && \
    ./configure --prefix=/usr/local/x86_64-intel-gnu && \
    make -j12 && make install && \
    rm -rf /tmp/*
RUN cd /tmp && \
    git clone -b intel-gmmlib-22.1.7 --depth=1 https://github.com/intel/gmmlib.git && cd gmmlib && \
    cmake -S . -Bbuild -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr/local/x86_64-intel-gnu && \
    cmake --build build/ --target install && \
    rm -rf /tmp/*
RUN cd /tmp && \
    git clone -b intel-media-22.5.2 --depth=1 https://github.com/intel/media-driver.git && cd media-driver && \
    cmake -S . -Bbuild -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr/local/x86_64-intel-gnu \
        -DPKG_CONFIG_USE_CMAKE_PREFIX_PATH=ON -DCMAKE_PREFIX_PATH=/usr/local/x86_64-intel-gnu && \
    cmake --build build/ --target install && \
    rm -rf /tmp/*
RUN cd /tmp && \
    git clone -b n4.4.2 --depth=1 http://git.mirror.gddi.io/mirror/FFmpeg.git && cd FFmpeg && \
    PKG_CONFIG_PATH="/usr/local/x86_64-intel-gnu/lib/pkgconfig" \
    ./configure \
        --prefix=/usr/local/x86_64-intel-gnu \
        --disable-debug \
        --disable-programs \
        --enable-shared \
        --enable-openssl \
        --enable-vaapi \
        --enable-nonfree \
        --enable-optimizations \
        --extra-libs="-lpthread -lm" && \
    make -j && make install && \
    rm -rf /tmp/*

# MLU270 编译环境
RUN mkdir -p /usr/local/x86_64-mlu270-gnu
RUN cd /tmp && \
    wget --content-disposition \
    http://cacher.devops.io/api/cacher/files/8893201dc1477168b297435dd41b6bd307bf0d49b9312e0b7ff55105304ab8d0 && \
    dpkg -i cntoolkit_1.7.3-2.ubuntu18.04_amd64.deb && \
    rm -rf /tmp/*
RUN apt update && \
    apt install -y cnas cncc cncodec cndev cndrv cnlicense cnpapi cnperf cnrt cnrtc && \
    rm -rf /var/lib/apt/lists/*
RUN cd /tmp && \
    git clone http://git.mirror.gddi.io/mirror/ffmpeg-mlu.git && \
    cd ffmpeg-mlu && \
    git clone http://git.mirror.gddi.io/mirror/FFmpeg.git -b release/4.2 --depth=1 && \
    cd FFmpeg && \
    git apply ../ffmpeg4.2_mlu200.patch && \
    ./configure \
        --prefix=/usr/local/x86_64-mlu270-gnu \
        --disable-debug \
        --disable-programs \
        --enable-shared \
        --enable-mlu \
        --enable-mlumpp \
        --enable-gpl \
        --enable-version3 \
        --enable-nonfree \
        --enable-stripping \
        --disable-x86asm \
        --enable-optimizations \
        --extra-libs="-lcnrt -lcncodec -lcndrv -lpthread -lm" \
        --extra-cflags="-I/usr/local/neuware/include" \
        --extra-ldflags="-L/usr/local/neuware/lib64" && \
    make -j && make install && \
    rm -rf /tmp/*
RUN cd /tmp && \
    git clone -b 4.5.5 --depth=1 http://git.mirror.gddi.io/mirror/opencv.git && cd opencv && \
    cmake -S . -Bbuild -G Ninja -DCMAKE_INSTALL_PREFIX=/usr/local/x86_64-mlu270-gnu -DCMAKE_BUILD_TYPE=Release \
        -DWITH_TBB=ON -DBUILD_LIST=core,highgui,improc,imgcodecs,video,videoio,photo,gapi,calib3d && \
    cmake --build build/ --target install && \
    rm -rf /tmp/*

# MLU370 编译环境
RUN mkdir -p /usr/local/x86_64-mlu370-gnu
RUN cd /tmp && \
    wget --content-disposition \
        http://cacher.devops.io/api/cacher/files/5c2963ede0eafb72a0928eba4a0d9bacc4ffad81856f111fac546b013ae0e491 && \
    wget --content-disposition \
        http://cacher.devops.io/api/cacher/files/a5ab3e9abbf32bb8ccd44d71473a798748a32565fdc9bb17d56dc4f3f93a9f4f && \
    wget --content-disposition \
        http://cacher.devops.io/api/cacher/files/f1ea87e0f937b471fac2a8d9c5a8a6f57ec02210522a3e2c99a6a8ab20ff168f && \
    wget --content-disposition \
        http://cacher.devops.io/api/cacher/files/afffb46eddf3d9cb8d0e68ad651e47da2cdc6bf8461e6a7f47ad86c8ee53a9da && \
    dpkg -x cntoolkit_3.0.2-1.ubuntu18.04_amd64.deb . && \
    dpkg -x cncv_1.0.0-1.ubuntu18.04_amd64.deb . && \
    dpkg -x cnlight_0.15.2-1.abiold.ubuntu18.04_amd64.deb . && \
    dpkg -x magicmind-0.13.1-1.ubuntu18.04_amd64.deb . && \
    dpkg -x var/cntoolkit-3.0.2/cnrt_6.0.2-1.ubuntu18.04_amd64.deb . && \
    dpkg -x var/cntoolkit-3.0.2/cndrv_2.0.2-1.ubuntu18.04_amd64.deb . && \
    dpkg -x var/cntoolkit-3.0.2/cncodec3_1.0.2-1.ubuntu18.04_amd64.deb . && \
    dpkg -x var/cntoolkit-3.0.2/cncodec_1.14.1-1.ubuntu18.04_amd64.deb . && \
    dpkg -x var/cntoolkit-3.0.2/cndev_3.0.2-1.ubuntu18.04_amd64.deb . && \
    dpkg -x var/cntoolkit-3.0.2/cnrtc_0.6.0-1.ubuntu18.04_amd64.deb . && \
    dpkg -x var/cntoolkit-3.0.2/llvm-mm-cxx11-old-abi_1.1.1-1.ubuntu18.04_amd64.deb . && \
    dpkg -x var/cntoolkit-3.0.2/cnpapi_3.0.2-1.ubuntu18.04_amd64.deb . && \
    dpkg -x var/cntoolkit-3.0.2/cnmlrt_6.0.0-1.ubuntu18.04_amd64.deb . && \
    cp -r usr/local/neuware/include /usr/local/x86_64-mlu370-gnu/include && \
    cp -rd usr/local/neuware/lib64 /usr/local/x86_64-mlu370-gnu/lib && \
    cp usr/local/neuware/lib/llvm-mm-cxx11-old-abi/lib/* /usr/local/x86_64-mlu370-gnu/lib/ && \
    rm -rf /tmp/* && \
    chmod -R 777 /tmp
RUN cd /tmp && \
    git clone http://git.mirror.gddi.io/mirror/ffmpeg-mlu.git && \
    cd ffmpeg-mlu && \
    git clone http://git.mirror.gddi.io/mirror/FFmpeg.git -b release/4.2 --depth=1 && \
    cd FFmpeg && \
    git apply ../ffmpeg4.2_mlu300.patch && \
    ./configure \
        --prefix=/usr/local/x86_64-mlu370-gnu \
        --disable-debug \
        --disable-programs \
        --enable-shared \
        --enable-mlu \
        --enable-mlumpp \
        --enable-gpl \
        --enable-version3 \
        --enable-nonfree \
        --enable-stripping \
        --disable-x86asm \
        --enable-optimizations \
        --extra-libs="-lcnrt -lcncodec -lcndrv -lpthread -lm" \
        --extra-cflags="-I/usr/local/x86_64-mlu370-gnu/include" \
        --extra-ldflags="-L/usr/local/x86_64-mlu370-gnu/lib" && \
    make -j && make install && \
    rm -rf /tmp/*
RUN cd /tmp && \
    git clone -b 4.5.5 --depth=1 http://git.mirror.gddi.io/mirror/opencv.git && cd opencv && \
    cmake -S . -Bbuild -G Ninja -DCMAKE_INSTALL_PREFIX=/usr/local/x86_64-mlu370-gnu -DCMAKE_BUILD_TYPE=Release \
        -DWITH_TBB=ON -DBUILD_LIST=core,highgui,improc,imgcodecs,video,videoio,photo,gapi,calib3d && \
    cmake --build build/ --target install && \
    rm -rf /tmp/*


# Armv8 交叉编译链
RUN apt update && \
    apt install -y gcc-aarch64-linux-gnu g++-aarch64-linux-gnu gdb-multiarch && \
    rm -rf /var/lib/apt/lists/*

# BM1684 交叉编译环境
RUN apt update && \
    apt install -y unzip && \
    rm -rf /var/lib/apt/lists/*
RUN mkdir -p /usr/local/aarch64-bm1684-soc/include && mkdir -p /usr/local/aarch64-bm1684-soc/lib
RUN cd /tmp && \
    wget --content-disposition \
        http://cacher.devops.io/api/cacher/files/d22036cc6fa6351a8f439f1cb59803021f8192089f85a827b8b7ac6c1d768a85 && \
    unzip Release_230301-public.zip && \
    dpkg -x Release_230301-public/sophon-img_20230327_063808/bsp-debs/sophon-soc-libsophon_0.4.6_arm64.deb . && \
    dpkg -x Release_230301-public/sophon-img_20230327_063808/bsp-debs/sophon-soc-libsophon-dev_0.4.6_arm64.deb . && \
    dpkg -x Release_230301-public/sophon-mw_20230327_040051/sophon-mw-soc-sophon-ffmpeg_0.6.0_arm64.deb . && \
    dpkg -x Release_230301-public/sophon-mw_20230327_040051/sophon-mw-soc-sophon-ffmpeg-dev_0.6.0_arm64.deb . && \
    dpkg -x Release_230301-public/sophon-mw_20230327_040051/sophon-mw-soc-sophon-opencv_0.6.0_arm64.deb . && \
    dpkg -x Release_230301-public/sophon-mw_20230327_040051/sophon-mw-soc-sophon-opencv-dev_0.6.0_arm64.deb . && \
    cp -r opt/sophon/libsophon-0.4.6/include/* /usr/local/aarch64-bm1684-soc/include && \
    cp -r opt/sophon/libsophon-0.4.6/lib/* /usr/local/aarch64-bm1684-soc/lib && \
    cp -r opt/sophon/sophon-ffmpeg_0.6.0/include/* /usr/local/aarch64-bm1684-soc/include && \
    cp -r opt/sophon/sophon-ffmpeg_0.6.0/lib/* /usr/local/aarch64-bm1684-soc/lib && \
    cp -r opt/sophon/sophon-opencv_0.6.0/include/* /usr/local/aarch64-bm1684-soc/include && \
    cp -r opt/sophon/sophon-opencv_0.6.0/lib/* /usr/local/aarch64-bm1684-soc/lib && \
    sed -i 's/\/usr\/local/\/usr\/local\/aarch64-bm1684-soc/g' /usr/local/aarch64-bm1684-soc/lib/pkgconfig/* && \
    rm -rf /tmp/*

# MLU220 交叉编译环境
RUN mkdir -p /usr/local/aarch64-mlu220-soc
RUN cd /tmp && \
    wget --content-disposition \
        http://cacher.devops.io/api/cacher/files/88b603b9721d752af8a3652e040c24105ac717a69bfc3711bce2db24db438d41 && \
    tar -zxvf cntoolkit-edge_1.7.12-1_arm64.tar.gz && \
    mv neuware/pc/include /usr/local/aarch64-mlu220-soc/include && \
    mv neuware/pc/lib64 /usr/local/aarch64-mlu220-soc/lib && \
    wget --content-disposition \
        http://cacher.devops.io/api/cacher/files/a95d68ade4a07c0cefe4794e6af3d66c3155813ff115ef7524eed4b8075d64eb && \
    dpkg -x cncv-edge_0.4.302-1_arm64.deb ./ && \
    mv usr/local/neuware/include/* /usr/local/aarch64-mlu220-soc/include && \
    mv usr/local/neuware/lib64/* /usr/local/aarch64-mlu220-soc/lib && \
    rm -rf /tmp/*
RUN cd /tmp && \
    git clone http://git.mirror.gddi.io/mirror/ffmpeg-mlu.git && \
    git clone http://git.mirror.gddi.io/mirror/FFmpeg.git -b release/4.2 --depth=1 && \
    cd FFmpeg && \
    git apply ../ffmpeg-mlu/ffmpeg4.2_mlu200.patch && \
    ./configure \
        --prefix=/usr/local/aarch64-mlu220-soc \
        --enable-cross-compile \
        --target-os=linux \
        --arch=arm64 \
        --cross-prefix=/usr/bin/aarch64-linux-gnu- \
        --enable-mlu \
        --enable-mlumpp \
        --disable-debug \
        --disable-programs \
        --enable-shared \
        --enable-gpl \
        --enable-version3 \
        --enable-nonfree \
        --disable-x86asm \
        --enable-optimizations \
        --extra-libs="-lcnrt -lcncodec -lcndrv -ljpu -lion -lpthread -lm -ldl" \
        --extra-cflags="-I/usr/local/aarch64-mlu220-soc/include" \
        --extra-ldflags="-L/usr/local/aarch64-mlu220-soc/lib" && \
    make -j && make install && \
    rm -rf /tmp/*

RUN cd /tmp && \
    git clone -b 4.5.5 --depth=1 http://git.mirror.gddi.io/mirror/opencv.git && cd opencv && \
    cmake -S . -Bbuild -G Ninja -DCMAKE_INSTALL_PREFIX=/usr/local/aarch64-mlu220-soc \
        -DCMAKE_BUILD_TYPE=Release -DCMAKE_SYSTEM_PROCESSOR=armv8 -DCMAKE_SYSTEM_NAME=Linux \
        -DCMAKE_C_COMPILER=aarch64-linux-gnu-gcc -DCMAKE_CXX_COMPILER=aarch64-linux-gnu-g++ \
        -DBUILD_LIST=core,highgui,improc,imgcodecs,video,videoio,photo,gapi,calib3d && \
    cmake --build build/ --target install && \
    rm -rf /tmp/*

# Jetson 交叉编译环境
# FFmpeg & OpenCV 只用于编译代码 部署需链接具体平台动态库
RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-key F60F4B3D7FA2AF80
RUN mkdir -p /usr/local/aarch64-jetson-soc/include && mkdir -p /usr/local/aarch64-jetson-soc/lib
RUN cd /tmp && \
    wget --content-disposition \
        http://cacher.devops.io/api/cacher/files/2964ee1da3f9949939b18bcb49396a5ff56d480c76a8df617f6417c4981d7e30 && \
    tar -zxvf jetpack_files.tar.gz && \
    dpkg -i /tmp/jetpack_files/cuda-repo-cross-aarch64*.deb /tmp/jetpack_files/cuda-repo-ubuntu*_amd64.deb && \
    dpkg -x /tmp/jetpack_files/libcudnn[7-8]_*-1+cuda10.[0-9]_arm64.deb /tmp/cudnn && \
    dpkg -x /tmp/jetpack_files/libcudnn[7-8]-dev_*-1+cuda10.[0-9]_arm64.deb /tmp/cudnn && \
    mv /tmp/cudnn/usr/include/aarch64-linux-gnu/* /usr/local/aarch64-jetson-soc/include && \
    mv /tmp/cudnn/usr/lib/aarch64-linux-gnu/* /usr/local/aarch64-jetson-soc/lib && \
    dpkg -x /tmp/jetpack_files/libnvinfer[0-8]_*-1+cuda10.[0-9]_arm64.deb /tmp/tensorrt && \
    dpkg -x /tmp/jetpack_files/libnvinfer-dev_*-1+cuda10.[0-9]_arm64.deb /tmp/tensorrt && \
    dpkg -x /tmp/jetpack_files/libnvparsers[6-8]_*-1+cuda10.[0-9]_arm64.deb /tmp/tensorrt && \
    dpkg -x /tmp/jetpack_files/libnvparsers-dev_*-1+cuda10.[0-9]_arm64.deb /tmp/tensorrt && \
    dpkg -x /tmp/jetpack_files/libnvinfer-plugin[6-8]_*-1+cuda10.[0-9]_arm64.deb /tmp/tensorrt && \
    dpkg -x /tmp/jetpack_files/libnvinfer-plugin-dev_*-1+cuda10.[0-9]_arm64.deb /tmp/tensorrt && \
    dpkg -x /tmp/jetpack_files/libnvonnxparsers[6-8]_*-1+cuda10.[0-9]_arm64.deb /tmp/tensorrt && \
    dpkg -x /tmp/jetpack_files/libnvonnxparsers-dev_*-1+cuda10.[0-9]_arm64.deb /tmp/tensorrt && \
    mv /tmp/tensorrt/usr/include/aarch64-linux-gnu/* /usr/local/aarch64-jetson-soc/include && \
    mv /tmp/tensorrt/usr/lib/aarch64-linux-gnu/* /usr/local/aarch64-jetson-soc/lib && \
    rm -rf /tmp/*
RUN apt update && \
    apt install -y cuda-cross-aarch64 && \
    rm -rf /var/lib/apt/lists/*
RUN cd /tmp && \
    git clone -b n4.2.2 --depth=1 http://git.mirror.gddi.io/mirror/FFmpeg.git && cd FFmpeg && \
    ./configure \
        --prefix=/usr/local/aarch64-jetson-soc \
        --enable-cross-compile \
        --target-os=linux \
        --arch=arm64 \
        --cross-prefix=/usr/bin/aarch64-linux-gnu- \
        --disable-debug \
        --disable-programs \
        --enable-shared \
        --enable-gpl \
        --enable-version3 \
        --enable-nonfree \
        --disable-x86asm \
        --enable-optimizations \
        --extra-libs="-lpthread -lm" \
        --extra-cflags="-I/usr/local/aarch64-jetson-soc/include" \
        --extra-ldflags="-L/usr/local/aarch64-jetson-soc/lib" && \
    make -j && make install && \
    rm -rf /tmp/*
RUN cd /tmp && \
    git clone -b 4.1.1 --depth=1 http://git.mirror.gddi.io/mirror/opencv.git && cd opencv && \
    cmake -S . -Bbuild -G Ninja -DCMAKE_INSTALL_PREFIX=/usr/local/aarch64-jetson-soc \
        -DCMAKE_BUILD_TYPE=Release -DCMAKE_SYSTEM_PROCESSOR=armv8 -DCMAKE_SYSTEM_NAME=Linux \
        -DCMAKE_C_COMPILER=aarch64-linux-gnu-gcc -DCMAKE_CXX_COMPILER=aarch64-linux-gnu-g++ \
        -DBUILD_LIST=core,highgui,improc,imgcodecs,video,videoio,photo,gapi,calib3d && \
    cmake --build build/ --target install && \
    rm -rf /tmp/*
RUN cd /tmp && \
    wget http://ports.ubuntu.com/pool/main/libg/libglvnd/libegl1_1.0.0-2ubuntu2_arm64.deb && \
    wget http://ports.ubuntu.com/pool/main/libg/libglvnd/libglvnd0_1.0.0-2ubuntu2_arm64.deb && \
    dpkg -x libegl1_1.0.0-2ubuntu2_arm64.deb /tmp/egl && \
    dpkg -x libglvnd0_1.0.0-2ubuntu2_arm64.deb /tmp/egl && \
    mv /tmp/egl/usr/lib/aarch64-linux-gnu/* /usr/local/aarch64-jetson-soc/lib && \
    rm -rf /tmp/*

# rv1126 交叉编译环境
RUN apt update && \
    apt install -y gcc-arm-linux-gnueabihf g++-arm-linux-gnueabihf pkg-config-arm-linux-gnueabihf && \
    rm -rf /var/lib/apt/lists/*
RUN mkdir -p /usr/local/armv7hf-rv1126-soc/include && \
    mkdir -p /usr/local/armv7hf-rv1126-soc/lib && \
    mkdir /usr/local/armv7hf-rv1126-soc/include/rga && \
    mkdir /usr/local/armv7hf-rv1126-soc/lib/pkgconfig
RUN cd /tmp && \
    wget http://mirrors.ustc.edu.cn/ubuntu-ports/pool/main/libd/libdrm/libdrm2_2.4.99-1ubuntu1~18.04.2_armhf.deb && \
    wget http://mirrors.ustc.edu.cn/ubuntu-ports/pool/main/libd/libdrm/libdrm-dev_2.4.99-1ubuntu1~18.04.2_armhf.deb && \
    dpkg -X libdrm2_2.4.99-1ubuntu1~18.04.2_armhf.deb /tmp/libdrm && \
    dpkg -X libdrm-dev_2.4.99-1ubuntu1~18.04.2_armhf.deb /tmp/libdrm && \
    mv libdrm/usr/include/libdrm /usr/local/armv7hf-rv1126-soc/include/ && \
    mv libdrm/usr/lib/arm-linux-gnueabihf/libdrm.so* /usr/local/armv7hf-rv1126-soc/lib/ && \
    mv libdrm/usr/lib/arm-linux-gnueabihf/pkgconfig/libdrm.pc /usr/local/armv7hf-rv1126-soc/lib/pkgconfig/ && \
    sed -i 's/\/usr/\/usr\/local\/armv7hf-rv1126-soc/g' /usr/local/armv7hf-rv1126-soc/lib/pkgconfig/libdrm.pc && \
    sed -i 's/\/arm-linux-gnueabihf//g' /usr/local/armv7hf-rv1126-soc/lib/pkgconfig/libdrm.pc && \
    rm -rf /tmp/*
RUN ls -al / && \
    cd /tmp && \
    git clone -b 1.3.2_release --depth 1 https://github.com/airockchip/librga.git && \
    mv librga/include/* /usr/local/armv7hf-rv1126-soc/include/rga/ && \
    mv librga/libs/Linux/gcc-armhf/librga.so /usr/local/armv7hf-rv1126-soc/lib/librga.so.2 && \
    cd /usr/local/armv7hf-rv1126-soc/lib && \
    ln -s librga.so.2 librga.so && \
    rm -rf /tmp/*
RUN cd /tmp && \
    git clone --depth=1 https://github.com/rockchip-linux/mpp.git && cd mpp && \
    cmake -S . -Bbuild -G Ninja -DCMAKE_INSTALL_PREFIX=/usr/local/armv7hf-rv1126-soc -DCMAKE_BUILD_TYPE=Release -DCMAKE_SYSTEM_PROCESSOR=armv7hf \
        -DCMAKE_SYSTEM_NAME=Linux -DCMAKE_C_COMPILER=arm-linux-gnueabihf-gcc -DCMAKE_CXX_COMPILER=arm-linux-gnueabihf-g++ && \
    cmake --build build/ --target install && \
    rm -rf /tmp/*
RUN cd /tmp && \
    git clone -b 4.4 --depth=1 http://git.mirror.gddi.io/mirror/ffmpeg-rk.git && cd ffmpeg-rk && \
    PKG_CONFIG_PATH="/usr/local/armv7hf-rv1126-soc/lib/pkgconfig" \
    ./configure \
        --prefix=/usr/local/armv7hf-rv1126-soc \
        --enable-cross-compile \
        --target-os=linux \
        --arch=armv7hf \
        --cross-prefix=/usr/bin/arm-linux-gnueabihf- \
        --disable-debug \
        --disable-programs \
        --enable-shared \
        --enable-libdrm \
        --enable-rkmpp \
        --enable-gpl \
        --enable-version3 \
        --enable-nonfree \
        --disable-x86asm \
        --enable-optimizations \
        --extra-libs="-ldrm -lpthread -lm" \
        --extra-cflags="-I/usr/local/armv7hf-rv1126-soc/include" \
        --extra-ldflags="-L/usr/local/armv7hf-rv1126-soc/lib" && \
    make -j && make install && \
    rm -rf /tmp/*
RUN cd /tmp && \
    git clone -b 4.4.0 --depth=1 http://git.mirror.gddi.io/mirror/opencv.git && cd opencv && \
    cmake -S . -Bbuild -G Ninja -DCMAKE_INSTALL_PREFIX=/usr/local/armv7hf-rv1126-soc \
        -DCMAKE_BUILD_TYPE=Release -DCMAKE_SYSTEM_PROCESSOR=armv7hf -DCMAKE_SYSTEM_NAME=Linux \
        -DCMAKE_C_COMPILER=arm-linux-gnueabihf-gcc -DCMAKE_CXX_COMPILER=arm-linux-gnueabihf-g++ \
        -DBUILD_LIST=core,highgui,improc,imgcodecs,video,videoio,photo,gapi,calib3d && \
    cmake --build build/ --target install && \
    rm -rf /tmp/*

# Conan 包管理工具
RUN ls -al / && \
    apt update && \
    apt install -y python3 python3-pip && \
    pip3 install conan==1.57.0 && \
    rm -rf /var/lib/apt/lists/*

# Docker
RUN apt update && \
    apt install -y curl lsb-release && \
    rm -rf /var/lib/apt/lists/*
RUN mkdir -p /etc/apt/keyrings && \
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /etc/apt/keyrings/docker.gpg && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
        $(lsb_release -cs) stable" | tee /etc/apt/sources.list.d/docker.list > /dev/null && \
    apt update && \
    apt -y install docker-ce docker-ce-cli containerd.io docker-compose-plugin && \
    rm -rf /var/lib/apt/lists/*

# User && ZSH
RUN apt update && \
    apt install -y sudo zsh && \
    rm -rf /var/lib/apt/lists/*

ARG uid=1000
ARG gid=1000
RUN groupadd -r -f -g ${gid} gddi && useradd -o -r -l -u ${uid} -g ${gid} -ms /bin/bash gddi && \
    usermod -aG sudo gddi && \
    echo 'gddi:gddi1234' | chpasswd && \
    mkdir -p /workspace && chown gddi /workspace && \
    gpasswd -a gddi docker

# Clear
RUN apt clean autoclean && apt autoremove --yes

# Env
ENV PATH=/usr/local/cuda-11.4/bin:/usr/local/bin:$PATH
ENV LD_LIBRARY_PATH=/usr/local/cuda-11.4/lib64:/usr/local/lib:$LD_LIBRARY_PATH
ENV C_INCLUDE_PATH=/usr/local/include:$C_INCLUDE_PATH
ENV CPLUS_INCLUDE_PATH=/usr/local/include:$CPLUS_INCLUDE_PATH

USER gddi

RUN conan remote add gddi-conan-local http://devops.io:8081/artifactory/api/conan/gddi-conan-local --insert 0 && \
    conan user -p Gddi@1234 -r gddi-conan-local gddi


RUN git clone https://github.com/ohmyzsh/ohmyzsh.git ~/.oh-my-zsh && \
    git clone https://github.com/zsh-users/zsh-autosuggestions ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-autosuggestions && \
    git clone https://github.com/zsh-users/zsh-syntax-highlighting.git ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-syntax-highlighting && \
    git clone https://github.com/zsh-users/zsh-history-substring-search ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-history-substring-search && \
    cp ~/.oh-my-zsh/templates/zshrc.zsh-template ~/.zshrc && \
    sed -i '/^#\ ZSH_CUSTOM/c ZSH_CUSTOM=~\/.oh-my-zsh\/custom' ~/.zshrc && \
    sed -i '/^plugins/c plugins=(git zsh-autosuggestions zsh-syntax-highlighting zsh-history-substring-search)' ~/.zshrc

WORKDIR /workspace

RUN ["/bin/zsh"]

# docker build . -f docker/base-devel/Dockerfile -t hub.gddi.com/devel/inference-engine-devel:1.0.1