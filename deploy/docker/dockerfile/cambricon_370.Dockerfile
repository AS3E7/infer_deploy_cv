FROM ubuntu:18.04

RUN mkdir /opt/install

COPY docker/common/sources_ubuntu18.list /etc/apt/sources.list
COPY docker/docker_message.md /root
COPY docker/install_build.sh /opt
COPY docker/install/ /opt/install

RUN chmod +x /opt/install_build.sh && ./opt/install_build.sh common

# apt update -y
# dpkg -i /volume1/gddi-data/lgy/cambricon/toolkits_370/cntoolkit_v3.0.2/Ubuntu/18.04/cntoolkit_3.0.2-1.ubuntu18.04_amd64.deb
# apt update -y
# apt install cntoolkit-cloud -y

# apt-get install -y --no-install-recommends \
#     curl git wget vim build-essential \
#     cmake make ca-certificates nasm yasm  \
#     openssh-server libgoogle-glog-dev libgflags-dev libtbb-dev libcurl4-openssl-dev
#     # libopencv-dev 
#     # libsdl2-dev lcov ca-certificates net-tools           
#     # software-properties-common         
#     # libgtk2.0-dev pkg-config libfreetype6-dev libc6-dev-i386         
#     # apt-utils unzip

dpkg -i /volume1/gddi-data/lgy/cambricon/toolkits_370/cncv_v1.0.0/Ubuntu/18.04/cncv_1.0.0-1.ubuntu18.04_amd64.deb
dpkg -i /volume1/gddi-data/lgy/cambricon/toolkits_370/cnnl_extra_0.18.0/Ubuntu/18.04/cnnlextra_0.18.0-1.ubuntu18.04_amd64.deb 
dpkg -i /volume1/gddi-data/lgy/cambricon/toolkits_370/cnnl_1.12.1/Ubuntu/18.04/cnnl_1.12.1-1.ubuntu18.04_amd64.deb
dpkg -i /volume1/gddi-data/lgy/cambricon/toolkits_370/cnlight_0.15.2/Ubuntu/18.04/cnlight_0.15.2-1.abiold.ubuntu18.04_amd64.deb
dpkg -i /var/cntoolkit-3.0.2/llvm-mm-cxx11-old-abi_1.1.1-1.ubuntu18.04_amd64.deb
dpkg -i /volume1/gddi-data/lgy/cambricon/toolkits_370/magicmind_0.13.1/Ubuntu/18.04/abiold/magicmind-0.13.1-1.ubuntu18.04_amd64.deb

echo "export NEUWARE_HOME=/usr/local/neuware" >> ~/.bashrc
echo "LD_LIBRARY_PATH=/usr/local/neuware/lib64:/usr/local/neuware/lib/llvm-mm-cxx11-old-abi/lib/" >> ~/.bashrc
echo "export PATH=$PATH:/usr/local/neuware/bin/" >> ~/.bashrc

# /data/code/gddi-cambricon/3rdparty/libmodcrypt/release/x86_64/lib/:
# /data/code/gddi-cambricon/gddsdk/thirdparty/ffmpeg-mlu_release_mlu370/lib/:
# /data/code/gddi-cambricon/3rdparty/libmodcrypt/release/x86_64/lib/
WORKDIR /root