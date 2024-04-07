FROM nvidia/cuda:11.4.0-cudnn8-devel-ubuntu18.04

RUN mkdir /opt/install

COPY docker/common/sources_ubuntu18.list /etc/apt/sources.list
COPY docker/docker_message.md /root
COPY docker/install_build.sh /opt
COPY docker/install/ /opt/install

RUN chmod +x /opt/install_build.sh && ./opt/install_build.sh common bmnn

WORKDIR /root