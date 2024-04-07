FROM ubuntu:18.04

RUN mkdir /opt/install

COPY docker/common/sources_ubuntu18.list /etc/apt/sources.list
COPY docker/docker_message.md /root
COPY docker/install_build.sh /opt
COPY docker/install/ /opt/install

RUN chmod +x /opt/install_build.sh && ./opt/install_build.sh common

WORKDIR /root