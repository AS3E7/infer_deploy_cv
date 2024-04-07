FROM openvino/ubuntu18_dev

USER root
RUN mkdir /opt/install

COPY docker/common/sources_ubuntu20.list /etc/apt/sources.list
COPY docker/docker_message.md /root
COPY docker/install_convert.sh /opt
COPY docker/install/ /opt/install

RUN chmod +x /opt/install_convert.sh && bash /opt/install_convert.sh common 
# openvino
RUN bash /opt/intel/openvino/extras/scripts/download_opencv.sh

WORKDIR /root