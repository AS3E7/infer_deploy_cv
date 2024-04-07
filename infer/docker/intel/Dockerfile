FROM openvino/ubuntu20_runtime:2022.1.0

USER root

RUN /opt/intel/openvino/extras/scripts/download_opencv.sh
RUN apt update && \
    apt install -y libva-dev libmfx-dev libva-drm2 libvpx-dev libssl-dev libtbb-dev && \
    rm -rf /var/lib/apt/lists/*

COPY data/jpeg-preview.json /home/templates/
COPY data/config/modbus_cfg.json /home/config/
COPY data/config/NotoSansCJK-Regular.ttc /home/config/
COPY release/x86_64/intel/lib/ /usr/local/lib/
COPY release/x86_64/intel/bin/ /usr/local/bin/

ENV LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
ENV LD_LIBRARY_PATH=/opt/intel/openvino_2022/extras/opencv/lib:$LD_LIBRARY_PATH
ENV LD_LIBRARY_PATH=/opt/intel/openvino_2022/runtime/lib/intel64:$LD_LIBRARY_PATH

WORKDIR /home

ENTRYPOINT [ "node_srv" ]