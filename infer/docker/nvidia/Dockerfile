FROM nvidia/cuda:11.4.0-cudnn8-runtime-ubuntu20.04

RUN apt update && apt install -y libnvinfer8=8.2.5-1+cuda11.4 libnvonnxparsers8=8.2.5-1+cuda11.4 \
    libnvparsers8=8.2.5-1+cuda11.4 libnvinfer-plugin8=8.2.5-1+cuda11.4 && \
    rm -rf /var/lib/apt/lists/*

RUN apt update && \
    apt install -y libjpeg-dev libpng-dev libtiff-dev libtbb-dev locales libcurl4-openssl-dev && \
    rm -rf /var/lib/apt/lists/*
RUN sed -i 's/# en_US.UTF-8/en_US.UTF-8/g' /etc/locale.gen
RUN sed -i 's/# zh_CN.GBK/zh_CN.GBK/g' /etc/locale.gen
RUN locale-gen

COPY data/feature-library.json /home/templates/
COPY data/jpeg-preview.json /home/templates/
COPY data/config/modbus_cfg.json /home/config/
COPY data/config/NotoSansCJK-Regular.ttc /home/config/
COPY data/config/alg_config.json /home/config/
COPY data/action/* /home/config/
COPY release/x86_64/nvidia/lib/ /usr/local/lib/
COPY release/x86_64/nvidia/bin/ /usr/local/bin/

WORKDIR /home

ENTRYPOINT [ "node_srv" ]