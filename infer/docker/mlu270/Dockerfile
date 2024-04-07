FROM ubuntu:18.04

RUN apt update && \
    apt install -y wget && \
    rm -rf /var/lib/apt/lists/*

RUN cd /tmp && \
    wget --content-disposition \
    http://cacher.devops.io/api/cacher/files/8893201dc1477168b297435dd41b6bd307bf0d49b9312e0b7ff55105304ab8d0 && \
    dpkg -i cntoolkit_1.7.3-2.ubuntu18.04_amd64.deb && \
    rm -rf /tmp/*

RUN apt update && \
    apt install -y libssl-dev locales libjpeg-dev libtbb-dev libunwind-dev libpng-dev libtiff-dev \
    cnas cncc cncodec cndev cndrv cnlicense cnpapi cnperf cnrt cnrtc && \
    rm -rf /var/lib/apt/lists/*
RUN sed -i 's/# en_US.UTF-8/en_US.UTF-8/g' /etc/locale.gen
RUN sed -i 's/# zh_CN.GBK/zh_CN.GBK/g' /etc/locale.gen
RUN locale-gen

COPY data/jpeg-preview.json /home/templates/
COPY data/config/modbus_cfg.json /home/config/
COPY data/config/NotoSansCJK-Regular.ttc /home/config/
COPY release/x86_64/mlu270/lib/ /usr/local/lib/
COPY release/x86_64/mlu270/bin/ /usr/local/bin/

ENV LD_LIBRARY_PATH=/usr/local/lib:/usr/local/neuware/lib64:$LD_LIBRARY_PATH

WORKDIR /home

ENTRYPOINT [ "node_srv" ]