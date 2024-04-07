FROM nvcr.io/nvidia/l4t-tensorrt:r8.2.1-runtime

RUN apt update && \
    apt install -y libssl-dev locales libgtk2.0-0 libtbb2 && \
    rm -rf /var/lib/apt/lists/*

RUN sed -i 's/# en_US.UTF-8/en_US.UTF-8/g' /etc/locale.gen
RUN sed -i 's/# zh_CN.GBK/zh_CN.GBK/g' /etc/locale.gen
RUN locale-gen

RUN cd /tmp && \
    wget --content-disposition \
        http://cacher.devops.io/api/cacher/files/0edfb3ece5cff97d452ac55ca9caf4e5e40f083eae91dbf804bfc99096447aa6 && \
    dpkg -i OpenCV-4.1.1-2-gd5a58aa75-aarch64-libs.deb && \
    wget --content-disposition \
        http://cacher.devops.io/api/cacher/files/7a0e72c15d3bea5a62d00ae05f9db265118af76354960dab5c368ea00d3e0199 && \
    tar -zxvf libffmpeg-jetson-4.6.2-libs.tar.gz -C /usr/local/lib/ && \
    rm /tmp/*

COPY data/jpeg-preview.json /home/templates/
COPY data/config/modbus_cfg.json /home/config/
COPY data/config/NotoSansCJK-Regular.ttc /home/config/
COPY data/config/alg_config.json /home/config/
COPY release/armv8/jetson/lib/ /usr/local/lib/
COPY release/armv8/jetson/bin/ /usr/local/bin/

ENV LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH

WORKDIR /home

ENTRYPOINT [ "node_srv" ]