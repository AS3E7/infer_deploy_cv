FROM arm64v8/ubuntu:20.04

RUN apt update && \
    apt install -y libssl-dev locales wget unzip libatomic1 libncurses5 && \
    rm -rf /var/lib/apt/lists/*
RUN sed -i 's/# en_US.UTF-8/en_US.UTF-8/g' /etc/locale.gen
RUN sed -i 's/# zh_CN.GBK/zh_CN.GBK/g' /etc/locale.gen
RUN locale-gen

RUN cd /tmp && \
    wget --content-disposition \
        http://cacher.devops.io/api/cacher/files/d22036cc6fa6351a8f439f1cb59803021f8192089f85a827b8b7ac6c1d768a85 && \
    unzip Release_230301-public.zip && \
    dpkg -i Release_230301-public/sophon-img_20230327_063808/bsp-debs/sophon-soc-libsophon_0.4.6_arm64.deb && \
    dpkg -i Release_230301-public/sophon-mw_20230327_040051/sophon-mw-soc-sophon-ffmpeg_0.6.0_arm64.deb && \
    dpkg -i Release_230301-public/sophon-mw_20230327_040051/sophon-mw-soc-sophon-opencv_0.6.0_arm64.deb && \
    rm Release_230301-public.zip && \
    rm -r Release_230301-public

COPY data/action/* /home/config/
COPY data/jpeg-preview.json /home/templates/
COPY data/config/modbus_cfg.json /home/config/
COPY data/config/NotoSansCJK-Regular.ttc /home/config/
COPY release/aarch64/bm1684/lib/ /usr/local/lib/
COPY release/aarch64/bm1684/bin/ /usr/local/bin/


ENV LD_LIBRARY_PATH=/opt/sophon/libsophon-0.4.6/lib:/usr/local/lib:$LD_LIBRARY_PATH

WORKDIR /home

ENTRYPOINT [ "node_srv" ]