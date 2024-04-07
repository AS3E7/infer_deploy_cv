FROM ubuntu:18.04

RUN apt update && \
    apt install -y wget libssl-dev locales libjpeg-dev libtbb-dev libunwind-dev libpng-dev libtiff-dev && \
    rm -rf /var/lib/apt/lists/*

RUN cd /tmp && \
    wget --content-disposition \
    http://cacher.devops.io/api/cacher/files/5c2963ede0eafb72a0928eba4a0d9bacc4ffad81856f111fac546b013ae0e491 && \
    dpkg -i cntoolkit_3.0.2-1.ubuntu18.04_amd64.deb && \
    rm -rf /tmp/*

RUN apt update && \
    apt install -y cnmlrt cnpapi cncodec cncodec3 cndev cndrv cnrt cnrtc llvm-mm-cxx11-old-abi && \
    rm -rf /var/lib/apt/lists/*

RUN cd /tmp && \
    wget --content-disposition \
    http://cacher.devops.io/api/cacher/files/a5ab3e9abbf32bb8ccd44d71473a798748a32565fdc9bb17d56dc4f3f93a9f4f && \
    wget --content-disposition \
    http://cacher.devops.io/api/cacher/files/f1ea87e0f937b471fac2a8d9c5a8a6f57ec02210522a3e2c99a6a8ab20ff168f && \
    wget --content-disposition \
    http://cacher.devops.io/api/cacher/files/afffb46eddf3d9cb8d0e68ad651e47da2cdc6bf8461e6a7f47ad86c8ee53a9da && \
    wget --content-disposition \
    http://cacher.devops.io/api/cacher/files/d52cff92714e294aa32ce8d344c10ccdebd221745b19a9890e30f0ac2b39c25e && \
    wget --content-disposition \
    http://cacher.devops.io/api/cacher/files/236fa7e027e8f3b016aca64ba94fa309d4989f2dbc889cfacd669f2eb557b202 && \
    dpkg -i cncv_1.0.0-1.ubuntu18.04_amd64.deb && \
    dpkg -i cnlight_0.15.2-1.abiold.ubuntu18.04_amd64.deb && \
    dpkg -i cnnl_1.12.1-1.ubuntu18.04_amd64.deb && \
    dpkg -i cnnlextra_0.18.0-1.ubuntu18.04_amd64.deb && \
    dpkg -i magicmind-0.13.1-1.ubuntu18.04_amd64.deb && \
    rm -rf /tmp/*

RUN sed -i 's/# en_US.UTF-8/en_US.UTF-8/g' /etc/locale.gen
RUN sed -i 's/# zh_CN.GBK/zh_CN.GBK/g' /etc/locale.gen
RUN locale-gen

COPY data/jpeg-preview.json /home/templates/
COPY data/config/modbus_cfg.json /home/config/
COPY data/config/NotoSansCJK-Regular.ttc /home/config/
COPY release/x86_64/mlu370/lib/ /usr/local/lib/
COPY release/x86_64/mlu370/bin/ /usr/local/bin/

ENV LD_LIBRARY_PATH=/usr/local/lib:/usr/local/neuware/lib64:/usr/local/neuware/lib/llvm-mm-cxx11-old-abi/lib:$LD_LIBRARY_PATH

WORKDIR /home

ENTRYPOINT [ "node_srv" ]