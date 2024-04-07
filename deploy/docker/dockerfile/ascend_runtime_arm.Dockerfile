FROM ubuntu:18.04

# RUN chmod +x /opt/install_build.sh && ./opt/install_build.sh common ascend
RUN apt update -y && apt install -y \
    gcc g++ make cmake wget zlib1g zlib1g-dev libbz2-dev openssl libsqlite3-dev libssl-dev libxslt1-dev \
    libffi-dev unzip pciutils net-tools libblas-dev gfortran libblas3 libopenblas-dev libncursesw5-dev libjpeg-dev 

RUN wget -P /tmp https://www.python.org/ftp/python/3.7.5/Python-3.7.5.tgz
RUN tar -zxvf /tmp/Python-3.7.5.tgz -C /tmp && cd /tmp/Python-3.7.5 && \
    ./configure --prefix=/usr/local/python3.7.5 --enable-loadable-sqlite-extensions --enable-shared && \
    make && make install && \
    ln -s /usr/local/python3.7.5/bin/python3 /usr/local/python3.7.5/bin/python3.7.5 && \
    ln -s /usr/local/python3.7.5/bin/pip3 /usr/local/python3.7.5/bin/pip3.7.5
ENV PATH=/usr/local/python3.7.5/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
ENV LD_LIBRARY_PATH=/usr/local/python3.7.5/lib

WORKDIR /root
