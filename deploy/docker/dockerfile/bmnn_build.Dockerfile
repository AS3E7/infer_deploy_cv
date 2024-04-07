FROM ubuntu:18.04

RUN mkdir /opt/install

COPY docker/common/sources_ubuntu18.list /etc/apt/sources.list
COPY docker/install_build.sh /opt
COPY docker/install/ /opt/install

RUN chmod +x /opt/install_build.sh && ./opt/install_build.sh common bmnn

# RUN apt-get install libssl-dev openssl pkg-config -y 
# RUN wget https://cmake.org/files/v3.22/cmake-3.22.1.tar.gz -P /tmp && cd /tmp/ && tar xvf cmake-3.22.1.tar.gz && cd cmake-3.22.1 && chmod 777 ./configure && ./configure && make -j12 && make install
# RUN update-alternatives --install /usr/bin/cmake cmake /usr/local/bin/cmake 1 --force

WORKDIR /root