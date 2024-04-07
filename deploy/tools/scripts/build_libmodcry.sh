#!/bin/bash

function install_modcrypt_build(){
    cd thirdparty/libmodcrypt/
    mkdir build -p && cd build
    cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr/local/3rdparty/libmodcrypt/release_x86
    make -j$(nproc) && make install
    cd -
}

function install_openssl_build(){
    cd thirdparty/openssl \
    && mkdir build -p && cd build \
    && ../config --prefix=/usr/local/3rdparty/openssl/release_x86 && make -j$(nproc) && make install
    cd -
}

# install_openssl_build
install_modcrypt_build