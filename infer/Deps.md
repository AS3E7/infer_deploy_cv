# 安装必要的依赖

## 1. For Linux

```shell
# libhv
git clone https://github.com/ithewei/libhv.git
cd libhv
mkdir build
cd build
cmake ..
make
sudo make install

# blend2d
git clone https://github.com/asmjit/asmjit
git clone https://github.com/blend2d/blend2d
mkdir blend2d/build
cd blend2d/build
cmake .. -DCMAKE_BUILD_TYPE=Release
sudo make install


# ffmpeg
sudo apt install libavdevice-dev libavformat-dev libavcodec-dev

# boost
sudo apt install libboost-program-options-dev libboost-dev

```

## 2. For vcpkg cmake
```shell
sudo apt install libxdamage-dev
sudo apt install libgtk2.0-dev pkg-config
```