# 1. How To Build

## 1.1 Linux
### 1.1.1 基础开发环境
```bash
docker run -itd --privileged=true --net=host --gpus=all -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,video hub.gddi.com/devel/inference-engine-devel:1.0.0
```
### 1.1.2 CMake 编译
```bash
# Nvidia
cmake -S . -Bbuild-nvidia -G Ninja -DCMAKE_BUILD_TYPE=Release -DTARGET_CHIP=nvidia -E PKG_CONFIG_PATH=/usr/local/x86_64-nvidia-gnu/lib/pkgconfig -DOpenCV_DIR=/usr/local/x86_64-nvidia-gnu/lib/cmake/opencv4

# 交叉编译 bm1684
cmake -S . -Bbuild-bm1684 -G Ninja -DCMAKE_BUILD_TYPE=Release -DTARGET_CHIP=bm1684 -DCMAKE_SYSTEM_NAME=Linux -DCMAKE_SYSTEM_PROCESSOR=armv8 -DCMAKE_C_COMPILER=aarch64-linux-gnu-gcc -DCMAKE_CXX_COMPILER=aarch64-linux-gnu-g++ -DCMAKE_FIND_ROOT_PATH_MODE_PROGRAM=NEVER -DCMAKE_FIND_ROOT_PATH_MODE_LIBRARY=ONLY -DCMAKE_FIND_ROOT_PATH_MODE_INCLUDE=ONLY -DCMAKE_FIND_ROOT_PATH_MODE_PACKAGE=ONLY -DCMAKE_SYSROOT=/usr/local/aarch64-bm1684-soc -DCMAKE_FIND_ROOT_PATH=/usr/local/aarch64-bm1684-soc -E PKG_CONFIG_PATH=/usr/local/aarch64-bm1684-soc/lib/pkgconfig

```

### 1.2.2 VSCode(CMake Tools)
> `Ctrl + Shift + P` 打开配置，搜索设置 `cmake.generator`，设置 `Ninja` 这样可以提高编译速度。

# 2. 配置文件说明

```typescript
// 以下为typescript代码
interface IConfig {
    version: 'v0';                              // 固定为'v0', 正式Release的时候会固定下来。
    nodes: {
        id: number;                             // 节点ID
        type: string;                           // 节点类型，必须是程序支持的类型
        name: string;                           // 节点名，可以任意命名
        runner: string;                         // 节点运行的目标线程名（目标线程会自动创建）
        props: {                                // 节点属性，可选，key:value的一个字典，用于配置节点
            [index: string]: string | number
        }
    }[];
    pipe: [number, number, number, number][]    // 节点数据流，from[ep_out], to[ep_in] 四个参数
}
```

# 2. FFmpeg 编译
1. 寒武纪 MLU220
```
./configure --prefix=/usr/aarch64-cambricon-soc --enable-cross-compile --target-os=linux --arch=arm64 --cc=aarch64-linux-gnu-gcc --enable-version3 --enable-gpl --disable-debug --enable-pthreads --disable-yasm --disable-asm --disable-static --enable-shared --disable-stripping --disable-optimizations --enable-mlumpp --extra-cflags="-I/usr/aarch64-cambricon-soc/include" --extra-ldflags="-L/usr/aarch64-cambricon-soc/lib64" --extra-cflags=-fPIC --extra-libs="-lcncodec -lcnrt -ldl -lcndev -lcndrv -lion -ljpu"
```
2. 英伟达 Jetson
```
./configure --prefix=/usr --enable-nvv4l2dec --enable-libv4l2 --enable-shared --extra-libs='-L/usr/lib/aarch64-linux-gnu/tegra -lnvbuf_utils' --extra-cflags='-I /usr/src/jetson_multimedia_api/include/'
```

# 3. 后处理 SDK 接口说明
## 编译环境
- ubuntu18.04
- gcc-version: 7.5
- cmake: 3.24.0
- pkg-config: 0.29.1

## 安装依赖
```bash
apt install -y cmake

# 交叉编译
apt install -y gcc-aarch64-linux-gnu g++-aarch64-linux-gnu
```

## 重新配置 pkg-config prefix 路径
```
sed -i "1s/.*/prefix=$(echo $PWD | sed -e 's/\//\\\//g')/g" lib/pkgconfig/gddi_post.pc
sed -i "1s/.*/prefix=$(echo $PWD | sed -e 's/\//\\\//g')/g" lib/pkgconfig/gddi_codec.pc
```

## 编译测试
```bash
# 配置
cmake -S . -G Ninja -Bbuild

# 编译
cmake --build buid/
```

## 模块说明
- [编解码模块](doc/codec.md)
- [跟踪模块](doc/target_tracker.md)
- [越界计数模块](doc/cross_border.md)