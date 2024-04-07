
# 1. SDK目录结构
```bash
.
├── bin     # 可直接执行的工具，初步测试用
│   ├── sample_runner_dataset
│   ├── sample_runner_pic
│   └── sample_runner_video
├── CMakeLists.txt
├── docker  # docker镜像文件或者dockerfile文件
│   ├── bmnn_build.Dockerfile
│   ├── bmnn_runtime.Dockerfile
│   ├── common
│   ├── install
│   └── install_build.sh
├── docs    
├── include # 接口头文件，包含三个层次的头文件
│   ├── api
│   ├── app
│   └── core
├── lib 
├── README.md
├── sample  # 测试样例，拷贝sample文件夹
│   ├── api
│   ├── app
│   ├── CMakeLists.txt
│   ├── common
│   ├── sample_runner_dataset.cpp
│   ├── sample_runner_pic.cpp
│   └── sample_runner_video.cpp
├── script  
│   ├── build.sh
│   ├── docker_build.sh
│   ├── docker_runtime.sh
│   └── env.sh
└── thirdparty
    ├── bin
    ├── include
    ├── lib
    └── share
```

# 2. 快速使用

快速使用分为两个阶段：代码编译和代码运行。代码编译需要在编译环境中进行，代码运行需要在目标机器上进行。

如果需要快速验证，则可跳过代码编译阶段，直接使用编译好的SDK进行代码运行。

## 2.1. 代码编译

### 1）创建编译环境的docker镜像

```bash
docker build -f docker/bmnn_build.Dockerfile -t devops.io:12580/lgy/test/gddeploy/bmnn/build/3.0:v0.2 .
```

或者可直接使用共达地提供的编译环境docker镜像

```bash
docker pull devops.io:12580/lgy/test/gddeploy/bmnn/build/3.0:v0.2
```

### 2） 运行编译环境docker镜像

```bash
docker run -it --rm -v $PWD:/root/bmnn devops.io:12580/lgy/test/gddeploy/bmnn/build/3.0:v0.2
```

### 3） 编译

```bash
cd /root/bmnn
bash script/build.sh
```


## 2.2 运行
拷贝SDK到目标机器上，运行脚本

```bash
# 声明环境变量
source script/env.sh
```

如果存在编译的系统库和运行自带的C++std库版本不兼容问题，建议运行容器环境，运行容器环境的命令如下：

```bash 
docker run -it --name test --privileged -v $PWD:/workspace -v /system/:/system ubuntu:20.04
```
进入容器后，运行脚本

```bash
cd /workspace/
source script/env.sh
```

运行工具获取gtx文件
```bash
./tools/gxt_maker
# 将打印：
# Usage: ./tools/gxt_maker [gxt_file_path]
# SN:HQDZKM6BJAABF0641, UUID :20184e8c-cb15-3014-6b06-558b215168ab 
# Gtx file will be save in: .//20184e8c-cb15-3014-6b06-558b215168ab.gxt
```
请将得到的gxt文件上传到共达地训练平台，得到model.gem和license文件，可以将model.gem和license文件放到data/models/目录下


运行和测试样例
```bash
# 查看pic用法
./bin/sample_runner_pic -h
# Options:
#   -h [ --help ]         Help screen
#   --model arg           model file path
#   --license arg         model license file path
#   --pic-path arg        pic file full path or just pic path
#   --save-pic arg        save file path

# 运行样例
./bin/sample_runner_pic --model ./data/models/model.gem --license ./data/models/license --pic-path ./data/pic/baidu_person/images/ --save-pic ./data/pic/baidu_person/preds/

# 查看video用法
./bin/sample_runner_video -h
# Options:
#   -h [ --help ]           Help screen
#   --model arg             model file path
#   --license arg           model license file path
#   --video-path arg        video file path
#   --multi-stream arg (=1) multi stream
#   --is-save arg (=1)      is save result pic
#   --save-pic arg          model file path
# 注意：由于边解码边保存图片会造成资源浪费，--save-pic功能目前没有开放，用户需要有需要可以修改sample/sample_runner_video.cpp文件后重新编译

# 运行样例
./bin/sample_runner_video --model ./data/models/model.gem --license ./data/models/license --video-path rtsp://admin:gddi1234@10.13.0.104:554/h264/ch1/main/av_stream --multi-stream 4 --is-save 0 --save-pic .
``` 