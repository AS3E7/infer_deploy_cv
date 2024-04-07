# 1. Windows下编译

## 1.1 安装

* vcpkg

```shell
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Debug \
      -DCMAKE_TOOLCHAIN_FILE=d:/vcpkg/scripts/buildsystems/vcpkg.cmake \
      ..
```

## 已知问题

 - [x] build目录不可以是二级目录，`vcpkg`会找不到`vcpkg.json`文件
 - [ ] `当前libhv默认的vcpkg版本编译进去后，不能正常的提供http服务`