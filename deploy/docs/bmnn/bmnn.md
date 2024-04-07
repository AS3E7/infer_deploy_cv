
# 环境搭建
交叉编译链使用6.3-2017.05
下载链接：https://releases.linaro.org/components/toolchain/binaries/6.3-2017.05/aarch64-linux-gnu/gcc-linaro-6.3.1-2017.05-x86_64_aarch64-linux-gnu.tar.xz
已经安装路径：/opt/gcc-linaro-6.3.1-2017.05-x86_64_aarch64-linux-gnu/bin/aarch64-linux-gnu-g++
注意：7.5一般兼容，但是过新会存在libstdc++.so库的符号表和盒子不匹配的情况，比如以下错误：
“/usr/lib/aarch64-linux-gnu/libstdc++.so.6: version `CXXABI_1.3.11' not found (required by”



