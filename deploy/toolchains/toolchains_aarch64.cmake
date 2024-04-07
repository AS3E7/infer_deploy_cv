set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR armv8)

# set(CMAKE_C_COMPILER /usr/bin/aarch64-linux-gnu-gcc-11)
# set(CMAKE_CXX_COMPILER /usr/bin/aarch64-linux-gnu-g++-11)
set(CMAKE_C_COMPILER /opt/gcc-linaro-7.5.0-2019.12-x86_64_aarch64-linux-gnu/bin/aarch64-linux-gnu-gcc)
set(CMAKE_CXX_COMPILER /opt/gcc-linaro-7.5.0-2019.12-x86_64_aarch64-linux-gnu/bin/aarch64-linux-gnu-g++)

# set(CMAKE_C_COMPILER /opt/gcc-linaro-7.4.1-2019.02-x86_64_aarch64-linux-gnu/bin/aarch64-linux-gnu-gcc)
# set(CMAKE_CXX_COMPILER /opt/gcc-linaro-7.4.1-2019.02-x86_64_aarch64-linux-gnu/bin/aarch64-linux-gnu-g++)

# set(CMAKE_C_COMPILER /volume1/gddi-data/lgy/cambricon/toolkits_220/toolchains/aarch64/gcc-linaro-6.2.1-2016.11-x86_64_aarch64-linux-gnu/bin/aarch64-linux-gnu-gcc)
# set(CMAKE_CXX_COMPILER /volume1/gddi-data/lgy/cambricon/toolkits_220/toolchains/aarch64/gcc-linaro-6.2.1-2016.11-x86_64_aarch64-linux-gnu/bin/aarch64-linux-gnu-g++)

# ts toolchains
# set(TOOLSCHAINS_PATH /volume1/gddi-data/lgy/ts/smart_box_SDK/TX536_T8_V1R020C030SP01/prebuilts/host/gcc/gcc-arm-10.2-2020.11-x86_64-aarch64-none-linux-gnu)
# set(CMAKE_C_COMPILER ${TOOLSCHAINS_PATH}/bin/aarch64-none-linux-gnu-gcc)
# set(CMAKE_CXX_COMPILER ${TOOLSCHAINS_PATH}/bin/aarch64-none-linux-gnu-g++)

set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)

# set(CMAKE_SYSROOT /usr/local/aarch64-mlu220-soc)
# set(CMAKE_FIND_ROOT_PATH /usr/local/aarch64-mlu220-soc)
# set(ENV{PKG_CONFIG_PATH} /usr/local/aarch64-mlu220-soc/lib/pkgconfig)

# set(TARGET_CHIP mlu220)