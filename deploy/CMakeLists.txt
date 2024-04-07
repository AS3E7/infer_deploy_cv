cmake_minimum_required (VERSION 3.20)

project(gddeploy)

# if (CMAKE_BUILD_TYPE STREQUAL "")
#     message(STATUS "CMAKE_BUILD_TYPE not defined, 'Release' will be used")
#     set(CMAKE_BUILD_TYPE "Release")
# endif()

# if(CMAKE_COMPILER_IS_GNUCXX)
#     set(CMAKE_CXX_FLAGS "-std=c++11 ${CMAKE_CXX_FLAGS}")
#     set(CMAKE_CXX_STANDARD 17)
# endif(CMAKE_COMPILER_IS_GNUCXX)

set(CMAKE_CXX_FLAGS   "-std=c++17")             # c++17
# set(CMAKE_CXX_FLAGS   "-fPIC")             # c++17
# set(CMAKE_CXX_FLAGS   "Wl,-Bsymbolic")                     # 调试信息
# set(CMAKE_CXX_FLAGS_DEBUG   "-g")                     # 调试信息
# set(CMAKE_CXX_FLAGS   "-Wall")                  # 开启所有警告
# set(CMAKE_CXX_FLAGS_DEBUG   "-O0" )             # 调试包不优化
# set(CMAKE_CXX_FLAGS_RELEASE "-O2 -DNDEBUG " )   # release包优化
# set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fsanitize=address -fno-omit-frame-pointer -g -O0")
# set(CMAKE_POSITION_INDEPENDENT_CODE ON)

#从文件中读取版本信息,将内容读入 PROGRAM_VERSION 变量中，该变量会去掉尾部的空格。当设置了变量 PROGRAM_VERSION ，就可以使用它来配置 version.hpp.in ，生成 generated/version.hpp 
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/VERSION")
  file(READ "${CMAKE_CURRENT_SOURCE_DIR}/VERSION" PROJECT_VERSION)
  string(STRIP "${PROJECT_VERSION}" PROJECT_VERSION)
else()
  message(FATAL_ERROR "File ${CMAKE_CURRENT_SOURCE_DIR}/VERSION not found")
endif()
message(STATUS "Program version: ${PROJECT_VERSION}")
 
# generate file version.hpp based on version.hpp.in
configure_file(
  version.h.in
  ${CMAKE_CURRENT_SOURCE_DIR}/include/version.h
  @ONLY
)
# 这里还要添加很多编译选项
option(BUILD_MODCRYPT "Build thirdparty modcrypt" OFF)
option(BUILD_TEST "Build test" ON)
option(BUILD_SAMPLE "Build sample" ON)
option(BUILD_ORT "Build onnx runtime" OFF)
option(BUILD_TARGET_CHIP "Build target chip" "bmnn")

message(STATUS "BUILD_TARGET_CHIP: ${BUILD_TARGET_CHIP}")

#这里添加lib和include的目录，或者find_package
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)

# 根据编译链的不同，设置不同的编译选项
# 如果是交叉编译链，设置TARGET_ARCH变量为"aaarch64"；如果是x86编译链，设置TARGET_ARCH变量为"x86"
message(${CMAKE_CXX_COMPILER})
if (CMAKE_CXX_COMPILER MATCHES "aarch64")
  message("Build on aarch64")
  set(TARGET_ARCH "aarch64")
else() #if (CMAKE_CXX_COMPILER MATCHES "x86")
  message("Build on x86")
  set(TARGET_ARCH "x86")
endif()

if (BUILD_MODCRYPT)
    message("Build test thirdparty modcrypt")
    add_subdirectory(thirdpart)
endif()

file(GLOB CMakeFiles "cmake/*.cmake")
foreach (cmake_module_file IN LISTS CMakeFiles)
    # get_filename_component(CMakeName ${cmake_module_file} NAME_WLE)
    get_filename_component(CMakePath ${cmake_module_file} DIRECTORY)
    message("LoadCMake: ${CMakeName}")
    include(${cmake_module_file})
endforeach (cmake_module_file)

add_subdirectory(src)

if (BUILD_TEST)
    message("Build test folder")
    add_subdirectory(test/cpp)
endif()


if (BUILD_SAMPLE)
    message("Build sample folder")
    add_subdirectory(sample)
endif()

add_subdirectory(tools/modcrypt)
# add_subdirectory(python)
add_subdirectory(release)

# 添加编译目标，src里面的core、model、common单独编译
# if (NOT DEFINED CMAKE_INSTALL_PREFIX)
set(CMAKE_INSTALL_PREFIX "${CMAKE_SOURCE_DIR}/install" CACHE PATH "Gddeploy's installation directory")
# endif ()
message(STATUS "CMAKE_INSTALL_PREFIX: ${CMAKE_INSTALL_PREFIX}")

install(DIRECTORY ${CMAKE_SOURCE_DIR}/include/core DESTINATION include)