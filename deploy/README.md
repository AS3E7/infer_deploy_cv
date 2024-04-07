# 简介
    gddeploy为共达地针对部署场景开发一套的推理引擎项目，具备多后端、多算法、高效利用硬件资源等功能。

框架解析
![alt text](docs/project/%E6%B5%81%E7%A8%8B%E5%9B%BE.jpg)
APP层：包含可以运行简约接口sample 
API层：包含三个不同层次接口 
Core层：包含Session/Engine/Pipeline等主要核心功能 
Registry层：算法注册和设备注册 

# 功能介绍

- 对接多算法类型，包括不限于：分类、检测、姿态、分割、动作
- 对接多后端芯片，包括不限于：NV、Ascend、RK、Cambricon、Sophon、Intel、Sigmastar
- 对接多类型上层请求接口
- 高效的推理引擎，具有多模型组合、异步调度、并发推理、动态批量、多卡多流推理、请求缓存等特性
- 简易构建pipeline，灵活添加各种后处理模块，串并联多模型
- 多语言支持，支持C++、python、rust、Java接口
- 完善sample、文档说明


# 使用说明
- 编译说明，docker部署
- Get start
- 示例代码
- benchmark评测


# 目前支持框架
表格：
分硬件、算法
| 硬件平台 | 分类 | 检测 | 姿态 | 分割 |
| :-----| :----: | :----: | :----: | :----: |
| sophon | [x] | [x] | [x] | [x] |
| nvidia | [ ] | [ ] | [ ] | [ ] |


# Roadmap
