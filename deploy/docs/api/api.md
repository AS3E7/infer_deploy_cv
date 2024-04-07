# API接口说明

API接口分为三种形式
1）获取processor，分别是前后处理和推理模块，客户自行构建多线程pipeline
2）获取Infer_server，session的方式
3）使用inferSync、inferAsync接口，直接输入解码后的帧数据

## processor方式

