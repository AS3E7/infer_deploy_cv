所有和设备相关的注册都放到这

文件夹级别：
    model:  新算法，主要传入config构建推理的pipeline
    device:
        cpu     默认
        厂商1
            设备1
            设备2
        厂商2
            设备1

需要注册的内容：
1）模型信息
loadModel
modelinfo get的方式和接口
model handle句柄
2）前处理CV接口
3）后处理接口
4）device类接口->对接core/device.cpp
5）mem接口->对接mem.cpp
6）视频编解码接口

这里需要准备好一份模板，只有所有的接口类，函数内容为空，对接只需要实现函数内容说明

