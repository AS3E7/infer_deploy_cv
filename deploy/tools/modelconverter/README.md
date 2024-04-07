优化器
在转模型之前，做一些优化方案，包括不限于：
图优化
修改onnx，添加sigmoid，concat output等操作

由于有些硬件直接提供docker镜像而没有提供dockerfile，不利于集成。
需要采用分分布式方式，多个docker容器负责不同的硬件模型转换，创建容器的时候均映射到/gddeploy目录，默认启动里面的tools/convert_service.py文件，传入不同的端口号区分。
收到转换请求时候，如果发现不是自己支持的硬件平台，转发到对应端口的容器去继续请求回复。


# 本目录包含
thirdparty：多种算法源码.采用gitlab源码
dockerfile包含miniconda环境安装，四种算法的环境安装
export.py：导出jit、onnx等模型
change_graph.py：修改部分图结构，包括最后的concat等操作
optimizer：图优化

# 结构
convert_service.py  //接受转换请求服务
model_converter.py  //转换模型执行，主要是调用对于厂商执行
export.py   //导出模型，包括onnx、jit

|-infer //多种硬件推理，定义基类，实现在各个硬件文件夹的infer.py，如果没有则使用torch/onnxruntime推理
|-preproc   //多种算法前处理，定义基类，实现基本是numpy的OpenCV，也可torch.transforms，也可以cuda实现的加速库
|-postproc  //多种算法后处理，定义基类，实现各个算法的后处理
|-bmnn
|---infer.py  //推理代码
|---convert_fp32.py  //转换fp32模型
|---convert_int8.py  //转换int8模型
|-ascend
....每个厂商每个工具链独立一个文件夹