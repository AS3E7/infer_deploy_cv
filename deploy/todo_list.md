# 待做
## app层
- [ ] 新增python接口
- [ ] 视频推理，增加ringbuffer，减少内存占用
- [ ] app层所有设备相关接口，迁移到common进行封装
- [ ] APP层需要封装avframe和cv::Mat的转换接口
- [ ] 删除sample中使用boost

## core层
- [ ] 注册方式修改，修改为不需要在core上注册，采用宏定义注册
- [ ] 注册方式修改，改为不实用map静态变量，减少core大小
- [ ] 思考多模型串联，第二模型如何输入的问题和保证效率的问题
- [ ] 新增视频编解码接口
- [ ] 新增图片编解码接口

## 设备层
- [ ] 各个芯片厂商新增视频编解码接口实现
- [ ] 各个芯片厂商新增图片推理接口实现

## 其他
- [ ] 完善release必要部分，编译脚本、readme、license等
- [ ] 完善文档，包括接口文档、使用文档、编译文档等
- [ ] 完善单元测试，包括core层、app层、设备层的单元测试
- [ ] 完善性能测试，包括core层、app层、设备层的性能测试
- [ ] 完善demo，包括core层、app层、设备层的demo
- [ ] 完善benchmark，包括core层、app层、设备层的benchmark
- [ ] 完善ci，包括core层、app层、设备层的ci
- [ ] 完善docker和安装脚本