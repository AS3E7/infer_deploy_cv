



统一读取模型，解密模型，读取模型信息，保存模型信息传到下去

device类：管理硬件资源

model类：管理模型

session类：会话类，管理各种资源，相当于以上的每个部分都有一个指针在这边放着，随时可以取

engine：推理逻辑

request类：对外请求推理服务

cache类：缓存多帧图片，凑batch用


engine类：一般包含pre/infer/post的处理单元，为了防止频繁的创建和销毁这些单元，根据硬件能力预先创建，需要的时候再按需申请
注意：扩展时资源重配置


pre/infer/post处理单元代码
主要参考mmdeploy的模块化设置

preprocess中： 
    cpu及其对应的设备里面只负责注册处理小单元(crop/resize/transform)，
    preprocess，基类
    具体算法从处理小单元中自由组合，注册preprocess类