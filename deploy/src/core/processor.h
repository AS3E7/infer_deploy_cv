
#ifndef GDDEPLOY_PROCESSOR_H
#define GDDEPLOY_PROCESSOR_H

#include <string>

namespace gddeploy
{
    
/*
配置信息
UserConfig：用户额外设置参数
ModelConfig：解析模型得到的模型参数信息
*/


//处理单元最基类，pre/post/infer都要继承这个类
//engine调用的处理单元来说，只会调用Processor，对应mmdeploy就是Module的概念
// class Processor{
// public:
//     virtual int Init() = 0; //一般是解析config，申请资源
//     virtual int Process() = 0;  //处理逻辑代码

//     virtual std::shared_ptr<Processor> Fork() = 0;

//     const std::string& TypeName() const noexcept { return type_name_; }

// private:
//     std::unique_lock<std::mutex> Lock() noexcept { return std::unique_lock<std::mutex>(process_lock_); }
//     std::string type_name_;
//     std::mutex process_lock_;
// };

} // namespace gddeploy

#endif