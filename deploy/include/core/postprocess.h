#ifndef GDDEPLOY_POSTPROC_H
#define GDDEPLOY_POSTPROC_H

#include <algorithm>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "core/infer_server.h"
#include "core/processor.h"
#include "core/model.h"

namespace gddeploy
{
    
/*
配置信息
UserConfig：用户额外设置参数
ModelConfig：解析模型得到的模型参数信息

基类preproc


创造类preproc_creator{}


各个设备都要自己的前处理加速api实现接口，提供对应的crop、resize、transform等接口，


*/


/*------------------------------------------------------------------------*/
/*
 config字符串样例：严格要求顺序
 {
     "input_data_type" : "NV12",
     "resize" : 640,
     "norm" : true;
     "std" : [234, 224, 224],
     "div" : [114, 114, 114],
     "pad" : true,
     "pad_value" : 114,
 }
 */
class PostProcConfig{
public:
    PostProcConfig(std::string config);   //解析字符串的config，到对应的变量

    std::map<std::string, std::string> GetOps(){return ops_;}

    void PrintConfig();

private:
    //添加上面的字段成员变量
    std::map<std::string, std::string> ops_;

};

using PostProcConfigPtr = std::shared_ptr<PostProcConfig>;


//输入输出以PackPtr作为参数类型
class PostProc : public ProcessorForkable<PostProc>{
public:
    PostProc() = default;
    ~PostProc() = default;
    PostProc(const std::string& type_name): ProcessorForkable<PostProc>(type_name){}
    // explicit PostProc(const std::string& type_name) noexcept : ProcessorForkable(type_name) {}
    virtual Status Init() noexcept override{ return gddeploy::Status::SUCCESS; };
    //一般是解析config， 得到处理算子列表，调用对应的creator创建对象， 申请资源，编排处理流程
    virtual Status Init(std::string config){ return Status::SUCCESS; }
    virtual Status Init(ModelPtr model, std::string config){ return Status::SUCCESS; }
    virtual Status Init(std::string product, std::string chip, std::string config) { return Status::SUCCESS;  } 

    virtual Status Process(PackagePtr data) noexcept override{ return gddeploy::Status::SUCCESS; }

private:
    std::string product_;
    std::string chip_;
    std::shared_ptr<PostProcConfig> config_;
};

//后面应该有几个常见的算法前处理配置文件继承，上面只需要传入product、chip、net_type就可以确认配置信息
using PostProcPtr = std::shared_ptr<PostProc>;


class PostProcCreator{
public:
    PostProcCreator(){}
    PostProcCreator(std::string name):postproc_creator_name_(name){}
    PostProcCreator(PostProcConfig *config){} //根据config配置信息创建对象，主要是有crop等需求的第二模型前处理
    ~PostProcCreator(){}

    virtual PostProcPtr Create() = 0;

private:
    std::string postproc_creator_name_;
};


class PostProcManager{
public: 
    // PostProcManager();

    static PostProcManager* Instance() noexcept{
        if (pInstance_ == nullptr){
            pInstance_ = new PostProcManager();
        }
        return pInstance_;
    }

    // void ClearCache() noexcept;

    // int CacheSize() noexcept;

    // std::shared_ptr<PostProc> GetPostproc(PostProcConfigPtr config, ModelPtr model) noexcept;
    PostProcCreator* GetPostProcCreator(std::string manufacturer, std::string chip){
        if (postproc_creator_.count(manufacturer) == 0)
            return nullptr;
        
        if (postproc_creator_[manufacturer].count("any") != 0)
            return postproc_creator_[manufacturer]["any"];

        return postproc_creator_[manufacturer][chip];
    }

    int register_postproc(std::string manufacturer, std::string chip, PostProcCreator *postprocs_creator) noexcept{
        std::unordered_map<std::string, PostProcCreator*> chip_postprocs_creator;
        chip_postprocs_creator[chip] = postprocs_creator;
        postproc_creator_[manufacturer] = chip_postprocs_creator;

        return 0;
    }

private:
    // static std::unordered_map<std::string, std::shared_ptr<PostProc>> postprocs_cache_;
    // static std::mutex postprocs_cache_mutex_;

    static PostProcManager *pInstance_;
    std::unordered_map<std::string, std::unordered_map<std::string, PostProcCreator*>> postproc_creator_;
};

// template <typename CreatorType>
// class PostprocRegisterer {
// public:
//     PostprocRegisterer() { PostProcManager::register_postproc(manu_, chip_, &inst_); }

// private:
//     std::string manu_;
//     std::string chip_;
//     CreatorType inst_;
// };


// #define REGISTER_POSTPROCESS_CREATOR(manufacturer, chip, postprocs_creator)   \
//     static PostprocRegisterer<postprocs_creator> g_register_##manufacturer_##chip_##postprocs_creator{};


int register_postproc_module();

} // namespace gddeploy

#endif

