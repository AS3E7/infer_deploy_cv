#ifndef GDDEPLOY_PREPROCESS_H
#define GDDEPLOY_PREPROCESS_H

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

#if 0
//非常细的处理算子，抽象类
class PreProcOp{
public :
    virtual int Init(std::string config) = 0; //算子的输入参数
    virtual int Process() = 0;  //算子处理

private:

};

using PreProcOpPtr = std::shared_ptr<PreProcOp>;

//非常细的处理算子的Creator，抽象类
//未来的集成级别：<product><chip><op_type>
class PreProcOpCreator{
public :
    PreProcOpCreator(std::string name):name_(name){}
    virtual PreProcOpPtr Create() = 0;
private:
    std::string name_;
};
using PreProcOpCreatorPtr = std::shared_ptr<PreProcOpCreator>;

class PreProcOpManager{
public :
    static PreProcOpManager* Instance() noexcept{
        static PreProcOpManager m;
        return &m;
    }

    PreProcOpCreator* GetPreProcOpCreator(const std::string& manufacturer, const std::string& chip, const std::string& op) noexcept{
        return preproc_op_creator_[manufacturer][chip][op];
    }

    static int RegisterModel(std::string manufacturer, std::string chip, std::string op, PreProcOpCreator *model_creator) noexcept{
        std::unordered_map<std::string, PreProcOpCreator*> op_creator;
        op_creator[op] = model_creator;
        std::unordered_map<std::string, std::unordered_map<std::string, PreProcOpCreator*>> chip_model_creator;
        chip_model_creator[chip] = op_creator;
        preproc_op_creator_[manufacturer] = chip_model_creator;

        return 0;
    }
private:
    static std::unordered_map<std::string, std::unordered_map<std::string, std::unordered_map<std::string, PreProcOpCreator*>>> preproc_op_creator_;
};

template <typename CreatorType>
class PreProcOpRegisterer {
public:
    PreProcOpRegisterer() { PreProcOpManager::RegisterModel(manu_, chip_, op_, &inst_); }

private:
    std::string manu_;
    std::string chip_;
    std::string op_;
    CreatorType inst_;
};

#define REGISTER_PREPROCOP_CREATOR(manufacturer, chip, op, model_creator)   \
    static PreProcOpRegisterer<model_creator> g_register_##manufacturer_##chip_##model_creator{};
#endif
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
class PreProcConfig{
public:
    PreProcConfig(std::string config);   //解析字符串的config，到对应的变量

    std::map<std::string, std::string> GetOps(){return ops_;}

    void PrintConfig();

private:
    //添加上面的字段成员变量
    std::map<std::string, std::string> ops_;

};

using PreProcConfigPtr = std::shared_ptr<PreProcConfig>;


//输入输出以PackPtr作为参数类型
class PreProc : public ProcessorForkable<PreProc>{
// class PreProc {
public:
    PreProc() = default;
    PreProc(const std::string& type_name): ProcessorForkable<PreProc>(type_name){}
    virtual Status Init() noexcept override{ return gddeploy::Status::SUCCESS; };
    //一般是解析config， 得到处理算子列表，调用对应的creator创建对象， 申请资源，编排处理流程
    virtual Status Init(std::string config){ return Status::SUCCESS; }
    virtual Status Init(ModelPtr model, std::string config){ return Status::SUCCESS; }
    virtual Status Init(std::string product, std::string chip, std::string config) { return Status::SUCCESS;  } 

    virtual Status Process(PackagePtr data) noexcept override; //noexcept override;

private:

    std::string product_;
    std::string chip_;

    std::shared_ptr<PreProcConfig> config_;
    // std::vector<PreProcOp> ops_; //按顺序解析运行
};

//后面应该有几个常见的算法前处理配置文件继承，上面只需要传入product、chip、net_type就可以确认配置信息
using PreProcPtr = std::shared_ptr<PreProc>;


class PreProcCreator{
public:
    PreProcCreator(){}
    PreProcCreator(std::string name):preproces_creator_name_(name){}
    // PreProcCreator(PreProcConfig *config){} //根据config配置信息创建对象，主要是有crop等需求的第二模型前处理
    ~PreProcCreator(){};

    virtual PreProcPtr Create() = 0;

    std::string GetCreatorName(){ return preproces_creator_name_; }

private:
    std::string preproces_creator_name_;
};

// int register_preproc();

class PreProcManager{
public: 
    PreProcManager(){}
    ~PreProcManager(){
        if (pInstance_ != nullptr){
            delete pInstance_;
            pInstance_ = nullptr;
        }
    }

    static PreProcManager* Instance() noexcept{
        if (pInstance_ == nullptr){
            pInstance_ = new PreProcManager();
        }
        return pInstance_;
    }

    void ClearCache() noexcept{}

    int CacheSize() noexcept{ return 0;}

    PreProcCreator* GetPreProcCreator(std::string manufacturer, std::string chip) noexcept{
        if (preproc_creator_.count(manufacturer) == 0)
            return nullptr;
        
        if (preproc_creator_[manufacturer].count("any") != 0)
            return preproc_creator_[manufacturer]["any"];
        return preproc_creator_[manufacturer][chip];
    }

    int register_preproc(std::string manufacturer, std::string chip, PreProcCreator *preproc_creator) noexcept{
        //TODO: 判断是否已经注册
        std::unordered_map<std::string, PreProcCreator*> chip_preproc_creator;
        chip_preproc_creator[chip] = preproc_creator;
        preproc_creator_[manufacturer] = chip_preproc_creator;

        return 0;
    }

    std::string GetName(){
        return name_;
    }

private:
    // static std::unordered_map<std::string, std::shared_ptr<PreProc>> preproc_cache_;
    // static std::mutex preproc_cache_mutex_;
    std::unordered_map<std::string, std::unordered_map<std::string, PreProcCreator*>> preproc_creator_;

    std::string name_;
    static PreProcManager *pInstance_;
};

int register_preproc_module();

// template <typename CreatorType>
// class PreProcRegisterer {
// public:
//     PreProcRegisterer(){ 
//         std::cout << "Register 2" << std::endl;
//     }
//     PreProcRegisterer(std::string manu, std::string chip){ 
//         std::cout << "Register " << std::endl;
//         PreProcManager::register_preproc(manu, chip, &inst_); 
//     }

// private:
//     // std::string manu_;
//     // std::string chip_;
//     CreatorType inst_;
// };


// // void register_preproc_list(std::function<int()> func);
// #define REGISTER_PREPROCESS_CREATOR(manufacturer, chip, preproc_creator)   \
//     static PreProcRegisterer<preproc_creator> g_register_##manufacturer_##chip_##preproc_creator(manufacturer, chip);
// #define REGISTER_PREPROCESS_CREATOR(manufacturer, chip, preproc_creator)   \
//     void register_preproc_creator(){            \
//         PreProcManager::Instance()->register_preproc(manufacturer, chip, new preproc_creator());  \
//     }   \
//     register_preproc_list(register_preproc_creator);\



} // namespace gddeploy

#endif

