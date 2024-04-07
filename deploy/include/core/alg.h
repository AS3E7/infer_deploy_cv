#pragma once

#include <algorithm>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "core/infer_server.h"
#include "core/model.h"

namespace gddeploy
{
/*
 本文件包含的类和功能说明：
 AlgConfig：算法配置
 Alg：算法基类
 AlgCreator：创建算法类
 AlgManager：算法类管理者
 AlgRegister：算法注册

 编写思路：
    1. Alg作为全部算法的基类，包含共有属性，包括name/algType等
    2. AlgConfig目前没有支持太多配置，主要给模型串联预留配置
    3. 继承类层次：
        Alg ->
            ->ClassifyAlg
            ->DetectAlg
            ->PoseAlg
            ->SegAlg
            ->Detect3DAlg
    4. 继承类需要完成的工作：
        1）按需要继承AlgConfig，添加客制化配置解析
        2）继承AlgCreator，继承Alg，复写对应类方法
        3）注册
    5. AlgManger根据model_type获取对应的Processor对象，串联形成pipeline的vector结构
 */


/*
 config字符串样例：
 {
    "pre":{
        "format":"RGB",
        "mean":[123.675, 116.28, 103.53],
        "std":[58.395, 57.12, 57.375],
        "keep_aspect_ratio":1,
    },
    "infer":{
        "func_name":"subnet0",
    },
    "post":{
        "iou_thresh" : 0.6,
        "conf_thresh" : 0.5,
        "anchor":[10.0, 13.0, 16.0, 30.0, 33.0, 23.0, 30.0, 61.0, 62.0, 45.0, 59.0, 119.0, 116.0, 90.0, 156.0, 198.0, 373.0, 326.0],
     }
 }
 */
class AlgConfig{
public:
    AlgConfig(){}
    AlgConfig(std::string config);   //解析字符串的config，到对应的变量

    std::map<std::string, std::string> GetConfig(){return config_;}

    void PrintConfig();

private:
    //添加上面的字段成员变量
    std::map<std::string, std::string> config_;

};


// class Alg
//输入输出以PackPtr作为参数类型
class Alg{
public:
    Alg(){}
    ~Alg(){}
    virtual int Init(std::string config, ModelPtr model) = 0; //一般是解析config， 得到处理算子列表，调用对应的creator创建对象， 申请资源，编排处理流程
    virtual int CreateProcessor(std::vector<ProcessorPtr> &processors) = 0;  //根据配置，形成Processor为单元的pipeline

protected:
    std::string model_type_;
    std::string net_type_;
    std::shared_ptr<AlgConfig> config_;
    ModelPtr model_;
    std::vector<gddeploy::ProcessorPtr> processors_; //按顺序解析运行
};

using AlgPtr = std::shared_ptr<Alg>;

// class AlgCreator
class AlgCreator{
public :
    AlgCreator(std::string name):name_(name){}
    virtual AlgPtr Create() = 0;
private:
    std::string name_;
};
using AlgCreatorPtr = std::shared_ptr<AlgCreator>;

// class AlgManager
class AlgManager{
public :
    static AlgManager* Instance() noexcept{
        if (pInstance_ == nullptr){
            pInstance_ = new AlgManager();
        }
        return pInstance_;
    }

    AlgCreator* GetAlgCreator(const std::string& model, const std::string& net) noexcept{
        for (auto &item : alg_creator_){
            if (item.first == model){
                std::map<std::string, AlgCreator*> tmp = item.second;
                if (tmp.find(net) != tmp.end()){
                    return tmp[net];
                }
            }
        }
        return nullptr;
    }

    int RegisterAlg(std::string model, std::string net, AlgCreator *alg_creator) noexcept{
        std::map<std::string, AlgCreator*> net_model_creator;
        net_model_creator[net] = alg_creator;
        alg_creator_.emplace_back(std::make_pair(model, net_model_creator));

        return 0;
    }
private:
    static AlgManager *pInstance_;
    std::vector<std::pair<std::string, std::map<std::string, AlgCreator*>>> alg_creator_;
};

// template <typename CreatorType>
// class AlgRegisterer {
// public:
//     AlgRegisterer() { AlgManager::RegisterModel(model_type_, net_type_, &inst_); }

// private:
//     std::string model_type_;
//     std::string net_type_;
//     CreatorType inst_;
// };

// #define REGISTER_ALG_CREATOR(model, net, alg_creator)   \
//     static AlgRegisterer<alg_creator> g_register_##model_##net_##alg_creator{};

int register_alg_module();

}