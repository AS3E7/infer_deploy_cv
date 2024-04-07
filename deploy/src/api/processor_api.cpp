#include <memory>

#include "core/model.h"
#include "api/processor_api.h"
#include "core/pipeline.h"
#include "core/register.h"

using namespace gddeploy;

namespace gddeploy{

//api的接口根据解析配置文件和模型文件，获取前中后处理单元
class ProcessorAPIPriv{
public:
    void Init(std::string config, std::string model_path, std::string license = "");

    // 根据模型获取processor，用于最基础层的接口
    std::vector<ProcessorPtr> GetProcessor(){
        return processors_;
    }

    // 用于添加更多的processor再后面，再设置进去
    int SetProcessor(std::vector<ProcessorPtr> processors);

    std::string GetModelType();
    std::vector<std::string> GetLabels();


private:
    std::vector<ProcessorPtr> processors_;
    ModelPtr model_;
};
} 

void ProcessorAPIPriv::Init(std::string config, std::string model_path, std::string license)
{
    //载入model
    std::string properties_path = "";
    model_ = gddeploy::ModelManager::Instance()->Load(model_path, properties_path, license, "");

    // 创建processors
    Pipeline::Instance()->CreatePipeline(config, model_, processors_);
}

int ProcessorAPIPriv::SetProcessor(std::vector<ProcessorPtr> processors){
    // for (auto processor : processors){
    //     processors_.emplace_back(processor);
    // }
    return 0;
}

std::string ProcessorAPIPriv::GetModelType()
{
    auto model_info_priv = model_->GetModelInfoPriv();
    return model_info_priv->GetModelType();
}

std::vector<std::string> ProcessorAPIPriv::GetLabels()
{
    ModelPropertiesPtr mp = model_->GetModelInfoPriv();
    
    return mp->GetLabels();
}

ProcessorAPI::ProcessorAPI(){
    priv_ = std::make_shared<ProcessorAPIPriv>();
}

void ProcessorAPI::Init(std::string config, std::string model_path, std::string license)
{
    
    priv_->Init(config, model_path, license);
}

std::vector<ProcessorPtr> ProcessorAPI::GetProcessor()
{
    return priv_->GetProcessor();
}

std::string ProcessorAPI::GetModelType()
{
    return priv_->GetModelType();
}

std::vector<std::string> ProcessorAPI::GetLabels()
{
    return priv_->GetLabels();
}