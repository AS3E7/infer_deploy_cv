#include "core/pipeline.h"
#include "core/alg.h"
#include "core/infer_server.h"

using namespace gddeploy;

Pipeline *Pipeline::pInstance_ = nullptr;


int Pipeline::CreatePipeline(std::string config, ModelPtr model, std::vector<ProcessorPtr> &processors)
{
    // TODO: 新增解析config
    auto model_info_priv = model->GetModelInfoPriv();
    std::string model_type = model_info_priv->GetModelType();
    std::string net_type   = model_info_priv->GetNetType();
    
    AlgCreator* alg_creator = AlgManager::Instance()->GetAlgCreator(model_type, net_type);
    AlgPtr alg_ptr = alg_creator->Create();

    alg_ptr->Init(config, model);    
    alg_ptr->CreateProcessor(processors);

    return 0;
}


int AddProcessor(std::vector<Processor> &base, std::vector<Processor> &new_processors)
{
    return 0;
}