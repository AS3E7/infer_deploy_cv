#include "gddeploy.h"


int main(int argc, char *argv[]) 
{
    // 1. infer server
    std::unique_ptr<gddeploy::InferServer> gddeploy;
    gddeploy.reset(new gddeploy::InferServer(0));

    // 2. desc
    // gddeploy::SessionDesc desc;
    // desc.strategy = gddeploy::BatchStrategy::STATIC;
    // desc.engine_num = 1;
    // desc.priority = 0;
    // desc.show_perf = true;
    // desc.name = "detection session";

    // // 3. load offline model
    // desc.model = gddeploy::InferServer::LoadModel(model_path, func_name);
    // // set preproc and postproc
    // desc.preproc = gddeploy::video::PreprocessorMLU::Create();
    // desc.postproc = gddeploy::Postprocessor::Create();
    // gddeploy::Session_t session_ = gddeploy_->CreateSyncSession(desc);

    // // 4. 推理
    // bool ret = gddeploy_->RequestSync(session_, std::move(in), &status, out);

    printf("over\n");

    return 0;
}

