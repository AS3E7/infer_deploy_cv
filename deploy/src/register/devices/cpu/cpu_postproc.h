#ifndef GDDEPLOY_CPUPOSTPROC_H_
#define GDDEPLOY_CPUPOSTPROC_H_

#include <vector>
#include <memory>
#include <string>
#include "core/postprocess.h"
#include "postproc/util/common_def.h"

namespace gddeploy{
    
class CpuPostProcPriv;
class CpuPostProc : public PostProc {
public:
    CpuPostProc() : PostProc("cpu_postproc"){}
    // CpuPostProc(){};
    // Status Init() noexcept override;
    // Status Init(std::string config) noexcept override{}

    Status Init(ModelPtr model, std::string config) override;

    Status Process(PackagePtr data) noexcept override;
    std::shared_ptr<Processor> Fork() override
    {
      auto p = std::make_shared<CpuPostProc>();
      p->CopyParamsFrom(*this);
      if (p->Init(model_, "") != Status::SUCCESS)
        return nullptr;
      return p;
    }

private:
    ModelPtr model_;
    // cv::Mat resize_mat_;
    std::shared_ptr<CpuPostProcPriv> priv_;
};


class CpuPostprocCreator : public PostProcCreator{
public:
    CpuPostprocCreator():PostProcCreator("CPUPostProcCreator"){
    }

    PostProcPtr Create() override {
        return std::make_shared<CpuPostProc>();
    }
    // std::string GetName() const override { return model_creator_name_; }

    // std::shared_ptr<Model> Create(const any& value) override {
    //     return std::shared_ptr<OrtModel>(new OrtModel(value));
    // }

private:
    std::string postproc_creator_name_ = "CPUPostProcCreator";

};  // class OrtModelCreator

}

#endif