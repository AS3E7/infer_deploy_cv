#pragma once

#include <memory>
#include <string>
#include "core/preprocess.h"

namespace gddeploy{
    
class CPUPreProcPriv;
class CPUPreProc : public PreProc {
public:
    CPUPreProc() noexcept : PreProc("cpu_preproc") {}

    // Status Init() noexcept override;
    Status Init(std::string config) noexcept override;

    Status Init(ModelPtr model, std::string config) override;

    Status Process(PackagePtr data) noexcept override;

    std::shared_ptr<Processor> Fork() override
    {
      auto p = std::make_shared<CPUPreProc>();
      p->CopyParamsFrom(*this);
      if (p->Init(model_, "") != Status::SUCCESS){
        printf("!!!!!!!!!!!!!!!CPUPreProc Fork failed\n");
        return nullptr;
      }
      return p;
    }

private:
    // cv::Mat resize_mat_;
    ModelPtr model_;
    std::shared_ptr<CPUPreProcPriv> priv_;
};

// class CPUPreProcCreator : public PreProcCreator{
class CPUPreProcCreator : public PreProcCreator{    
public:
    CPUPreProcCreator():PreProcCreator("CPUPreProcCreator"){
        // std::cout << "Register2 " << GetCreatorName() << std::endl;
    }

    PreProcPtr Create() override {
        return std::make_shared<CPUPreProc>();
    }
    // std::string GetName() const override { return model_creator_name_; }

    // std::shared_ptr<Model> Create(const any& value) override {
    //     return std::shared_ptr<OrtModel>(new OrtModel(value));
    // }

private:

};  // class OrtModelCreator

}