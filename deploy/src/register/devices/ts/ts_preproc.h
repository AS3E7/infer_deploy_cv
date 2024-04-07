#pragma once

#include <memory>
#include <string>
#include "core/preprocess.h"

namespace gddeploy{
    
class TsPreProcPriv;
class TsPreProc : public PreProc {
public:
    TsPreProc() : PreProc("ts_preproc"){}
    // Status Init() noexcept override;
    Status Init(std::string config) noexcept override;

    Status Init(ModelPtr model, std::string config) override;

    Status Process(PackagePtr data) noexcept override;

    std::shared_ptr<Processor> Fork() override
    {
      auto p = std::make_shared<TsPreProc>();
      p->CopyParamsFrom(*this);
      if (p->Init(model_, "") != Status::SUCCESS){
        printf("!!!!!!!!!!!!!!!TsPreProc Fork failed\n");
        return nullptr;
      }
      return p;
    }

private:
    // cv::Mat resize_mat_;
    ModelPtr model_;
    std::shared_ptr<TsPreProcPriv> priv_;
};

// class TsPreProcCreator : public PreProcCreator{
class TsPreProcCreator : public PreProcCreator{    
public:
    TsPreProcCreator() : PreProcCreator("TsPreProcCreator"){
        // std::cout << "Register2 " << GetCreatorName() << std::endl;
    }

    PreProcPtr Create() override {
        return std::make_shared<TsPreProc>();
    }

private:

};  // class TsPreProcCreator

}