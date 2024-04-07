#pragma once

#include <memory>
#include <string>
#include "core/preprocess.h"

namespace gddeploy{
    
class RkPreProcPriv;
class RkPreProc : public PreProc {
public:
    RkPreProc() : PreProc("rk_preproc"){}
    // Status Init() noexcept override;
    Status Init(std::string config) noexcept override;

    Status Init(ModelPtr model, std::string config) override;

    Status Process(PackagePtr data) noexcept override;

    std::shared_ptr<Processor> Fork() override
    {
      auto p = std::make_shared<RkPreProc>();
      p->CopyParamsFrom(*this);
      if (p->Init(model_, "") != Status::SUCCESS){
        printf("!!!!!!!!!!!!!!!RkPreProc Fork failed\n");
        return nullptr;
      }
      return p;
    }

private:
    // cv::Mat resize_mat_;
    ModelPtr model_;
    std::shared_ptr<RkPreProcPriv> priv_;
};

// class RkPreProcCreator : public PreProcCreator{
class RkPreProcCreator : public PreProcCreator{    
public:
    RkPreProcCreator() : PreProcCreator("RkPreProcCreator"){
        // std::cout << "Register2 " << GetCreatorName() << std::endl;
    }

    PreProcPtr Create() override {
        return std::make_shared<RkPreProc>();
    }

private:

};  // class RkPreProcCreator

}