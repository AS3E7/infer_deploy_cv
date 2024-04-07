#pragma once

#include <memory>
#include <string>
#include "core/preprocess.h"

namespace gddeploy{
    
class AscendPreProcPriv;
class AscendPreProc : public PreProc {
public:
    AscendPreProc() : PreProc("ascend_preproc"){}
    // Status Init() noexcept override;
    Status Init(std::string config) noexcept override;

    Status Init(ModelPtr model, std::string config) override;

    Status Process(PackagePtr data) noexcept override;

    std::shared_ptr<Processor> Fork() override
    {
      auto p = std::make_shared<AscendPreProc>();
      p->CopyParamsFrom(*this);
      if (p->Init(model_, "") != Status::SUCCESS){
        printf("!!!!!!!!!!!!!!!AscendPreProc Fork failed\n");
        return nullptr;
      }
      return p;
    }

private:
    // cv::Mat resize_mat_;
    ModelPtr model_;
    std::shared_ptr<AscendPreProcPriv> priv_;
};

// class AscendPreProcCreator : public PreProcCreator{
class AscendPreProcCreator : public PreProcCreator{    
public:
    AscendPreProcCreator() : PreProcCreator("AscendPreProcCreator"){
        // std::cout << "Register2 " << GetCreatorName() << std::endl;
    }

    PreProcPtr Create() override {
        return std::make_shared<AscendPreProc>();
    }

private:

};  // class AscendPreProcCreator

}