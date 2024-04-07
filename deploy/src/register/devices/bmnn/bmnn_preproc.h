#ifndef GDDEPLOY_BMNN_PREPROC_H_
#define GDDEPLOY_BMNN_PREPROC_H_

#include <memory>
#include <string>
#include "core/preprocess.h"

namespace gddeploy{
    
class BmnnPreProcPriv;
class BmnnPreProc : public PreProc {
public:
    BmnnPreProc() : PreProc("bmnn_preproc"){}
    // Status Init() noexcept override;
    Status Init(std::string config) noexcept override;

    Status Init(ModelPtr model, std::string config) override;

    Status Process(PackagePtr data) noexcept override;

    std::shared_ptr<Processor> Fork() override
    {
      auto p = std::make_shared<BmnnPreProc>();
      p->CopyParamsFrom(*this);
      if (p->Init(model_, "") != Status::SUCCESS){
        printf("!!!!!!!!!!!!!!!BmnnPreProc Fork failed\n");
        return nullptr;
      }
      return p;
    }

private:
    // cv::Mat resize_mat_;
    ModelPtr model_;
    std::shared_ptr<BmnnPreProcPriv> priv_;
};

// class BmnnPreProcCreator : public PreProcCreator{
class BmnnPreProcCreator : public PreProcCreator{    
public:
    BmnnPreProcCreator() : PreProcCreator("BmnnPreProcCreator"){
        // std::cout << "Register2 " << GetCreatorName() << std::endl;
    }

    PreProcPtr Create() override {
        return std::make_shared<BmnnPreProc>();
    }

private:

};  // class BmnnPreProcCreator

}

#endif