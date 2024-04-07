#pragma once

#if 1
#include <memory>
#include <string>
#include "core/preprocess.h"

namespace gddeploy{
    
class CambriconPreProcPriv;
class CambriconPreProc : public PreProc {
public:
    CambriconPreProc() : PreProc("bmnn_preproc"){}
    // Status Init() noexcept override;
    Status Init(std::string config) noexcept override;

    Status Init(ModelPtr model, std::string config) override;

    Status Process(PackagePtr data) noexcept override;

private:
    // cv::Mat resize_mat_;
    std::shared_ptr<CambriconPreProcPriv> priv_;
};

// class CambriconPreProcCreator : public PreProcCreator{
class CambriconPreProcCreator : public PreProcCreator{    
public:
    CambriconPreProcCreator() : PreProcCreator("CambriconPreProcCreator"){
        // std::cout << "Register2 " << GetCreatorName() << std::endl;
    }

    PreProcPtr Create() override {
        return std::make_shared<CambriconPreProc>();
    }

private:

};  // class CambriconPreProcCreator

}

#endif