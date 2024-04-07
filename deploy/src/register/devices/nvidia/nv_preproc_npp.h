#pragma once


#include <memory>
#include <string>
#include "core/preprocess.h"

namespace gddeploy{
    
class NvPreProcNPPPriv;
class NvPreProcNPP : public PreProc {
public:
    NvPreProcNPP() : PreProc("npp_preproc"){}
    // Status Init() noexcept override;
    Status Init(std::string config) noexcept override;

    Status Init(ModelPtr model, std::string config) override;

    Status Process(PackagePtr data) noexcept override;

private:
    // cv::Mat resize_mat_;
    std::shared_ptr<NvPreProcNPPPriv> priv_;
};

// class NvPreProcNPPCreator : public PreProcCreator{
class NvPreProcNPPCreator : public PreProcCreator{    
public:
    NvPreProcNPPCreator() : PreProcCreator("NvPreProcNPPCreator"){
        // std::cout << "Register2 " << GetCreatorName() << std::endl;
    }

    PreProcPtr Create() override {
        return std::make_shared<NvPreProcNPP>();
    }

private:

};  // class NvPreProcNPPCreator

}

