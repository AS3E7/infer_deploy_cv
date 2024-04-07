#include "api/infer_api.h"
#include <memory>
#include <string>
#include "common/logger.h"
#include "core/model.h"
#include "api/processor_api.h"
#include "api/session_api.h"

using namespace gddeploy;


namespace gddeploy{

class GddObserver : public gddeploy::Observer {
public:
    GddObserver(){}
    GddObserver(InferAsyncCallback cb) {
        callback_ = cb;
    }
    void Response(gddeploy::Status status, gddeploy::PackagePtr out, gddeploy::any user_data) noexcept {
        if (callback_ != nullptr){
            callback_(status, out, user_data);
        }
    }

    void SetCallback(InferAsyncCallback cb){
        callback_ = cb;
    }
private:
    InferAsyncCallback callback_ = nullptr;
};


class InferAPIPrivate {
public:
    InferAPIPrivate();
    ~InferAPIPrivate();

    int Init(const std::string config, const std::string model_path, std::string license = "", ENUM_API_TYPE api_type = ENUM_API_PROCESSOR_API);

    int InferSync(const gddeploy::PackagePtr &in, gddeploy::PackagePtr &out);

    int InferAsync(const gddeploy::PackagePtr &in, int timeout = 0);

    int WaitTaskDone(const std::string& tag);

    std::vector<std::string> GetLabels();

    void SetCallback(InferAsyncCallback cb){
        // observer_ = std::make_shared<GddObserver>(cb);
        // infer_server_->SetObserver(session_async_, observer_);
        api_sess_.SetCallback(cb);
        cb_ = cb;
    }

    std::string GetModelType();

    int dev_id_; 
    int stream_id_;
private:
    ENUM_API_TYPE api_type_;

    // ProcessorAPI 相关
    ProcessorAPI processor_api_;
    std::vector<ProcessorPtr> processors_;

    // SessionAPI 相关
    InferAsyncCallback cb_;
    SessionAPI api_sess_;
};
}   // namespace gddeploy


InferAPIPrivate::InferAPIPrivate()
{
}

InferAPIPrivate::~InferAPIPrivate()
{
}

int InferAPIPrivate::Init(const std::string config, const std::string model_path, std::string license, ENUM_API_TYPE api_type)
{
    api_type_ = api_type;
    if (api_type == ENUM_API_SESSION_API){
        api_sess_.Init(config, model_path, "", license);
    }else if (api_type == ENUM_API_PROCESSOR_API){
        processor_api_.Init(config, model_path, license);
        processors_ = processor_api_.GetProcessor();
    }

    return 0;
}

int InferAPIPrivate::InferSync(const gddeploy::PackagePtr &in, gddeploy::PackagePtr &out)
{
    if (api_type_ == ENUM_API_SESSION_API){
        bool ret = api_sess_.InferSync(in, out);
        if (ret) {
            GDDEPLOY_ERROR("infer server request sync false");
            return -1;
        }
    }else if (api_type_ == ENUM_API_PROCESSOR_API){
        // 4. 循环执行每个processor的Process函数
        for (auto processor : processors_){
            if (gddeploy::Status::SUCCESS != processor->Process(in))
                break;
        }
        out = in;
    }
    return 0;
}

int InferAPIPrivate::InferAsync(const gddeploy::PackagePtr &in, int timeout)
{
    bool ret = api_sess_.InferAsync(in, cb_, timeout);
    if (ret) {
        GDDEPLOY_ERROR("infer server request sync false");
        return -1;
    }
    
    return 0;
}

int InferAPIPrivate::WaitTaskDone(const std::string& tag)
{
    api_sess_.WaitTaskDone(tag);
    return 0;
}

std::string InferAPIPrivate::GetModelType()
{
    std::string model_type = "";
    if (api_type_ == ENUM_API_SESSION_API){
        model_type =  api_sess_.GetModelType();
    }else if (api_type_ == ENUM_API_PROCESSOR_API){
        model_type = processor_api_.GetModelType();
    }

    return  model_type;
}

std::vector<std::string> InferAPIPrivate::GetLabels()
{
    std::vector<std::string> labels;

    if (api_type_ == ENUM_API_SESSION_API){
        labels =  api_sess_.GetLabels();
    }else if (api_type_ == ENUM_API_PROCESSOR_API){
        labels = processor_api_.GetLabels();
    }
    
    return labels;
}

InferAPI::InferAPI()
{
    priv_ = std::make_shared<InferAPIPrivate>();
}

InferAPI::~InferAPI()
{
}

void InferAPI::Init(std::string config, std::string model_path, std::string license, ENUM_API_TYPE api_type)
{
    priv_->Init(config, model_path, license, api_type);
}


int InferAPI::InferSync(const gddeploy::PackagePtr &in, gddeploy::PackagePtr &out)
{
    if (-1 == priv_->InferSync(in, out))
        return -1;
    return 0;
}

void InferAPI::SetCallback(InferAsyncCallback cb)
{
    if (cb != nullptr)
        priv_->SetCallback(cb);
}

int InferAPI::InferAsync(const gddeploy::PackagePtr &in, InferAsyncCallback cb, int timeout)
{
    if (cb != nullptr)
        priv_->SetCallback(cb);
    if (-1 == priv_->InferAsync(in, timeout))
        return -1;
    return 0;
}

int InferAPI::WaitTaskDone(const std::string& tag)
{
    priv_->WaitTaskDone(tag);
    return 0;
}

std::string InferAPI::GetModelType()
{
    return priv_->GetModelType();
}

std::vector<std::string> InferAPI::GetLabels()
{
    return priv_->GetLabels();
}