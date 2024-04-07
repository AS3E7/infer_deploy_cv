#pragma once
#include "core/alg.h"
#include "core/infer_server.h"

namespace gddeploy{

using InferAsyncCallback = std::function<void(gddeploy::Status status, gddeploy::PackagePtr data, gddeploy::any user_data)>;

enum ENUM_API_TYPE{
    ENUM_API_PROCESSOR_API,
    ENUM_API_SESSION_API
};

class InferAPIPrivate;
//api的接口根据解析配置文件和模型文件，获取前中后处理单元
class InferAPI{
public:
    InferAPI();
    ~InferAPI();

    // api_type: 选择底层的api接口为processor或者session api，区别在于有无预分配空间
    void Init(std::string config, std::string model_path, std::string license = "", ENUM_API_TYPE api_type = ENUM_API_PROCESSOR_API);
    
    int InferSync(const gddeploy::PackagePtr &in, gddeploy::PackagePtr &out);  //opencv4可支持解码图片格式

    void SetCallback(InferAsyncCallback cb);
    int InferAsync(const gddeploy::PackagePtr &in, InferAsyncCallback cb = nullptr, int timeout = 0);  
    int WaitTaskDone(const std::string& tag="");    

    std::string GetModelType();
    std::vector<std::string> GetLabels();
    
private:
    std::shared_ptr<InferAPIPrivate> priv_;
};

/**
 * @brief 获取设备 UUID
 * @return string
 */
// std::string GetDeviceUUID();

/**
 * @brief 获取软件version 
 * @return string
 */
// std::string GetVersion();

}