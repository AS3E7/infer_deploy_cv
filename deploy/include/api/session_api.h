#pragma once
#include "core/alg.h"
#include "core/model.h"

namespace gddeploy{

using InferAsyncCallback = std::function<void(gddeploy::Status status, gddeploy::PackagePtr data, gddeploy::any user_data)>;

class SessionAPIPrivate;

typedef struct {
    std::string name;
    BatchStrategy strategy;   // 可选static和dynamic
    int batch_timeout;      // dynamic时可用
    int engine_num;         // 底层可并行运行engine个数
    int priority;           // 优先级
    bool show_perf;         // default false
} SessionAPI_Param;

//api的接口根据解析配置文件和模型文件，获取前中后处理单元
class SessionAPI{
public:
    SessionAPI();

    int Init(const std::string config, const std::string model_path, const std::string properties_path = "", std::string license = "");
    int Init(const SessionAPI_Param &config, const std::string model_path, const std::string properties_path = "", std::string license = "");
    
    //in: 目前只支持NV12输入
    int InferSync(const gddeploy::PackagePtr &in, gddeploy::PackagePtr &out);  //opencv4可支持解码图片格式

    void SetCallback(InferAsyncCallback cb);
    int InferAsync(const gddeploy::PackagePtr &in, InferAsyncCallback cb = nullptr, int timeout = 0);  
    int WaitTaskDone(const std::string& tag="");

    std::string GetModelType();
    std::vector<std::string> GetLabels();

private:
    std::shared_ptr<SessionAPIPrivate> priv_;
};

}