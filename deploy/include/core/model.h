#ifndef GDDEPLOY_MODEL_H
#define GDDEPLOY_MODEL_H

#include <algorithm>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>
#include <thread>

#include "core/buffer.h"
#include "core/infer_server.h"
#include "core/processor.h"
#include "core/shape.h"
#include "core/model_properties.h"

namespace gddeploy
{
class Model;
class ModelRunner {
public:
    explicit ModelRunner(int device_id) : device_id_(device_id) {}
    ModelRunner(const ModelRunner& other) = delete;
    ModelRunner& operator=(const ModelRunner& other) = delete;
    ModelRunner(ModelRunner&& other) = default;
    ModelRunner& operator=(ModelRunner&& other) = default;
    ~ModelRunner();

    bool Init(Model* model) noexcept;
    bool ForkFrom(const ModelRunner& other) noexcept;
    bool CanInferOutputShape() noexcept { return true; }

    Status Run(ModelIO* input, ModelIO* output) noexcept;  // NOLINT
private:
    void** params_{nullptr};
    uint32_t input_num_{0};
    uint32_t output_num_{0};
    int device_id_{0};
};

class Model : public ModelInfo {
public:
    Model() {};
    ~Model(){
        exit_flag_ = true;
        if (license_time_recode_thread_.joinable())
            license_time_recode_thread_.join();
    };

    virtual int Init(const std::string& model_path, const std::string& param){ return 0; };
    virtual int Init(void* mem_ptr, size_t mem_size, const std::string& param){ return 0; };
    bool HasInit() const noexcept { return has_init_; }

//     const Shape& InputShape(int index) const noexcept override {
//         // CHECK(index < i_num_ || index >= 0) << "[gddeploy InferServer] [Model] Input shape index overflow";
//         return input_shapes_[index];
//     }
//     const Shape& OutputShape(int index) const noexcept override {
//         // CHECK(index < o_num_ || index >= 0) << "[gddeploy InferServer] [Model] Output shape index overflow";
//         return output_shapes_[index];
//     }
//     const DataLayout& InputLayout(int index) const noexcept override {
//         // CHECK(index < i_num_ || index >= 0) << "[gddeploy InferServer] [Model] Input shape index overflow";
//         return i_mlu_layouts_[index];
//     }
//     const DataLayout& OutputLayout(int index) const noexcept override {
//         // CHECK(index < o_num_ || index >= 0) << "[gddeploy InferServer] [Model] Input shape index overflow";
//         return o_mlu_layouts_[index];
//     }
//     uint32_t InputNum() const noexcept override { return i_num_; }
//     uint32_t OutputNum() const noexcept override { return o_num_; }
//     uint32_t BatchSize() const noexcept override { return model_batch_size_; }

//     // bool FixedOutputShape() noexcept override { return FixedShape(output_shapes_); }
//     std::shared_ptr<ModelRunner> GetRunner(int device_id) noexcept {
//         std::unique_lock<std::mutex> lk(runner_map_mutex_);
//         if (!runner_map_.count(device_id)) {
//             auto runner = std::make_shared<ModelRunner>(device_id);
//             if (!runner->Init(this)) {
//                 return nullptr;
//             } else {
//                 runner_map_[device_id] = runner;
//                 return runner;
//             }
//         } else {
//             auto ret = std::make_shared<ModelRunner>(device_id);
//             if (!ret->ForkFrom(*runner_map_[device_id])) {
//                 return nullptr;
//             } else {
//                 return ret;
//             }
//         }
//     }

// private:
//     std::map<int, std::shared_ptr<ModelRunner>> runner_map_;
//     std::mutex runner_map_mutex_;
//     std::string path_, func_name_;

//     std::vector<DataLayout> i_mlu_layouts_, o_mlu_layouts_;
//     std::vector<Shape> input_shapes_, output_shapes_;

//     int i_num_{0}, o_num_{0};
//     uint32_t model_batch_size_{1};
    void SetModelInfoPriv(ModelPropertiesPtr model_info){
        model_info_priv_ = model_info;
    }

    virtual any GetModel(){ return nullptr; }

    ModelPropertiesPtr GetModelInfoPriv(){ return model_info_priv_; }

    std::vector<std::string> GetLabels();

    bool GetAuthorizationExpiredFlag(){ return authorization_expired_flag_; }
    bool InitExpiredTimeFile(std::string license_file_path, long int model_expired_time_l);

private: 

    ModelPropertiesPtr model_info_priv_;
    bool has_init_{false};

    // license相关
    bool exit_flag_ = false;
    void licenseThread();
    
    std::string license_root_;

    std::thread license_time_recode_thread_;
    bool authorization_expired_flag_ = false;
    long int expired_time_;
};// class Model

using ModelPtr = std::shared_ptr<Model>;

class ModelCreator{
public:
    ModelCreator(std::string model_creator_name){
        model_creator_name_ = model_creator_name;
    }

    virtual std::string GetName() const { return model_creator_name_; }

    virtual std::shared_ptr<Model> Create(const any& value) = 0;

private:
    std::string model_creator_name_;
};

// use environment CNIS_MODEL_CACHE_LIMIT to control cache limit
class ModelManager {
 public:
    ModelManager(){}
    ~ModelManager(){
        if (pInstance_ != nullptr){
            delete pInstance_;
            pInstance_ = nullptr;
        }
    }

    static ModelManager* Instance() noexcept{
        if (pInstance_ == nullptr){
            pInstance_ = new ModelManager();
        }
        return pInstance_;
    }

    void SetModelDir(const std::string& model_dir) noexcept { model_dir_ = model_dir; }

    ModelPtr Load(const std::string& model_path, const std::string& properties_path, const std::string &license_path = "", std::string param = "") noexcept;
    ModelPtr Load(void* mem_cache, size_t mem_size, const std::string &license_path = "", long int model_expired_time_l = 0, ModelPropertiesPtr model_info = nullptr, std::string param = "") noexcept;

    bool Unload(ModelInfoPtr model) noexcept;

    void ClearCache() noexcept;

    int CacheSize() noexcept;

    ModelPtr GetModel(const std::string& name) noexcept;

    static int RegisterModel(std::string manu, std::string chip, ModelCreator *model_creator) noexcept{
        std::unordered_map<std::string, ModelCreator*> chip_model_creator;
        chip_model_creator[chip] = model_creator;
        model_creator_[manu] = chip_model_creator;

        return 0;
    }

    static ModelCreator* GetModelCreator(std::string manu, std::string chip);

private:
    std::string DownloadModel(const std::string& url) noexcept;
    void CheckAndCleanCache() noexcept;

    std::string model_dir_{"."};

    static ModelManager *pInstance_;
    static std::unordered_map<std::string, std::shared_ptr<Model>> model_cache_;
    static std::mutex model_cache_mutex_;
    static std::unordered_map<std::string, std::unordered_map<std::string, ModelCreator*>> model_creator_;
};  // class ModelManager

// template <typename CreatorType>
// class ModelRegisterer {
// public:
//     ModelRegisterer() { ModelManager::RegisterModel(manu_, chip_, &inst_); }

// private:
//     std::string manu_;
//     std::string chip_;
//     CreatorType inst_;
// };


// #define REGISTER_MODEL_CREATOR(manu, chip, model_creator)   \
//     static ModelRegisterer<model_creator> g_register_##manu_##chip_##model_creator{};
    // static model_creator g_model_creator_inst; \
    // ModelManager::RegisterModel(manu, chip, &g_model_creator_inst);


int register_model_module();

} // namespace gddeploy

#endif