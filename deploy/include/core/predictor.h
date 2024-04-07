#ifndef GDDEPLOY_PREDICTOR_H
#define GDDEPLOY_PREDICTOR_H

#include <algorithm>
#include <map>
#include <mutex>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "core/infer_server.h"
#include "core/processor.h"
#include "core/model.h"

namespace gddeploy
{
    class PredictorConfig
    {
    public:
        PredictorConfig(std::string config); // 解析字符串的config，到对应的变量

        void PrintConfig();

        std::string GetConfigStr() { return config_; }

    private:
        std::string config_;
        std::map<std::string, std::string> config_map_;
    };
    using PredictorConfigPtr = std::shared_ptr<PredictorConfig>;

    // 输入输出以PackPtr作为参数类型
    class Predictor : public ProcessorForkable<Predictor>
    {
    public:
        Predictor() = default;
        Predictor(const std::string &type_name) : ProcessorForkable<Predictor>(type_name) {}
        /**
         * @brief Perform predict
         *
         * @param data processed data
         * @retval Status::SUCCESS Succeeded
         * @retval Status::INVALID_PARAM PredictorOld get uncontinuous data
         * @retval Status::WRONG_TYPE PredictorOld get data of wrong type (bad_any_cast)
         * @retval Status::ERROR_BACKEND Predict failed
         */
        virtual Status Process(PackagePtr data) noexcept override { return gddeploy::Status::SUCCESS; };

        /**
         * @brief Initialize predictor
         *
         * @retval Status::SUCCESS Init succeeded
         * @retval Status::INVALID_PARAM PredictorOld donot have enough params or get wrong params, @see BaseObject::SetParam
         * @retval Status::WRONG_TYPE PredictorOld get params of wrong type (bad_any_cast)
         */
        virtual Status Init() noexcept override { return gddeploy::Status::SUCCESS; };
        virtual Status Init(PredictorConfigPtr config, ModelPtr model) noexcept { return gddeploy::Status::SUCCESS; };

    private:
        ModelPtr model_;
        PredictorConfigPtr config_;
        std::string predictor_name_;
    };

    // 后面应该有几个常见的算法前处理配置文件继承，上面只需要传入product、chip、net_type就可以确认配置信息

    class PredictorCreator
    {
    public:
        PredictorCreator() {}
        PredictorCreator(std::string name) : predictor_creator_name_(name) {}
        PredictorCreator(PredictorConfig *config) {}
        ~PredictorCreator() {}

        virtual std::shared_ptr<Predictor> Create(PredictorConfigPtr config, ModelPtr model) = 0;

    private:
        std::string predictor_creator_name_;
    };

    class PredictorManager
    {
    public:
        static PredictorManager *Instance() noexcept
        {
            if (pInstance_ == nullptr)
            {
                pInstance_ = new PredictorManager();
            }
            return pInstance_;
        }

        // void ClearCache() noexcept;

        // int CacheSize() noexcept;

        // std::shared_ptr<Predictor> GetPredictor(PredictorConfigPtr config, ModelPtr model) noexcept;
        PredictorCreator *GetPredictorCreator(std::string manufacturer, std::string chip)
        {
            if (predictor_creator_.count(manufacturer) == 0)
                return nullptr;
        
            if (predictor_creator_[manufacturer].count("any") != 0)
                return predictor_creator_[manufacturer]["any"];

            return predictor_creator_[manufacturer][chip];
        }

        int register_predictor(std::string manufacturer, std::string chip, PredictorCreator *predictor_creator) noexcept
        {
            std::unordered_map<std::string, PredictorCreator *> chip_predictor_creator;
            chip_predictor_creator[chip] = predictor_creator;
            predictor_creator_[manufacturer] = chip_predictor_creator;

            return 0;
        }

    private:
        // static std::unordered_map<std::string, std::shared_ptr<Predictor>> predictor_cache_;
        // static std::mutex predictor_cache_mutex_;
        std::unordered_map<std::string, std::unordered_map<std::string, PredictorCreator *>> predictor_creator_;

        static PredictorManager *pInstance_;
    };

    int register_predictor_module();

    // template <typename CreatorType>
    // class PredictorRegisterer {
    // public:
    //     PredictorRegisterer() { PredictorManager::register_predictor(manu_, chip_, &inst_); }

    // private:
    //     std::string manu_;
    //     std::string chip_;
    //     CreatorType inst_;
    // };

    // #define REGISTER_PREDICTOR_CREATOR(manufacturer, chip, predictor_creator)   \
//     static PredictorRegisterer<predictor_creator> g_register_##manufacturer_##chip_##predictor_creator{};

}

#endif