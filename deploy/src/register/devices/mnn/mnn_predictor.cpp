#include "mnn_predictor.h"
#include "core/model.h"
#include "core/processor.h"
#include "core/mem/buf_surface_util.h"
#include "core/mem/buf_surface.h"

#include <MNN/MNNDefine.h>
#include <MNN/MNNForwardType.h>
#include <MNN/Interpreter.hpp>

using namespace gddeploy;
// 通过ort_model的priv_获取到session
// 拷贝输入PackPtr里面的数据到session  Mnn::Value input_tensor = Mnn::Value::CreateTensor
// session推理，得到的结果拷贝到输出的output PackPtr

namespace gddeploy
{
class MnnPredictorPrivate{
public:
    MnnPredictorPrivate() = default;
    MnnPredictorPrivate(ModelPtr model):model_(model){
        net_ = gddeploy::any_cast<std::shared_ptr<MNN::Interpreter>>(model->GetModel());

        sess_config_.numThread = 4;
        sess_config_.type      = static_cast<MNNForwardType>(MNN_FORWARD_CPU);
        MNN::BackendConfig backendConfig;
        backendConfig.precision = (MNN::BackendConfig::PrecisionMode)2;
        // backendConfig.precision =  MNN::PrecisionMode Precision_Normal; // static_cast<PrecisionMode>(Precision_Normal);
        sess_config_.backendConfig = &backendConfig;
        sess_ = std::shared_ptr<MNN::Session>(net_->createSession(sess_config_), [this](MNN::Session *session){
            this->net_->releaseSession(session);
        });

    }

    ~MnnPredictorPrivate(){
        for (auto pool : pools_){
            pool->DestroyPool();
        }
    }

    int Init(std::string config);

    std::shared_ptr<MNN::Session> GetSession() { return sess_; }
    std::shared_ptr<MNN::Interpreter> GetNet() { return net_; }

    BufSurfWrapperPtr RequestBuffer(int idx){
        return pools_[idx]->GetBufSurfaceWrapper();
    }

    ModelPtr GetModelPtr(){ return model_; }

private:
    ModelPtr model_;

    std::shared_ptr<MNN::Session> sess_;
    MNN::ScheduleConfig sess_config_;
    std::shared_ptr<MNN::Interpreter> net_;

    size_t num_inputs_;
    size_t num_outputs_;
    std::vector<std::string> input_node_names_;
    std::vector<std::string> output_node_names_;

    std::vector<BufPool*> pools_;
};
}

int MnnPredictorPrivate::Init(std::string config)
{
    size_t o_num = model_->OutputNum();
    for (size_t i_idx = 0; i_idx < o_num; ++i_idx) {
        const DataLayout input_layout =  model_->OutputLayout(i_idx);
        auto dtype = input_layout.dtype;
        auto order = input_layout.order;
        int data_size = 0;
        if (dtype == DataType::INT8 || dtype == DataType::UINT8){
            data_size = sizeof(uint8_t);
        }else if (dtype == DataType::FLOAT16 || dtype == DataType::UINT16 || dtype == DataType::INT16){
            data_size = sizeof(uint16_t);
        }else if (dtype == DataType::FLOAT32 || dtype == DataType::INT32){
            data_size = sizeof(uint32_t);
        }

        int model_h, model_w, model_c, model_b;
        auto shape = model_->OutputShape(i_idx);
        if (order == DimOrder::NCHW){
            model_b = shape[0];
            model_h = shape[2];
            model_w = shape[3];
            model_c = shape[1];
        }else if (order == DimOrder::NHWC){
            model_b = shape[0];
            model_h = shape[1];
            model_w = shape[2];
            model_c = shape[3];
        }

        BufSurfaceCreateParams create_params;
        memset(&create_params, 0, sizeof(create_params));
        create_params.mem_type = GDDEPLOY_BUF_MEM_SYSTEM;
        create_params.force_align_1 = 1;  // to meet mm's requirement
        create_params.device_id = 0;
        create_params.batch_size = model_b;
        create_params.size = model_h * model_w * model_c;
        create_params.size *= data_size;
        create_params.width = model_w;
        create_params.height = model_h;
        create_params.color_format = GDDEPLOY_BUF_COLOR_FORMAT_RGB;

        BufPool *pool = new BufPool;
        if (pool->CreatePool(&create_params, 3) < 0) {
            return -1;
        }
        pools_.emplace_back(pool);
    }


    return 0;
}

Status MnnPredictor::Init(PredictorConfigPtr config, ModelPtr model) noexcept 
{
    priv_ = std::make_shared<MnnPredictorPrivate>(model);
    priv_->Init("");
    model_ = model;

    return gddeploy::Status::SUCCESS; 
}


Status MnnPredictor::Process(PackagePtr pack) noexcept
{
    BufSurfWrapperPtr in_buf = pack->predict_io->GetLref<BufSurfWrapperPtr>();

    Status s = Status::SUCCESS;

    std::shared_ptr<MNN::Session> sess = priv_->GetSession();
    std::shared_ptr<MNN::Interpreter> net = priv_->GetNet();

    // // 1. 拷贝数据
    auto inputTensor = net->getSessionInput(sess.get(), nullptr);
    int model_w = priv_->GetModelPtr()->InputShape(0)[3];
    int model_h = priv_->GetModelPtr()->InputShape(0)[2];
    std::vector<int> dims{1, model_h, model_w, 3};
    auto nhwc_Tensor = MNN::Tensor::create<float>(dims, NULL, MNN::Tensor::TENSORFLOW);
    auto nhwc_data   = nhwc_Tensor->host<float>();
    auto nhwc_size   = nhwc_Tensor->size();
    std::memcpy(nhwc_data, in_buf->GetData(0, 0), nhwc_size);

    inputTensor->copyFromHostTensor(nhwc_Tensor);
    
    // // 2. 推理
    net->runSession(sess.get());


    // // 1. 后处理，画图
    std::vector<BufSurfWrapperPtr> out_bufs;

    const std::map<std::string, MNN::Tensor*> output_maps = net->getSessionOutputAll(sess.get());
    std::vector<std::pair<std::string, MNN::Tensor*>> output_maps_v(output_maps.begin(), output_maps.end());
    std::sort(output_maps_v.begin(), output_maps_v.end(), 
            [](std::pair<std::string, MNN::Tensor*> &lhs, std::pair<std::string, MNN::Tensor*> &rhs){
                auto lhs_shape = lhs.second->shape();
                auto rhs_shape = rhs.second->shape();
        return lhs_shape[3] > rhs_shape[3];
    });

    int output_index = 0; 
    for (auto output_tensor: output_maps_v){
        //取出模型output数据
        MNN::Tensor *output  = net->getSessionOutput(sess.get(), output_tensor.first.c_str());
        MNN::Tensor host(output, output->getDimensionType());
        output->copyToHostTensor(&host);
        void *data = host.host<float>();

        // 申请模型输出内存
        BufSurfWrapperPtr buf = priv_->RequestBuffer(output_index);
        auto output_shape = priv_->GetModelPtr()->OutputShape(output_index++);
        auto data_count  = output_shape.BatchDataCount();
        
        memcpy((void *)buf->GetData(0, 0), data, data_count * sizeof(float));

        out_bufs.emplace_back(buf);
    }
    pack->predict_io->Set(out_bufs);

    return gddeploy::Status::SUCCESS; 
}


// REGISTER_PREDICTOR_CREATOR("ort", "cpu", MnnPredictorCreator)