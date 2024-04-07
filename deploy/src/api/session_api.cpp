#include <memory>
#include "api/session_api.h"
#include "core/infer_server.h"
#include "core/model.h"
#include "core/cv.h"
#include "core/mem/buf_surface.h"
#include "core/mem/buf_surface_util.h"
#include "core/mem/buf_surface_impl.h"
#include "common/json.hpp"
#include "api/global_config.h"
#include "core/result_def.h"

using namespace gddeploy;

namespace gddeploy{

class GddObserver : public gddeploy::Observer {
public:
    GddObserver(){}
    GddObserver(InferAsyncCallback cb) {
        callback_ = cb;
    }
    void Response(gddeploy::Status status, gddeploy::PackagePtr out, gddeploy::any user_data) noexcept {
        // std::cout << "GddObserver" << std::endl;
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


class SessionAPIPrivate {
public:
    SessionAPIPrivate();
    ~SessionAPIPrivate();

    int Init(const std::string config, const std::string model_path, const std::string properties_path, std::string license = "");

    int Init(const SessionAPI_Param config, const std::string model_path, const std::string properties_path, std::string license = "");

    int InferSync(const gddeploy::PackagePtr &in, gddeploy::PackagePtr &out);

    int InferAsync(const gddeploy::PackagePtr &in, int timeout = 0);

    int WaitTaskDone(const std::string& tag);

    std::vector<std::string> GetLabels();

    void SetCallback(InferAsyncCallback cb){
        // observer_ = std::make_shared<GddObserver>(cb);
        // infer_server_->SetObserver(session_async_, observer_);
        observer_->SetCallback(cb);
    }

    std::string GetModelType();

    // 切图函数
    int SplitImage(const gddeploy::PackagePtr &in, gddeploy::PackagePtr &out, int slice_height, int slice_width, std::string manu, std::string chip);
    int CombineResult(gddeploy::PackagePtr &out){
        // 这里需要合并结果
        gddeploy::InferResult result;
        result.result_type.emplace_back(GDD_RESULT_TYPE_DETECT);
        result.detect_result.batch_size = 1;

        for (auto &data : out->data) {
            if (false == data->HasMetaValue())
                return -1;

            gddeploy::InferResult result_tmp = data->GetMetaData<gddeploy::InferResult>(); 
            if (result_tmp.result_type[0] == GDD_RESULT_TYPE_DETECT){

                for (uint32_t i = 0; i < result_tmp.detect_result.detect_imgs.size(); i++){
                    CropParam param = data->GetUserData<CropParam>();
                    std::pair<int, int> frame_data = data->GetFrameData<std::pair<int, int>>();
                    int slice_width = frame_data.first;
                    int slice_height = frame_data.second;

                    if (result.detect_result.detect_imgs.size() == 0)
                        result.detect_result.detect_imgs.emplace_back(result_tmp.detect_result.detect_imgs[i]);

                    // 这里需要将detect_imgs的坐标转换回去
                    for (uint32_t j = 0; j < result_tmp.detect_result.detect_imgs[i].detect_objs.size(); j++){
                        result_tmp.detect_result.detect_imgs[i].detect_objs[j].bbox.x += param.start_x - (slice_width - param.crop_w)/2;
                        result_tmp.detect_result.detect_imgs[i].detect_objs[j].bbox.y += param.start_y - (slice_height - param.crop_h)/2;
                        

                        result.detect_result.detect_imgs[0].detect_objs.emplace_back(result_tmp.detect_result.detect_imgs[i].detect_objs[j]);
                    }                    
                }
            }
        }

        out->data[0]->SetMetaData(result);

        return 0;
    }

    int dev_id_; 
    int stream_id_;
private:
    ModelPtr model_;
    std::shared_ptr<gddeploy::InferServer> infer_server_ = nullptr;
    gddeploy::SessionDesc desc_;
    gddeploy::Session_t session_async_ = nullptr;
    gddeploy::Session_t session_sync_ = nullptr;

    std::shared_ptr<GddObserver> observer_ = nullptr;
};
}   // namespace gddeploy

SessionAPIPrivate::SessionAPIPrivate()
{
}

SessionAPIPrivate::~SessionAPIPrivate()
{
    if (infer_server_ && session_async_ && session_sync_) {
        WaitTaskDone("");
        infer_server_->DestroySession(session_async_);
        infer_server_->DestroySession(session_sync_);
        std::cout << "Release session" << std::endl;
        session_async_ = nullptr;
        session_sync_ = nullptr;
        infer_server_ = nullptr;
    }
}

int ParseJson2Param(const std::string config, SessionAPI_Param &param)
{
    if (config == ""){
        param.name = "default_session";
        param.strategy = (gddeploy::BatchStrategy)SESSION_DESC_STRATEGY;
        param.batch_timeout = SESSION_DESC_BATCH_TIMEOUT;
        param.engine_num = SESSION_DESC_ENGINE_NUM;
        param.priority = SESSION_DESC_PRIORITY;
        param.show_perf = SESSION_DESC_SHOW_PERF;
        return -1;
    }

    nlohmann::json j = nlohmann::json::parse(config);
    if (j.contains("name") && j["name"].is_string()) {
        param.name = j["name"].get<std::string>();
    } else {
        param.name = "default_session";
    }

    if (j.contains("strategy") && j["strategy"].is_number()) {
        param.strategy = (gddeploy::BatchStrategy)j["strategy"].get<int>();
    } else {
        param.strategy = (gddeploy::BatchStrategy)SESSION_DESC_STRATEGY;
    }

    if (j.contains("batch_timeout") && j["batch_timeout"].is_number()) {
        param.batch_timeout = j["batch_timeout"].get<int>();
    } else {
        param.batch_timeout = SESSION_DESC_BATCH_TIMEOUT;
    }

    if (j.contains("engine_num") && j["engine_num"].is_number()) {
        param.engine_num = j["engine_num"].get<int>();
    } else {
        param.engine_num = SESSION_DESC_ENGINE_NUM;
    }

    if (j.contains("priority") && j["priority"].is_number()) {
        param.priority = j["priority"].get<int>();
    } else {
        param.priority = SESSION_DESC_PRIORITY;
    }

    if (j.contains("show_perf") && j["show_perf"].is_boolean()) {
        param.show_perf = j["show_perf"].get<bool>();
    } else {
        param.show_perf = SESSION_DESC_SHOW_PERF;
    }

    return 0;
}

int SessionAPIPrivate::Init(const std::string config, const std::string model_path, const std::string properties_path, std::string license)
{
    SessionAPI_Param param;
    ParseJson2Param(config, param);
    
    infer_server_.reset(new gddeploy::InferServer(0));
    desc_.strategy = param.strategy; //STATIC;
    desc_.batch_timeout = param.batch_timeout;
    desc_.engine_num = param.engine_num;
    desc_.priority = param.priority;
    desc_.show_perf = param.show_perf;
    desc_.name = param.name;

    model_ = gddeploy::ModelManager::Instance()->Load(model_path, properties_path, license, "");
    desc_.model = model_;

    observer_ = std::make_shared<GddObserver>();
    session_async_ = infer_server_->CreateSession(desc_, observer_);
    desc_.strategy = BatchStrategy::STATIC; //STATIC;
    session_sync_  = infer_server_->CreateSession(desc_, nullptr);
    return 0;
}

int SessionAPIPrivate::Init(const SessionAPI_Param param, const std::string model_path, const std::string properties_path, std::string license)
{    
    infer_server_.reset(new gddeploy::InferServer(0));
    desc_.strategy = param.strategy; //STATIC;
    desc_.batch_timeout = param.batch_timeout;
    desc_.engine_num = param.engine_num;
    desc_.priority = param.priority;
    desc_.show_perf = param.show_perf;
    desc_.name = param.name;

    model_ = gddeploy::ModelManager::Instance()->Load(model_path, properties_path, license, "");
    desc_.model = model_;

    observer_ = std::make_shared<GddObserver>();
    session_async_ = infer_server_->CreateSession(desc_, observer_);
    desc_.strategy = BatchStrategy::STATIC; //STATIC;
    session_sync_  = infer_server_->CreateSession(desc_, nullptr);
    return 0;
}

std::vector<std::string> SessionAPIPrivate::GetLabels()
{
    ModelPropertiesPtr mp = model_->GetModelInfoPriv();
    
    return mp->GetLabels();
}

// 切图函数
int SessionAPIPrivate::SplitImage(const gddeploy::PackagePtr &in, gddeploy::PackagePtr &out, int slice_height, int slice_width, std::string manu, std::string chip)
{
    // 1. 根据切图的slice_height和slice_width参数，计算out需要多个图像
    auto &data = in->data[0];
    auto surf = data->GetLref<gddeploy::BufSurfWrapperPtr>();
    BufSurface *surface = surf->GetBufSurface();
    BufSurfaceParams *src_param = surf->GetSurfaceParams(0);

    // 计算h和w方向的切图数量，如果不整除，最后一个切图的大小加padding
    int slice_h_num = src_param->height % slice_height == 0 ? src_param->height / slice_height : src_param->height / slice_height + 1;
    int slice_w_num = src_param->width % slice_width == 0 ? src_param->width / slice_width : src_param->width / slice_width + 1;
    int slice_num = slice_h_num * slice_w_num;

    // 2. 为out分配内存
    out = gddeploy::Package::Create(slice_num);

    BufSurfaceCreateParams params;
#if WITH_BM1684
    params.mem_type = GDDEPLOY_BUF_MEM_BMNN;
#elif WITH_NVIDIA
    // params.mem_type = GDDEPLOY_BUF_MEM_NVIDIA;
    params.mem_type = GDDEPLOY_BUF_MEM_SYSTEM;
#elif WITH_TS
    params.mem_type = GDDEPLOY_BUF_MEM_TS;
#else 
    params.mem_type = GDDEPLOY_BUF_MEM_SYSTEM;
#endif
    params.device_id = 0;
    params.width = slice_width;
    params.height = slice_height;
    params.color_format = src_param->color_format;
    params.force_align_1 = 1;
    params.bytes_per_pix = 1;
    params.size = 0;
    // params.size = params.width * params.height * 3 / 2;  // 这里要注意不要赋值，内部会根据是否需要对齐调整大小
    params.batch_size = 1;

    std::vector<gddeploy::BufSurfWrapperPtr> out_data(slice_num);
    for (int i = 0; i < slice_num; i++){
        // 2.1 创建out的BufSurface
        // std::shared_ptr<BufSurfaceWrapper> tmp(new BufSurfaceWrapper(new BufSurface, true)
        std::shared_ptr<BufSurfaceWrapper> surf_ptr(new BufSurfaceWrapper(new BufSurface, true));
        BufSurface *surf = surf_ptr->GetBufSurface();
        memset(surf, 0, sizeof(BufSurface));
        CreateSurface(&params, surf);

        out->data[i]->Set(surf_ptr);
        out->data[i]->SetFrameData(std::pair<int, int>(slice_width, slice_height));
        out_data[i] = surf_ptr;
    }
    // 3. 设置切图的参数
    std::vector<CropParam> crop_params;
    for (int i = 0; i < slice_num; i++){
        int start_x = i % slice_w_num * slice_width;
        int start_y = i / slice_w_num * slice_height;
        int end_x = start_x + slice_width;
        int end_y = start_y + slice_height;

        if (end_x > src_param->width){
            end_x = src_param->width;
        }

        if (end_y > src_param->height){
            end_y = src_param->height;
        }

        CropParam param;
        param.start_x = start_x;
        param.start_y = start_y;
        param.crop_w = end_x - start_x;
        param.crop_h = end_y - start_y;
        crop_params.push_back(param);

        out->data[i]->SetUserData(param);
    }

    // 4. 调用切图函数
    CV *cv = CV::Instance()->GetCV(manu, chip);

    cv->Crop(surf, out_data, crop_params);

    return 0;
}

static std::vector<DetectObject> nms(std::vector<DetectObject> objInfos, float iou_thresh=0.45, float ios_thresh = 0.45)
{
    std::sort(objInfos.begin(), objInfos.end(), [](DetectObject lhs, DetectObject rhs)
              { return lhs.score > rhs.score; });
    if (objInfos.size() > 2000)
    {
        objInfos.erase(objInfos.begin() + 2000, objInfos.end());
    }

    std::vector<DetectObject> result;

    while (objInfos.size() > 0){
        result.push_back(objInfos[0]);
  
        for (auto it = objInfos.begin() + 1; it != objInfos.end();)
        {
            auto box1 = objInfos[0].bbox;
            auto box2 = (*it).bbox;

            float x1 = std::max(box1.x, box2.x);
            float y1 = std::max(box1.y, box2.y);
            float x2 = std::min(box1.x+box1.w, box2.x+box2.w);
            float y2 = std::min(box1.y+box1.h, box2.y+box2.h);
            float over_w = std::max(0.0f, x2 - x1);
            float over_h = std::max(0.0f, y2 - y1);
            float over_area = over_w * over_h;
            float iou_value = over_area / ((box1.w ) * (box1.h ) + (box2.w ) * (box2.h ) - over_area);

            float area1 = box1.w * box1.h;
            float area2 = box2.w * box2.h;
            float min_area = std::min(area1, area2);
            float ios_value = over_area / min_area;

            printf("iou_value: %f, ios_value: %f\n", iou_value, ios_value);
            if (ios_value > ios_thresh || iou_value > iou_thresh)
                it = objInfos.erase(it);
            else
                it++; 
        }
        objInfos.erase(objInfos.begin());
    }

    return result;
}


int SessionAPIPrivate::InferSync(const gddeploy::PackagePtr &in, gddeploy::PackagePtr &out)
{
    gddeploy::Status status = gddeploy::Status::SUCCESS;

    auto model_info_priv = model_->GetModelInfoPriv();

    if (model_info_priv->GetNeedClip()){
    // if (false){
        // 先全图推理，再切图推理
        // 1. 全图推理
        bool ret = infer_server_->RequestSync(session_sync_, in, &status, out);
        if (!ret || status != gddeploy::Status::SUCCESS) {
            std::cout << "infer server request sync false" << std::endl;
            return -1;
        }

        std::vector<DetectObject> all_detect_objs;
        for (auto &data : out->data) {
            if (false == data->HasMetaValue())
                return -1;

            gddeploy::InferResult result_tmp = data->GetMetaData<gddeploy::InferResult>(); 
            if (result_tmp.result_type[0] == GDD_RESULT_TYPE_DETECT){
                for (uint32_t i = 0; i < result_tmp.detect_result.detect_imgs.size(); i++){
                    all_detect_objs.insert(all_detect_objs.end(), result_tmp.detect_result.detect_imgs[i].detect_objs.begin(), result_tmp.detect_result.detect_imgs[i].detect_objs.end());
                }
            }
        }

        // 2. 切图推理
        gddeploy::PackagePtr split_in;
        std::string manu = model_info_priv->GetProductType();
        std::string chip = model_info_priv->GetChipType();

        int slice_h = model_info_priv->GetInputSize();
        int slice_w = model_info_priv->GetInputSize();
        if (slice_h != 0 && slice_w != 0){
            // 切图
            if (-1 == SplitImage(out, split_in, slice_h, slice_w, manu, chip)){
                std::cout << "split image error" << std::endl;
                return -1;
            }
        }

        ret = infer_server_->RequestSync(session_sync_, split_in, &status, out);
        if (!ret || status != gddeploy::Status::SUCCESS) {
            std::cout << "infer server request sync false" << std::endl;
            return -1;
        }

        CombineResult(out);

        // 去除out->data[0]以后的data
        out->data.erase(out->data.begin() + 1, out->data.end());

        for (auto &data : out->data) {
            if (false == data->HasMetaValue())
                return -1;

            gddeploy::InferResult result_tmp = data->GetMetaData<gddeploy::InferResult>(); 
            if (result_tmp.result_type[0] == GDD_RESULT_TYPE_DETECT){
                for (uint32_t i = 0; i < result_tmp.detect_result.detect_imgs.size(); i++){
                    all_detect_objs.insert(all_detect_objs.end(), result_tmp.detect_result.detect_imgs[i].detect_objs.begin(), result_tmp.detect_result.detect_imgs[i].detect_objs.end());
                }
            }
        }

        // 3. nms过滤
        std::vector<DetectObject> split_detect_objs = nms(all_detect_objs, 0.45, 0.55);
        gddeploy::InferResult result_tmp = out->data[0]->GetMetaData<gddeploy::InferResult>(); 
        result_tmp.detect_result.detect_imgs[0].detect_objs.clear();
        result_tmp.detect_result.detect_imgs[0].detect_objs = split_detect_objs;

        out->data[0]->SetMetaData(result_tmp);
    } else {
        bool ret = infer_server_->RequestSync(session_sync_, in, &status, out);
        if (!ret || status != gddeploy::Status::SUCCESS) {
            std::cout << "infer server request sync false" << std::endl;
            return -1;
        }
    }

    return 0;
}

int SessionAPIPrivate::InferAsync(const gddeploy::PackagePtr &in, int timeout)
{
    gddeploy::Status status = gddeploy::Status::SUCCESS;
    bool ret = infer_server_->Request(session_async_, std::move(in), &status, timeout);
    if (!ret || status != gddeploy::Status::SUCCESS) {
        std::cout << "infer server request sync false" << std::endl;
        return -1;
    }
    
    return 0;
}

int SessionAPIPrivate::WaitTaskDone(const std::string& tag)
{
    infer_server_->WaitTaskDone(session_async_, tag);
    return 0;
}

std::string SessionAPIPrivate::GetModelType()
{
    auto model_info_priv = model_->GetModelInfoPriv();
    return model_info_priv->GetModelType();
}



SessionAPI::SessionAPI()
{
    priv_ = std::make_shared<SessionAPIPrivate>();
}


int SessionAPI::Init(const std::string config, const std::string model_path, const std::string properties_path, std::string license)
{
    //载入model
    return priv_->Init(config, model_path,  properties_path, license);
}

int SessionAPI::Init(const SessionAPI_Param & config, const std::string model_path, const std::string properties_path, std::string license)
{
    //载入model
    return priv_->Init(config, model_path,  properties_path, license);
}

int SessionAPI::InferSync(const gddeploy::PackagePtr &in, gddeploy::PackagePtr &out)
{
    if (-1 == priv_->InferSync(in, out))
        return -1;
    return 0;
}

void SessionAPI::SetCallback(InferAsyncCallback cb)
{
    if (cb != nullptr)
        priv_->SetCallback(cb);
}

int SessionAPI::InferAsync(const gddeploy::PackagePtr &in, InferAsyncCallback cb, int timeout)
{
    if (cb != nullptr)
        priv_->SetCallback(cb);
    if (-1 == priv_->InferAsync(in, timeout))
        return -1;
    return 0;
}

int SessionAPI::WaitTaskDone(const std::string& tag)
{
    return priv_->WaitTaskDone(tag);
}

std::vector<std::string> SessionAPI::GetLabels()
{
    return priv_->GetLabels();
}

std::string SessionAPI::GetModelType()
{
    return priv_->GetModelType();
}