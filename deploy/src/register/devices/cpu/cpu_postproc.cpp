#include <vector>
#include <memory>
#include <string>

#include "cpu_postproc.h"
#include "core/model.h"
#include "core/result_def.h"
#include "opencv2/opencv.hpp"

#include "postproc/postproc_yolov5.h"
#include "postproc/postproc_yolov6.h"
#include "postproc/postproc_classify.h"
#include "postproc/postproc_pose.h"
#include "postproc/postproc_rtmpose.h"
#include "postproc/postproc_seg.h"
#include "postproc/postproc_image_retrieval.h"
#include "postproc/postproc_face_retrieval.h"
#include "postproc/postproc_ocr_det.h"
#include "postproc/postproc_ocr_rec.h"
#include "postproc/postproc_action.h"

#define DUMP_OUTPUT 1
using namespace gddeploy;

namespace gddeploy
{
    class CpuPostProcPriv
    {
    public:
        int Init(std::string config) { return 0; } // 算子的输入参数

    private:
        // cv::Mat resize_mat_;
    };
}

Status CpuPostProc::Init(ModelPtr model, std::string config)
{
    // if (HaveParam("model_info")){
    //     ModelPtr model = GetParam<ModelPtr>("model_info");
    // }

    model_ = model;
    priv_ = std::make_shared<CpuPostProcPriv>();

    return gddeploy::Status::SUCCESS;
}

Status CpuPostProc::Process(PackagePtr pack) noexcept
{
    // auto t0 = std::chrono::high_resolution_clock::now();

    // auto t0 = std::chrono::high_resolution_clock::now();
    std::vector<BufSurfWrapperPtr> out_bufs = pack->predict_io->GetLref<std::vector<BufSurfWrapperPtr>>();
    // #if DUMP_OUTPUT
#if 0
    for (auto i = 0; i < out_bufs.size(); i++){
        char pic_name[32] = {0};
        sprintf(pic_name, "/tmp/output_tensor_c%d.bin", i);
        FILE *file_handle = fopen(pic_name, "wb+");
        if (file_handle != NULL) {           
            void *data = out_bufs[i]->GetHostData(0);
            int size = out_bufs[i]->GetSurfaceParams(0)->data_size;

            fwrite(data, size, 1, file_handle);
            fclose(file_handle);
        }
    }
#endif
    auto model_info_priv = model_->GetModelInfoPriv();
    std::string model_type = model_info_priv->GetModelType();
    std::string net_type = model_info_priv->GetNetType();
    std::vector<std::string> labels = model_info_priv->GetLabels();

    const DataLayout output_layout =  model_->OutputLayout(0);
    auto order = output_layout.order;

    auto scale = model_->OutputScale();

    gddeploy::InferResult result;  

    if (true == pack->data[0]->HasMetaValue()){
        result = pack->data[0]->GetMetaData<gddeploy::InferResult>();
    }

    float thresh = model_info_priv->GetConfidenceThresh();
    if (thresh < 0.001)
        thresh = 0.3;

    std::string product = model_info_priv->GetProductType();

    std::vector<FrameInfo> frame_info;
    for (auto & data : pack->data){
        auto surf = data->GetLref<gddeploy::BufSurfWrapperPtr>();
        int img_w = surf->GetSurfaceParams(0)->width;
        int img_h = surf->GetSurfaceParams(0)->height;

        FrameInfo fi = {0, 0, img_w, img_h};
        frame_info.emplace_back(fi);
    }

    gddeploy::PostParam post_param;
    post_param.conf_thresh = model_info_priv->GetConfidenceThresh();
    post_param.iou_thresh = 0.45; //model_info_priv->GetIouThresh();
    post_param.output_scale = model_->OutputScale();
    post_param.output_zp = model_->OutputZp();
    post_param.labels = model_info_priv->GetLabels();
    // auto t0 = std::chrono::high_resolution_clock::now();
    if (model_type == "classification" && net_type == "ofa") {
        ClassifyDecodeOutputNCHW(out_bufs, result, thresh, frame_info, model_);
    }else if (model_type == "pose" && net_type == "yolox") {
        if (out_bufs.size() == 1) {
            PoseDecodeOutput1NCHW(out_bufs, result, thresh, frame_info, model_);
        } else {
            if (order == DimOrder::NCHW){
                PoseDecodeOutputNCHW(out_bufs, result, thresh, frame_info, model_);
            } else if (order == DimOrder::NHWC){
                PoseDecodeOutputNHWC(out_bufs, result, thresh, frame_info, model_);
            }
        }
    } else if (model_type == "pose" && net_type == "rtmpose"){
        RTMPoseDecodeOutput(out_bufs, result, post_param, frame_info, model_);
    } else if (model_type == "detection" && net_type == "yolo"){
        if (order == DimOrder::NCHW){
            if (product == "Intel"){
                Yolov5DecodeOutputNCHWSigmoid(out_bufs, result, thresh, frame_info, model_);
            } else {
                // Yolov5DecodeOutputNCHW(out_bufs, result, post_param, frame_info, model_);
                Yolov5DecodeOutput(out_bufs, result, post_param, frame_info, model_);
            }
        } else if (order == DimOrder::NHWC){
            Yolov5DecodeOutput(out_bufs, result, post_param, frame_info, model_);
            // Yolov5DecodeOutputNHWC(out_bufs, result, thresh, frame_info, model_);
        }
    } else if (model_type == "detection" && net_type == "yolov6"){
        Yolov6DecodeOutput(out_bufs, result, post_param, frame_info, model_);
    } else if (model_type == "segmentation" && net_type == "OCRNet"){
        SegDecodeOutputNCHW(out_bufs, result, post_param, frame_info, model_);
    } else if (model_type == "action" && net_type == "tsn_gddi") {
        ActionDecodeOutputNCHW(out_bufs, result, thresh, frame_info, model_);
    } else if (model_type == "image-retrieval" && net_type == "dolg") {
        ImageRetrievalDecodeOutputNCHW(out_bufs, result, thresh, frame_info, model_);
    } else if (model_type == "image-retrieval" && net_type == "arcface") {
        FaceRetrievalDecodeOutputNCHW(out_bufs, result, thresh, frame_info, model_);
    } else if (model_type == "ocr" && net_type == "ocr_det") {
        OcrDetectDecodeOutputNCHW(out_bufs, result, thresh, frame_info, model_);
    } else if (model_type == "ocr" && net_type == "ocr_rec" ||
             model_type == "ocr" && net_type == "resnet31v2ctc") {
        OcrRetrievalDecodeOutputNCHW(out_bufs, result, thresh, frame_info, model_);
    }
    // auto t1 = std::chrono::high_resolution_clock::now();
    // printf("!!!!!!!!!!!!!!!post time: %d us\n", std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count());
    for (auto result_type : result.result_type){
        if (result_type == GDD_RESULT_TYPE_DETECT){
            for (uint32_t i = 0; i < result.detect_result.detect_imgs.size(); i++){
                for (auto &obj : result.detect_result.detect_imgs[i].detect_objs) {
                    if (obj.class_id < labels.size())
                        obj.label = labels[obj.class_id];
                }
            }
        }
    }
    // auto t1 = std::chrono::high_resolution_clock::now();
    // printf("inference time: %d us\n", std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count());

    // TODO: 把result分拆不同batch，分别设置pack->data[batch_idx]
    // TODO: 不同算法后处理只有一个，差异点根据map类型的param传入，处理函数去除调整自身的decode逻辑
    for (auto result_type : result.result_type){
        if (result_type == GDD_RESULT_TYPE_CLASSIFY || result_type == GDD_RESULT_TYPE_ACTION){
            for (uint32_t i = 0; i < result.classify_result.detect_imgs.size(); i++){
                gddeploy::InferResult result_tmp;
                result_tmp.result_type.emplace_back(result_type);
                result_tmp.classify_result.detect_imgs.emplace_back(result.classify_result.detect_imgs[i]);
                result_tmp.classify_result.batch_size = 1;

                pack->data[i]->SetMetaData(std::move(result_tmp));
            }
        } else if (result_type == GDD_RESULT_TYPE_DETECT){
            for (uint32_t i = 0; i < result.detect_result.detect_imgs.size(); i++){
                gddeploy::InferResult result_tmp;
                result_tmp.result_type.emplace_back(GDD_RESULT_TYPE_DETECT);
                result_tmp.detect_result.detect_imgs.emplace_back(result.detect_result.detect_imgs[i]);
                result_tmp.detect_result.batch_size = 1;

                pack->data[i]->SetMetaData(std::move(result_tmp));
            }
        } else if (result_type == GDD_RESULT_TYPE_DETECT_POSE){
            for (uint32_t i = 0; i < result.detect_pose_result.detect_imgs.size(); i++){
                gddeploy::InferResult result_tmp;
                result_tmp.result_type.emplace_back(GDD_RESULT_TYPE_DETECT_POSE);
                result_tmp.detect_pose_result.detect_imgs.emplace_back(result.detect_pose_result.detect_imgs[i]);
                result_tmp.detect_pose_result.batch_size = 1;

                pack->data[i]->SetMetaData(std::move(result_tmp));
            }
        } else if (result_type == GDD_RESULT_TYPE_SEG){
            for (uint32_t i = 0; i < result.detect_pose_result.detect_imgs.size(); i++){
                gddeploy::InferResult result_tmp;
                result_tmp.result_type.emplace_back(GDD_RESULT_TYPE_SEG);
                result_tmp.seg_result.seg_imgs.emplace_back(result.seg_result.seg_imgs[i]);
                result_tmp.seg_result.batch_size = 1;

                pack->data[i]->SetMetaData(std::move(result_tmp));
            }
        } else if (result_type == GDD_RESULT_TYPE_OCR_DETECT){
            for (uint32_t i = 0; i < result.ocr_detect_result.ocr_detect_imgs.size(); i++){
                gddeploy::InferResult result_tmp;
                result_tmp.result_type.emplace_back(GDD_RESULT_TYPE_OCR_DETECT);
                result_tmp.ocr_detect_result.ocr_detect_imgs.emplace_back(result.ocr_detect_result.ocr_detect_imgs[i]);
                result_tmp.ocr_detect_result.batch_size = 1;

                pack->data[i]->SetMetaData(std::move(result_tmp));
            }
        } else if (result_type == GDD_RESULT_TYPE_OCR_RETRIEVAL){
            for (uint32_t i = 0; i < result.ocr_rec_result.ocr_rec_imgs.size(); i++){
                gddeploy::InferResult result_tmp;
                result_tmp.result_type.emplace_back(GDD_RESULT_TYPE_OCR_RETRIEVAL);
                result_tmp.ocr_rec_result.ocr_rec_imgs.emplace_back(result.ocr_rec_result.ocr_rec_imgs[i]);
                result_tmp.ocr_rec_result.batch_size = 1;

                pack->data[i]->SetMetaData(std::move(result_tmp));
            }
        } else if (result_type == GDD_RESULT_TYPE_FACE_RETRIEVAL){
            for (uint32_t i = 0; i < result.face_retrieval_result.face_retrieval_imgs.size(); i++){
                gddeploy::InferResult result_tmp;
                result_tmp.result_type.emplace_back(GDD_RESULT_TYPE_FACE_RETRIEVAL);
                result_tmp.face_retrieval_result.face_retrieval_imgs.emplace_back(result.face_retrieval_result.face_retrieval_imgs[i]);
                result_tmp.face_retrieval_result.batch_size = 1;

                pack->data[i]->SetMetaData(std::move(result_tmp));
            }
        } else if (result_type == GDD_RESULT_TYPE_IMAGE_RETRIEVAL){
            for (uint32_t i = 0; i < result.image_retrieval_result.image_retrieval_imgs.size(); i++){
                gddeploy::InferResult result_tmp;
                result_tmp.result_type.emplace_back(GDD_RESULT_TYPE_IMAGE_RETRIEVAL);
                result_tmp.image_retrieval_result.image_retrieval_imgs.emplace_back(result.image_retrieval_result.image_retrieval_imgs[i]);
                result_tmp.image_retrieval_result.batch_size = 1;

                pack->data[i]->SetMetaData(std::move(result_tmp));
            }
        }
    }
    // pack->data[0]->Set(result);
    // pack->data[1]->Set(result);
    // pack->data[2]->Set(result);
    // pack->data[3]->Set(result);
    // pack->data[0]->SetMetaData(result);

    // auto t1 = std::chrono::high_resolution_clock::now();
    // printf("post time: %d us\n", std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count());

    return gddeploy::Status::SUCCESS;
}