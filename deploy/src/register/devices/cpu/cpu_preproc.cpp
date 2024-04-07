#include "cpu_preproc.h"
#include <memory>
#include <string>
#include <vector>

#include "core/mem/buf_surface.h"
#include "core/preprocess.h"
#include "opencv2/opencv.hpp"

#include "core/model.h"
#include "core/result_def.h"
#include "core/mem/buf_surface_util.h"
#include "common/type_convert.h"

#include "preproc/preproc_opencv.h"
#include "preproc/action/action.h"

// #include "json.hpp"

namespace gddeploy{   
class CPUPreProcPriv{
public:
    CPUPreProcPriv(ModelPtr model):model_(model){
        for (int i = 0; i < model->InputNum(); i++){
            auto shape = model_->InputShape(i);
            const DataLayout input_layout =  model->InputLayout(0);
            auto order = input_layout.order;
            if (order == DimOrder::NCHW){
                model_h_ = shape[2];
                model_w_ = shape[3];
            }else if (order == DimOrder::NHWC){
                model_h_ = shape[1];
                model_w_ = shape[2];
            }
        }
    }
    ~CPUPreProcPriv(){
        for (auto pool : pools_){
            pool->DestroyPool();
        }
        
    }

    int Init(std::string config); 

    int PreProc(cv::Mat &input_mat, cv::Mat &output_mat, InferResult &results);
    int PreProcAction(PackagePtr pack, BufSurfWrapperPtr &buf);

    int SetModel(ModelPtr model){
        if (model == nullptr){
            return -1;
        }else{
            model_ = model;
        }
        return 0;
    }

    BufSurfWrapperPtr RequestBuffer(){
        BufSurfWrapperPtr buf = pools_[0]->GetBufSurfaceWrapper();

        return buf;
    }

    int GetModelWidth(){
        return model_w_;
    }

    int GetModelHeight(){
        return model_h_;
    }

    std::string GetModelType(){
        ModelPropertiesPtr mp = model_->GetModelInfoPriv();
        return mp->GetModelType();
    }

private:
    ModelPtr model_;
    int model_h_;
    int model_w_;

    std::string config_;

    action::ActionPreProc action_preproc_;
    action::ActionParam param_;

    std::vector<BufPool*> pools_;
    // cv::Mat resize_mat_;
};
}

using namespace gddeploy;

int CreatePool(ModelPtr model, BufPool *pool, BufSurfaceMemType mem_type, int block_count) {
    // 解析model，获取必要结构
    const DataLayout input_layout =  model->InputLayout(0);
    auto dtype = input_layout.dtype;
    auto order = input_layout.order;
    int data_size = 0;
    if (dtype == DataType::INT8 || dtype == DataType::UINT8){
        data_size = sizeof(uint8_t);
    // }else if (dtype == DataType::FLOAT16 || dtype == DataType::UINT16 || dtype == DataType::INT16){
    //     data_size = sizeof(uint16_t);
    }else if (dtype == DataType::FLOAT16 || dtype == DataType::UINT16 || dtype == DataType::INT16 || \
              dtype == DataType::FLOAT32 || dtype == DataType::INT32){
        data_size = sizeof(uint32_t);
    }

    int model_h, model_w, model_c, model_b;
    auto shape = model->InputShape(0);
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
    create_params.mem_type = mem_type;
    create_params.force_align_1 = 1;  // to meet mm's requirement
    create_params.device_id = 0;
    create_params.batch_size = model_b;
    create_params.size = model_h * model_w * model_c;
    create_params.size *= data_size;
    create_params.width = model_w;
    create_params.height = model_h;
    create_params.color_format = GDDEPLOY_BUF_COLOR_FORMAT_RGB;

    
    if (pool->CreatePool(&create_params, block_count) < 0) {
        return -1;
    }
    return 0;
}

int CPUPreProcPriv::Init(std::string config){
    size_t i_num = model_->InputNum();
    for (size_t i_idx = 0; i_idx < i_num; ++i_idx) {
        BufPool *pool = new BufPool;
        CreatePool(model_, pool, GDDEPLOY_BUF_MEM_SYSTEM, 3);
        pools_.emplace_back(pool);
    }

    // TODO: 解析config 填充参数
    action_preproc_.Init(param_);

    config_ = config;
    return 0;
}

int CPUPreProcPriv::PreProc(cv::Mat &input_mat, cv::Mat &output_mat, InferResult &results)
{
    ModelPropertiesPtr mp = model_->GetModelInfoPriv();
    std::string model_type = mp->GetModelType();
    std::string net_type = mp->GetNetType();

    std::string product = mp->GetProductType();

    const DataLayout input_layout =  model_->InputLayout(0);
    auto order = input_layout.order;

#if 0
    std::vector<std::pair<std::string, gddeploy::any>> ops;
    if (model_type == "classification" && net_type == "ofa"){
        gddeploy::transform::ComposeResizeParam resize_param = {
            .in_w = input_mat.cols,
            .in_h = input_mat.rows,
            .out_w = model_w_,
            .out_h = model_h_,
            .type = gddeploy::transform::ResizeProcessType::RESIZE_PT_DEFAULT,
            .padding_num = 0,
        };
        gddeploy::transform::ComposeNormalizeParam norn_param = {
            .mean = {123.675, 116.28, 103.53},
            .std = {1 / 58.395, 1 / 57.12, 1 / 57.375},
        };
        ops.emplace_back(std::pair<std::string, gddeploy::any>("bgr2rgb", true));
        ops.emplace_back(std::pair<std::string, gddeploy::any>("resize", resize_param));
        ops.emplace_back(std::pair<std::string, gddeploy::any>("normalize", norn_param));
        ops.emplace_back(std::pair<std::string, gddeploy::any>("hwc2chw", true));
        if (order == DimOrder::NHWC){
            for (auto &iter : ops){
                if (iter.first == "hwc2chw"){
                    iter.second = false;
                }
            }
        }
    } else if (model_type == "detection" && net_type == "yolo"){
        gddeploy::transform::ComposeResizeParam resize_param = {
            .in_w = input_mat.cols,
            .in_h = input_mat.rows,
            .out_w = model_w_,
            .out_h = model_h_,
            .type = gddeploy::transform::ResizeProcessType::RESIZE_PT_CENTER,
            .padding_num = 114,
        };
        gddeploy::transform::ComposeNormalizeParam norn_param = {
            .mean = {0, 0, 0},
            .std = {1 / 255, 1 / 255, 1 / 255},
        };
        ops.emplace_back(std::pair<std::string, gddeploy::any>("bgr2rgb", true));
        ops.emplace_back(std::pair<std::string, gddeploy::any>("resize", resize_param));
        ops.emplace_back(std::pair<std::string, gddeploy::any>("normalize", norn_param));
        ops.emplace_back(std::pair<std::string, gddeploy::any>("hwc2chw", true));
        

        if (product == "Tsingmicro"){
            for (auto &iter : ops){
                if (iter.first == "hwc2chw"){
                    iter.second = false;
                }
            }
            ops.insert(ops.begin()+2, std::pair<std::string, gddeploy::any>("float", true));
        } else if (product == "Cambricon"){
            for (auto &iter : ops){
                if (iter.first == "hwc2chw"){
                    iter.second = false;
                }
            }
        } else if (product == "Intel"){
            ops.erase(ops.begin()+2);
        } 
    } else if (model_type == "pose" && net_type == "yolox"){
        gddeploy::transform::ComposeResizeParam resize_param = {
            .in_w = input_mat.cols,
            .in_h = input_mat.rows,
            .out_w = model_w_,
            .out_h = model_h_,
            .type = gddeploy::transform::ResizeProcessType::RESIZE_PT_LEFT_TOP,
            .padding_num = 114,
        };
        ops.emplace_back(std::pair<std::string, gddeploy::any>("bgr2rgb", true));
        ops.emplace_back(std::pair<std::string, gddeploy::any>("resize", resize_param));
        ops.emplace_back(std::pair<std::string, gddeploy::any>("hwc2chw", true));
        
        if (product == "Tsingmicro"){
            for (auto &iter : ops){
                if (iter.first == "hwc2chw"){
                    iter.second = false;
                }
            }
            ops.insert(ops.begin()+2, std::pair<std::string, gddeploy::any>("float", true));
        } else if (product == "Cambricon"){
            for (auto &iter : ops){
                if (iter.first == "hwc2chw"){
                    iter.second = false;
                }
            }
            ops.emplace_back(std::pair<std::string, gddeploy::any>("float", true));
        }
    } else if (model_type == "segmentation" && net_type == "OCRNet"){
        gddeploy::transform::ComposeResizeParam resize_param = {
            .in_w = input_mat.cols,
            .in_h = input_mat.rows,
            .out_w = model_w_,
            .out_h = model_h_,
            .type = gddeploy::transform::ResizeProcessType::RESIZE_PT_LEFT_TOP,
            .padding_num = 114,
        };

        gddeploy::transform::ComposeNormalizeParam norn_param = {
            .mean = {123.675, 116.28, 103.53},
            .std = {1 / 58.395, 1 / 57.12, 1 / 57.375},
        };

        ops.emplace_back(std::pair<std::string, gddeploy::any>("bgr2rgb", true));
        ops.emplace_back(std::pair<std::string, gddeploy::any>("resize", resize_param));
        ops.emplace_back(std::pair<std::string, gddeploy::any>("normalize", norn_param));
        ops.emplace_back(std::pair<std::string, gddeploy::any>("hwc2chw", true));
        if (product == "Tsingmicro"){
            for (auto &iter : ops){
                if (iter.first == "hwc2chw"){
                    iter.second = false;
                }
            }
            ops.insert(ops.begin()+2, std::pair<std::string, gddeploy::any>("float", true));
        } else if (product == "Cambricon"){
            for (auto &iter : ops){
                if (iter.first == "hwc2chw"){
                    iter.second = false;
                }
            }
        }
    // }else if (net_type == "action"){
    //     Preprocyolov5(input_mat, output_mat, model_h_, model_w_);
    } else if (model_type == "image-retrieval" && net_type == "dolg"){
        gddeploy::transform::ComposeResizeParam resize_param = {
            .in_w = input_mat.cols,
            .in_h = input_mat.rows,
            .out_w = model_w_,
            .out_h = model_h_,
            .type = gddeploy::transform::ResizeProcessType::RESIZE_PT_CENTER,
            .padding_num = 114,
        };

        gddeploy::transform::ComposeNormalizeParam norn_param = {
            .mean = {123.675, 116.28, 103.53},
            .std = {1 / 58.395, 1 / 57.12, 1 / 57.375},
        };

        ops.emplace_back(std::pair<std::string, gddeploy::any>("bgr2rgb", true));
        ops.emplace_back(std::pair<std::string, gddeploy::any>("resize", resize_param));
        ops.emplace_back(std::pair<std::string, gddeploy::any>("normalize", norn_param));
        ops.emplace_back(std::pair<std::string, gddeploy::any>("hwc2chw", true));
        if (product == "Tsingmicro"){
            for (auto &iter : ops){
                if (iter.first == "hwc2chw"){
                    iter.second = false;
                }
            }
            ops.insert(ops.begin()+2, std::pair<std::string, gddeploy::any>("float", true));
        } else if (product == "Cambricon"){
            for (auto &iter : ops){
                if (iter.first == "hwc2chw"){
                    iter.second = false;
                }
            }
        }
    } else if (model_type == "image-retrieval" && net_type == "arcface"){
        gddeploy::transform::ComposeResizeParam resize_param = {
            .in_w = input_mat.cols,
            .in_h = input_mat.rows,
            .out_w = model_w_,
            .out_h = model_h_,
            .type = gddeploy::transform::ResizeProcessType::RESIZE_PT_DEFAULT,
            .padding_num = 0,
        };

        gddeploy::transform::ComposeNormalizeParam norn_param = {
            .mean = {127.5, 127.5, 127.5},
            .std = {1 / 127.5, 1 / 127.5, 1 / 127.5},
        };

        ops.emplace_back(std::pair<std::string, gddeploy::any>("bgr2rgb", true));
        ops.emplace_back(std::pair<std::string, gddeploy::any>("resize", resize_param));
        ops.emplace_back(std::pair<std::string, gddeploy::any>("normalize", norn_param));
        ops.emplace_back(std::pair<std::string, gddeploy::any>("hwc2chw", true));
        if (product == "Tsingmicro"){
            for (auto &iter : ops){
                if (iter.first == "hwc2chw"){
                    iter.second = false;
                }
            }
            ops.insert(ops.begin()+2, std::pair<std::string, gddeploy::any>("float", true));
        } else if (product == "Cambricon"){
            for (auto &iter : ops){
                if (iter.first == "hwc2chw"){
                    iter.second = false;
                }
            }
        }
    } else if (model_type == "ocr" && net_type == "ocr_rec"){
        gddeploy::transform::ComposeResizeParam resize_param = {
            .in_w = input_mat.cols,
            .in_h = input_mat.rows,
            .out_w = model_w_,
            .out_h = model_h_,
            .type = gddeploy::transform::ResizeProcessType::RESIZE_PT_DEFAULT,
            .padding_num = 114,
        };

        gddeploy::transform::ComposeNormalizeParam norn_param = {
            .mean = {103.53, 116.28, 123.675},
            .std = {1 / 57.375, 1 / 57.12, 1 / 58.395},
        };

        ops.emplace_back(std::pair<std::string, gddeploy::any>("bgr2rgb", true));
        ops.emplace_back(std::pair<std::string, gddeploy::any>("resize", resize_param));
        ops.emplace_back(std::pair<std::string, gddeploy::any>("normalize", norn_param));
        ops.emplace_back(std::pair<std::string, gddeploy::any>("hwc2chw", true));
        if (product == "Tsingmicro"){
            for (auto &iter : ops){
                if (iter.first == "hwc2chw"){
                    iter.second = false;
                }
            }
            ops.insert(ops.begin()+2, std::pair<std::string, gddeploy::any>("float", true));
        } else if (product == "Cambricon"){
            for (auto &iter : ops){
                if (iter.first == "hwc2chw"){
                    iter.second = false;
                }
            }
        }
    } else if (model_type == "ocr" && net_type == "ocr_rec" || model_type == "ocr" && net_type == "resnet31v2ctc"){
        gddeploy::transform::ComposeResizeParam resize_param = {
            .in_w = input_mat.cols,
            .in_h = input_mat.rows,
            .out_w = model_w_,
            .out_h = model_h_,
            .type = gddeploy::transform::ResizeProcessType::RESIZE_PT_DEFAULT,
            .padding_num = 114,
        };

        gddeploy::transform::ComposeNormalizeParam norn_param = {
            .mean = {123.675, 116.28, 103.53},
            .std = {1 / 58.395, 1 / 57.12, 1 / 57.375},
        };

        ops.emplace_back(std::pair<std::string, gddeploy::any>("bgr2rgb", true));
        ops.emplace_back(std::pair<std::string, gddeploy::any>("resize", resize_param));
        ops.emplace_back(std::pair<std::string, gddeploy::any>("normalize", norn_param));
        ops.emplace_back(std::pair<std::string, gddeploy::any>("hwc2chw", true));
        if (product == "Tsingmicro"){
            for (auto &iter : ops){
                if (iter.first == "hwc2chw"){
                    iter.second = false;
                }
            }
            ops.insert(ops.begin()+2, std::pair<std::string, gddeploy::any>("float", true));
        } else if (product == "Cambricon"){
            for (auto &iter : ops){
                if (iter.first == "hwc2chw"){
                    iter.second = false;
                }
            }
        }
    }
    gddeploy::transform::Compose(input_mat, output_mat, ops);
    // cv::imwrite("/data/gddeploy/preds/pre_test.jpg", output_mat);
#else 
    if (model_type == "classification" && net_type == "ofa"){
        if (order == DimOrder::NCHW){
            PreprocClassify(input_mat, output_mat, model_h_, model_w_);
        } else if (order == DimOrder::NHWC){
            PreprocClassifyNHWC(input_mat, output_mat, model_h_, model_w_);
        }
    } else if (model_type == "detection" && net_type == "yolo" || model_type == "detection" && net_type == "yolov6"){
        if (product == "Tsingmicro" || product == "Rockchip"){
            PreprocYolov5Ts(input_mat, output_mat, model_h_, model_w_);
        } else if (product == "Cambricon"){
            PreprocYolov5NHWC(input_mat, output_mat, model_h_, model_w_);
        } else if (product == "Intel"){
            PreprocYolov5Intel(input_mat, output_mat, model_h_, model_w_);
        } else {
            PreprocYolov5(input_mat, output_mat, model_h_, model_w_);
        }
    } else if (model_type == "pose" && net_type == "yolox"){
        if (order == DimOrder::NCHW){
        PreprocYolox(input_mat, output_mat, model_h_, model_w_);
        } else if (order == DimOrder::NHWC){
            PreprocYoloxNHWC(input_mat, output_mat, model_h_, model_w_);
        }
    } else if (model_type == "pose" && net_type == "rtmpose"){
        PreprocRTMPoseNHWC(input_mat, output_mat, model_h_, model_w_, results);
    } else if (model_type == "segmentation" && net_type == "OCRNet"){
        PreprocSeg(input_mat, output_mat, model_h_, model_w_);
    // }else if (net_type == "action"){
    //     Preprocyolov5(input_mat, output_mat, model_h_, model_w_);
    } else if (model_type == "image-retrieval" && net_type == "dolg"){
        PreprocImageRetrieval(input_mat, output_mat, model_h_, model_w_);
    } else if (model_type == "image-retrieval" && net_type == "arcface"){
        PreprocFaceRetrieval(input_mat, output_mat, model_h_, model_w_);
    } else if (model_type == "ocr" && net_type == "ocr_det"){
        PreprocOcrDet(input_mat, output_mat, model_h_, model_w_);
    } else if (model_type == "ocr" && net_type == "ocr_rec" || model_type == "ocr" && net_type == "resnet31v2ctc"){
        PreprocOcrRec(input_mat, output_mat, model_h_, model_w_);
    }
#endif

    return 0;
}

bool is_element_in_vector(std::vector<int> v, int element){
    std::vector<int>::iterator it = find(v.begin(),v.end(),element);
    if (it!=v.end()){
        return true;
    } else {
        return false;
    }
}

int CPUPreProcPriv::PreProcAction(PackagePtr pack, BufSurfWrapperPtr &buf)
{
    const gddeploy::InferResult& postproc_results = pack->data[0]->GetLref<gddeploy::InferResult>();
    
    if (is_element_in_vector(postproc_results.result_type, (int)gddeploy::GDD_RESULT_TYPE_DETECT_POSE)){
        action_preproc_.Clear();
        std::vector<DetectPoseObject> detect_objs;
        detect_objs.reserve(postproc_results.detect_pose_result.detect_imgs.size());
        for (auto &obj : postproc_results.detect_pose_result.detect_imgs[0].detect_objs) {
            action::InputData in;
            action::OutputData out;
            
            in.img_h = postproc_results.detect_pose_result.detect_imgs[0].img_h;
            in.img_w = postproc_results.detect_pose_result.detect_imgs[0].img_w;
            in.trace_id = 0;
            in.img_id = postproc_results.detect_pose_result.detect_imgs[0].img_id;
            in.pose_data = {
                .m_iClassId = obj.score,
                .m_fProb = obj.class_id,
            };

            in.pose_data.m_bbox[0] = obj.bbox.x;
            in.pose_data.m_bbox[1] = obj.bbox.y;
            in.pose_data.m_bbox[2] = obj.bbox.w;
            in.pose_data.m_bbox[3] = obj.bbox.h;

            for (uint32_t i = 0; i < 17; i++){
                action::PoseKeyPoint point = {obj.point[i].x, obj.point[i].y, obj.point[i].score};
                in.pose_data.point.emplace_back(point);
            }

            // action preprocess process 
            if ( -1 == action_preproc_.Process(in, out)){
                continue;
            }

            if (out.gaussian_blur_data.size() == 0){
                continue;
            }

            if (out.gaussian_blur_data.size()){
                std::string prefix = "action_out_"+std::to_string(in.img_id)+"_";
                action_preproc_.DrawPoint("/data/preds/", prefix, out);
            }

            void *data = buf->GetHostData(0, 0);
            memcpy(data, out.gaussian_blur_data.data(),out.gaussian_blur_data.size()*sizeof(float));
        }
    }

    return 0;
}

#include "core/mem/buf_surface_util.h"

Status CPUPreProc::Init(std::string config) noexcept
{ 
    printf("CPU Init\n");

    //TODO: 这里要补充解析配置，得到网络类型等新型
    // if (false == HaveParam("model_info")){
    //     return gddeploy::Status::INVALID_PARAM;
    // }
    // ModelPtr model = GetParam<ModelPtr>("model_info");

    // priv_ = std::make_shared<CPUPreProcPriv>(model);

    priv_->Init(config);

    return gddeploy::Status::SUCCESS; 
}

Status CPUPreProc::Init(ModelPtr model, std::string config)
{
    priv_ = std::make_shared<CPUPreProcPriv>(model);

    priv_->Init(config);

    model_ = model;

    return gddeploy::Status::SUCCESS; 
}

Status CPUPreProc::Process(PackagePtr pack) noexcept
{
    BufSurfWrapperPtr buf = priv_->RequestBuffer();    

    auto model_type = priv_->GetModelType();
    if (model_type == "action"){
        priv_->PreProcAction(pack, buf);
    } else {
        for (auto &data : pack->data){
            auto surf = data->GetLref<gddeploy::BufSurfWrapperPtr>();
            BufSurface *surface = surf->GetBufSurface();
            cv::Mat input_mat;
            convertBufSurface2Mat(input_mat, data->GetLref<gddeploy::BufSurfWrapperPtr>());
            // cv::imwrite("./input_mat.jpg", input_mat);

            InferResult results;
            if (true == data->HasMetaValue())
                results = data->GetMetaData<gddeploy::InferResult>();

            int model_w = priv_->GetModelWidth();
            int model_h = priv_->GetModelHeight();
            cv::Mat output_mat(model_w, model_h, CV_8UC3, (uint8_t *)buf->GetData(0, 0));

            // auto t0 = std::chrono::high_resolution_clock::now();
            priv_->PreProc(input_mat, output_mat, results);
            // auto t1 = std::chrono::high_resolution_clock::now();
            // printf("!!!!!!!!!!!!!!!!cpu preproc time: %d us\n", std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count());

        }
    }

    std::shared_ptr<InferData> infer_data = std::make_shared<InferData>();
    infer_data->Set(std::move(buf));
    
    // pack->data.clear();
    pack->predict_io = infer_data;

    return gddeploy::Status::SUCCESS; 
}
