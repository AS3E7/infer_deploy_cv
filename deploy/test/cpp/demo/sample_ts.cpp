#include <iostream>
#include <ts_rne_c_api.h>
#include <ts_rne_log.h>
#include <ts_rne_version.h>
#include <ts_rne_device.h>

#include <vector>
#include <fstream>

#include "opencv2/opencv.hpp"

#define TS_MPI_TRP_RNE_MASK_BITS(m) ((1ll << (m)) - 1)

#define TS_MPI_TRP_RNE_W_ALIGN_BYTES_NUM (4)

typedef struct {
    float x;
    float y;
    float w;
    float h;
}Bbox;

typedef struct {
  // int label;
  int detect_id;
  int class_id;
  float score;
  Bbox bbox;
}DetectObject;  // struct DetectObject


void get_rect(int img_w, int img_h,int model_w, int model_h, Bbox *bbox) {
    int w, h, x, y;
    float r_w = model_w / (img_w * 1.0);
    float r_h = model_h / (img_h * 1.0);

    if (r_h > r_w) 
    {
        bbox->x = bbox->x / r_w;
        bbox->w = bbox->w / r_w;
        bbox->h = bbox->h / r_w;

        h = r_w * img_h;
        y = (model_h - h) / 2;
        bbox->y = (bbox->y - y) / r_w;
    }else{
        bbox->y = bbox->y / r_h;
        bbox->w = bbox->w / r_h;
        bbox->h = bbox->h / r_h;

        w = r_h * img_w;
        x = (model_w - w) / 2;
        bbox->x = (bbox->x - x) / r_h;
    }

    bbox->x = std::max(0.0f, bbox->x);
    bbox->y = std::max(0.0f, bbox->y);

    bbox->w = std::min((float)bbox->x+img_w, bbox->x+bbox->w) - bbox->x;
    bbox->h = std::min((float)bbox->x+img_h, bbox->y+bbox->h) - bbox->y;
}

std::vector<DetectObject> nms(std::vector<DetectObject> objInfos, float conf_thresh)
{
    std::sort(objInfos.begin(), objInfos.end(), [](DetectObject lhs, DetectObject rhs)
              { return lhs.score > rhs.score; });
    if (objInfos.size() > 1000)
    {
        objInfos.erase(objInfos.begin() + 1000, objInfos.end());
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

            if (iou_value > conf_thresh)
                it = objInfos.erase(it);
            else
                it++; 
        }
        objInfos.erase(objInfos.begin());
    }

    return result;
}

int preproc_yolov5(cv::Mat input_mat, cv::Mat &output_mat, int model_h, int model_w)
{
    cv::Mat img;

    if (input_mat.channels() == 1)
        cv::cvtColor(input_mat, img, cv::COLOR_GRAY2RGB);
    else
        cv::cvtColor(input_mat, img, cv::COLOR_BGR2RGB);

    int w, h, x, y;

    int input_mat_w_ = img.cols;
    int input_mat_h_ = img.rows;
    float r_w = model_h / (input_mat_w_*1.0);
    float r_h = model_w / (input_mat_h_*1.0);

    if (r_h > r_w) {
        w = model_h;
        h = r_w * input_mat_h_;
        x = 0;
        y = (model_w - h) / 2;
    } else {
        w = r_h * input_mat_w_;
        h = model_w;
        x = (model_h - w) / 2;
        y = 0;
    }
    cv::Mat re(h, w, CV_8UC3);
    cv::Mat out(model_w, model_h, CV_8UC3, cv::Scalar(114, 114, 114));

    cv::resize(img, re, re.size(), 0, 0, cv::INTER_LINEAR);
    
    re.copyTo(out(cv::Rect(x, y, re.cols, re.rows)));

    // out.convertTo(output_mat, CV_32FC3, 1.0);
    output_mat = out;

    return 0;
}

int main()
{
    TS_MPI_TRP_RNE_SetLogLevel(RNE_LOG_INFO);
    TS_MPI_TRP_RNE_Info("current log level : %d\n", TS_MPI_TRP_RNE_GetLogLevel());
    TS_MPI_TRP_RNE_Info("current lib version :%s\n", TS_MPI_TRP_RNE_GetSdkVersion());
    TS_MPI_TRP_RNE_Info("main start...\n");

    /* 打开RNE设备 */
    TS_S32 ret = TS_MPI_TRP_RNE_OpenDevice(NULL, NULL);
    if (ret) {
        TS_MPI_TRP_RNE_Error("open device error!\n");
        return ret;
    }

    // 1. load model
    std::ifstream cfg_file("/root/gddeploy/data/models/helmet/_yolov5_8_r.cfg", std::ios::in|std::ios::binary);
    std::ifstream weight_file("/root/gddeploy/data/models/helmet/_yolov5_8_r.weight", std::ios::in|std::ios::binary);
    std::ifstream weight_param_file("/root/gddeploy/data/models/helmet/_yolov5_8_r.mweight", std::ios::in|std::ios::binary);

    cfg_file.seekg(0, std::ios::end);
    int cfg_length = cfg_file.tellg();   
    cfg_file.seekg(0, std::ios::beg);    
    char* cfg_buffer = new char[cfg_length];    
    cfg_file.read(cfg_buffer, cfg_length); 

    weight_file.seekg(0, std::ios::end);
    int weight_length = weight_file.tellg();
    weight_file.seekg(0, std::ios::beg);  
    char* weight_buffer = new char[weight_length];
    weight_file.read(weight_buffer, weight_length); 

    weight_param_file.seekg(0, std::ios::end);
    int weight_param_length = weight_param_file.tellg();
    weight_param_file.seekg(0, std::ios::beg);  
    char* weight_param_buffer = new char[weight_param_length];
    weight_param_file.read(weight_param_buffer, weight_param_length); 

    RNE_NET_S *net = new RNE_NET_S;
    memset(net, 0, sizeof(RNE_NET_S));
    net->u8pGraph = (TS_U8 *)cfg_buffer;
    net->u8pParams = (TS_U8 *)weight_buffer;
    // net->u8pWeight = (TS_U8 *)weight_buffer;
    net->eInputType = RNE_NET_INPUT_TYPE_INT8_HWC;

    TS_U8 *paramStride = (TS_U8 *)TS_MPI_TRP_RNE_AllocLinearMem((TS_SIZE_T)(weight_length + TS_MPI_TRP_RNE_W_ALIGN_BYTES_NUM));
    // if (((TS_SIZE_T)net->u8pParams & (TS_MPI_TRP_RNE_W_ALIGN_BYTES_NUM - 1)) != 0) {
    //     TS_U8 *paramStride = (TS_U8 *)TS_MPI_TRP_RNE_Alloc(weight_length + TS_MPI_TRP_RNE_W_ALIGN_BYTES_NUM);
    if (NULL == paramStride) {
        TS_MPI_TRP_RNE_Error("insufficient memory!\n");
        return -1;
    }
    TS_SIZE_T addr = (TS_SIZE_T)paramStride;
    addr += TS_MPI_TRP_RNE_W_ALIGN_BYTES_NUM - 1;
    addr &= ~(TS_MPI_TRP_RNE_W_ALIGN_BYTES_NUM - 1);
    memcpy((TS_VOID *)addr, net->u8pParams, weight_length);
    net->u8pParams = (TS_U8 *)addr;
    /* 初始化单个网路 */
    ret = TS_MPI_TRP_RNE_LoadModel(net);
    if (ret) {
        TS_MPI_TRP_RNE_Error("load model error!\n");
        return ret;
    }
    /* net once load* 仅有网络模型配置为once load情况下，内部才真正执行once load */
    ret = TS_MPI_TRP_RNE_OnceLoad(net);
    if (ret) {
        TS_MPI_TRP_RNE_Error("once load error!\n");
        return ret;
    }

    // 获取模型信息
    RNE_BLOBS_S *in_blobs = TS_MPI_TRP_RNE_GetInputBlobs(net);
    RNE_BLOB_S *in_stpBlob = &in_blobs->stpBlob[0];
    int model_h = in_stpBlob->s32H;
    int model_w = in_stpBlob->s32W;


    // 2. 读取图片和前处理
    cv::Mat input_mat = cv::imread("/root/gddeploy/data/pic/helmet2.jpg");
    cv::Mat input_pre;
    preproc_yolov5(input_mat, input_pre, model_h, model_w);
    int img_h = input_mat.rows;
    int img_w = input_mat.cols;
    cv::imwrite("/root/gddeploy/preds/pre.jpg", input_pre);

    // std::ifstream input_bin_file("/root/gddeploy/data/models/helmet/batch_0.bin");
    // input_bin_file.seekg(0, std::ios::end);
    // int bin_length = input_bin_file.tellg();
    // input_bin_file.seekg(0, std::ios::beg);  
    // char* input_bin_buffer = new char[bin_length];
    // input_bin_file.read(input_bin_buffer, bin_length); 

    // 3. 推理
    net->vpInput = (TS_U8 *)input_pre.data;
    RNE_BLOBS_S *out_blobs = TS_MPI_TRP_RNE_Forward(net);
    if (out_blobs == NULL) {
        TS_MPI_TRP_RNE_Error("net forward error!\n");
        return -1;
    }

    // 取出数据
    std::vector<float *> outputs;
    for (TS_U32 idx = 0; idx < out_blobs->u32NBlob; ++idx) {
        RNE_BLOB_S *stpBlob = &out_blobs->stpBlob[idx];
        
        int output_size = stpBlob->s32N * stpBlob->s32H * stpBlob->s32W * stpBlob->s32C; 
        float *out_float_ptr = new float[output_size];

        TS_U8 *out_int_ptr = (TS_U8 *)(out_blobs->stpBlob[idx].vpAddr);
        TS_FLOAT fCoeff = *out_blobs->stpBlob[idx].fCoeff;

        TS_S32 bb = out_blobs->stpBlob[idx].s32N;
        TS_S32 hh = out_blobs->stpBlob[idx].s32H;
        TS_S32 ww = out_blobs->stpBlob[idx].s32W;
        TS_S32 cc = out_blobs->stpBlob[idx].s32C;
        TS_S32 cStride = TS_MPI_TRP_RNE_CStride(cc, out_blobs->stpBlob[idx].s32BitNum, out_blobs->stpBlob[idx].bIsJoined);
        TS_S32 uSize = (out_blobs->stpBlob[idx].s32BitNum / CHAR_BIT);

        for(TS_S32 b = 0; b < bb; ++b) {
            for (TS_S32 h = 0; h < hh; ++h) {
                for (TS_S32 w = 0; w < ww; ++w) {
                    for (TS_S32 c = 0; c < cc; ++c) {
                        // TS_U8 *d = (TS_U8 *)(out + (((j + i * w) * cStride + k) * uSize));
                        TS_S32 *d = (TS_S32 *)(out_int_ptr + (((w + h * ww + b*hh*ww) * cStride + c) * uSize));
                        TS_S32 data = *d & TS_MPI_TRP_RNE_MASK_BITS(out_blobs->stpBlob[idx].s32BitNum);

                        // *(out_float_ptr + ((h * w) * k + i * w + j)) = data * fCoeff;
                        *(out_float_ptr + b*hh*ww*cc+h*ww*cc+w*cc+c) = data * fCoeff;
                        // printf("%d\n", data);
                    }
                }
            }
        }
        

        outputs.emplace_back(out_float_ptr);
    }

    // 4. 后处理
    float anchors[] = {10,  13, 16,  30,  33,  23, 
                116, 90, 156, 198, 373, 326,
                30,  61, 62,  45,  59,  119};
    TS_S32 bb = out_blobs->stpBlob[0].s32N;
    for (size_t b = 0; b < bb; b++) {
        std::vector<DetectObject> detect_objs;

        for (TS_U32 idx = 0; idx < out_blobs->u32NBlob; ++idx) {
            RNE_BLOB_S *stpBlob = &out_blobs->stpBlob[idx];
            
            int output_size = stpBlob->s32N * stpBlob->s32H * stpBlob->s32W * stpBlob->s32C; 
            float *out_float_ptr = new float[output_size];

            TS_U8 *out_int_ptr = (TS_U8 *)(out_blobs->stpBlob[idx].vpAddr);
            TS_FLOAT fCoeff = *out_blobs->stpBlob[idx].fCoeff;

            
            TS_S32 hh = out_blobs->stpBlob[idx].s32H;
            TS_S32 ww = out_blobs->stpBlob[idx].s32W;
            TS_S32 cc = out_blobs->stpBlob[idx].s32C;
            int class_num = cc / 3 - 5;

            int step = hh * ww;
            int ratio = model_w / ww;
            

            float *data_ptr = (float*)outputs[idx];
            
        
            for(int h = 0; h < hh; h++)
            {
                for(int w = 0; w < ww; w++)
                {
                    for (int a_idx = 0; a_idx < 3; a_idx++) {   // for anchor num
                        float *group_addr = data_ptr + b * cc * hh * ww + \
                                h * ww * cc + w * cc + a_idx * cc / 3 ;

                        float conf = *(group_addr + 4);

                        if(conf > 0.1)
                        {
                            DetectObject obj;
                            memset(&obj, 0, sizeof(DetectObject));
                            float bbox_x = *(group_addr + 0);
                            obj.bbox.x = (bbox_x * 2.0f - 0.5f + w) * ratio;
                            float bbox_y = *(group_addr + 1);
                            obj.bbox.y = (bbox_y * 2.0f - 0.5f + h) * ratio;
                            float bbox_w = *(group_addr + 2);
                            float bbox_h = *(group_addr + 3);
                            obj.bbox.w = pow(bbox_w * 2.0f,2) * anchors[(idx)*6 + a_idx*2];
                            obj.bbox.h = pow(bbox_h * 2.0f,2) * anchors[(idx)*6 + a_idx*2+1];

                            obj.bbox.x = obj.bbox.x - obj.bbox.w / 2;
                            obj.bbox.y = obj.bbox.y - obj.bbox.h / 2;

                            for(int cls = 0; cls < class_num; cls++)
                            {
                                float cls_conf = *(group_addr + 5 + cls) * conf;

                                if(cls_conf > obj.score)
                                {
                                    obj.score = cls_conf;
                                    obj.class_id = cls;
                                }
                            }

                            detect_objs.emplace_back(obj);
                        }
                    }
                }
            }
        }
        if (detect_objs.size() > 0){
            detect_objs = nms(detect_objs, 0.5);

            for (auto &obj : detect_objs){
                get_rect(img_w, img_h, model_w, model_h, &obj.bbox);
            }
        }

            //--------------test begin---------------------------------------------
        #if 1
        cv::Mat frame = cv::imread("/root/gddeploy/data/pic/helmet2.jpg");
        std::cout << "---------------------------------------------------" << std::endl;

        std::cout << "Detect num: " << detect_objs.size() << std::endl;
        for (auto &obj : detect_objs) {
            std::cout << "Detect result: " << "box[" << obj.bbox.x \
                << ", " << obj.bbox.y << ", " << obj.bbox.w << ", " \
                << obj.bbox.h << "]" \
                << "   score: " << obj.score 
                << "   class id: " << obj.class_id << std::endl;
            cv::Point p1(obj.bbox.x, obj.bbox.y);
            cv::Point p2(obj.bbox.x+obj.bbox.w, obj.bbox.y+obj.bbox.h);
            cv::rectangle(frame, p1, p2, cv::Scalar(0, 255, 0), 1);
        }

        cv::imwrite("/root/gddeploy/preds/result_img.jpg", frame);
        #endif
        //--------------test end---------------------------------------------

    }

    for (auto out_float_ptr : outputs){
        delete out_float_ptr;
    }

    return 0;
}