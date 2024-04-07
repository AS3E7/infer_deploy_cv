#include "postproc_ocr_rec.h"

#include <math.h>
#include <string>
#include <numeric>

#include "opencv2/imgproc.hpp"
#include "opencv2/opencv.hpp"

using namespace gddeploy;

int OcrRetrievalDecodeOutputNCHW(std::vector<gddeploy::BufSurfWrapperPtr> out_data, 
                                    gddeploy::InferResult &result,  
                                    gddeploy::any param,  
                                    std::vector<gddeploy::FrameInfo> frame_info, 
                                    gddeploy::ModelPtr model_ptr)
{   
    float threshold = any_cast<float>(param);

    auto output_shape = model_ptr->OutputShape(0);

    int out_text_num = output_shape[1];
    int out_char_num = output_shape[2];

    auto input_shape = model_ptr->InputShape(0);
    int model_w = input_shape[2];
    int model_h = input_shape[3];
    int model_b = input_shape[0];

    // get label from model
    auto model_info_priv = model_ptr->GetModelInfoPriv();
    auto labels = model_info_priv->GetLabels();

    for (size_t b = 0; b < out_data.size(); b++) {
        OcrRecImg ocr_rec_img;
        ocr_rec_img.img_id = b;
        ocr_rec_img.img_w = frame_info[b].width;
        ocr_rec_img.img_h = frame_info[b].height;

        float *data_ptr = static_cast<float*>(out_data[0]->GetHostData(0, b));
        int currend_index = 0;

        OcrRecObject ocr_obj_chars;
        ocr_obj_chars.detect_id = b;
        ocr_obj_chars.class_id = 0;
        ocr_obj_chars.score = 0.0f;

        for(int t = 0 ; t < out_text_num; t++){
            float max_prob = 0.0f;
            int max_prob_index = 0;
            
            for(int c = 0 ; c < out_char_num; c++){
                float data = *((float *)data_ptr + b*out_text_num*out_char_num + t*out_char_num + c);
                if (max_prob < data){
                    max_prob = data;
                    max_prob_index = c;
                }
            }
            if (max_prob_index == currend_index){   //连号去重，比如111322变成132
                continue;
            }
            
            currend_index = max_prob_index;

            if (max_prob_index == 0){       // class id为0是空白字符，一般用于隔开字符作用
                continue;                   // 防止连号情况，比如11911，出来会变成1019101
            }

            // printf("index:%d, score:%f\n", currend_index, max_prob);
            if (max_prob_index == labels.size())
                continue;

            OcrChar obj_char;
            obj_char.class_id = max_prob_index;
            obj_char.score = max_prob;
            obj_char.name = labels[max_prob_index];

            ocr_obj_chars.chars.emplace_back(obj_char);
        }
        for (auto &char_str : ocr_obj_chars.chars){
            ocr_obj_chars.chars_str += char_str.name;
        }
        ocr_rec_img.ocr_rec_objs.emplace_back(ocr_obj_chars);
        
        result.ocr_rec_result.batch_size++;
        result.ocr_rec_result.ocr_rec_imgs.emplace_back(ocr_rec_img);
    }
    result.result_type.emplace_back(GDD_RESULT_TYPE_OCR_RETRIEVAL);
    
    return 0;
}

int OcrRetrievalDecodeOutputNHWC(std::vector<gddeploy::BufSurfWrapperPtr> out_data, 
                                    gddeploy::InferResult &result,  
                                    gddeploy::any param,  
                                    std::vector<gddeploy::FrameInfo> frame_info, 
                                    gddeploy::ModelPtr model_ptr)
{   
    OcrRetrievalDecodeOutputNCHW(out_data, result, param, frame_info, model_ptr);

    return 0;
}
