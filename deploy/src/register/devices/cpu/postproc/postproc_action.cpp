#include "postproc_classify.h"

#include <math.h>

using namespace gddeploy;

//
int ActionDecodeOutputNCHW(std::vector<gddeploy::BufSurfWrapperPtr> out_data, 
                                    gddeploy::InferResult &result,  
                                    gddeploy::any param,  
                                    std::vector<gddeploy::FrameInfo> frame_info, 
                                    gddeploy::ModelPtr model_ptr)
{   
    float threshold = any_cast<float>(param);

    auto output_shape = model_ptr->OutputShape(0);
    int num_cls     = output_shape[1];
    int batch_size  = output_shape[0];

    for (size_t b = 0; b < batch_size; b++) {
        float *data_ptr = static_cast<float*>(out_data[0]->GetHostData(0, b));

        float max_prob = 0;
        int max_prob_class = -1;

        float exp_sum = 0.0f;
        for (int n = 0; n < num_cls; n++)
        {
            exp_sum += expf(data_ptr[n]);
        }
        
        for(int i = 0 ; i < num_cls; i++)
        {
            data_ptr[i] = expf(data_ptr[i]);
            if(data_ptr[i] > max_prob) 
            {
                max_prob = data_ptr[i];
                max_prob_class = i;
            }
        }
        max_prob /= exp_sum;

        ClassifyObject class_obj;
        class_obj.detect_id = 0;
        class_obj.class_id = max_prob_class;
        class_obj.score = max_prob;

        ClassifyImg class_img;
        class_img.img_id = 0;
        class_img.detect_objs.emplace_back(class_obj);
        result.result_type.emplace_back(GDD_RESULT_TYPE_ACTION);
        result.action_result.batch_size++;
        result.action_result.detect_imgs.emplace_back(class_img);
    }
    
    return 0;
}

int ActionDecodeOutputNHWC(std::vector<gddeploy::BufSurfWrapperPtr> out_data, 
                                    gddeploy::InferResult &result,  
                                    gddeploy::any param,  
                                    std::vector<gddeploy::FrameInfo> frame_info, 
                                    gddeploy::ModelPtr model_ptr)
{   
    float threshold = any_cast<float>(param);

    auto output_shape = model_ptr->OutputShape(0);
    int num_cls     = output_shape[1];
    int batch_size  = output_shape[0];

    for (size_t b = 0; b < out_data.size(); b++) {
        float *data_ptr = static_cast<float*>(out_data[0]->GetHostData(0, b));

        float max_prob = 0;
        int max_prob_class = -1;

        float exp_sum = 0.0f;
        for (int n = 0; n < num_cls; n++)
        {
            exp_sum += expf(data_ptr[n]);
        }
        
        for(int i = 0 ; i < num_cls; i++)
        {
            data_ptr[i] = expf(data_ptr[i]);
            if(data_ptr[i] > max_prob) 
            {
                max_prob = data_ptr[i];
                max_prob_class = i;
            }
        }
        max_prob /= exp_sum;

        ClassifyObject class_obj;
        class_obj.detect_id = 0;
        class_obj.class_id = max_prob_class;
        class_obj.score = max_prob;

        ClassifyImg class_img;
        class_img.img_id = 0;
        class_img.detect_objs.emplace_back(class_obj);
        result.classify_result.detect_imgs.emplace_back(class_img);
    }

    return 0;
}
