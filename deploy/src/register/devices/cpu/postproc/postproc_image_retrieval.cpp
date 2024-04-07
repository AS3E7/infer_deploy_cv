#include "postproc_image_retrieval.h"

#include <math.h>

#include "opencv2/opencv.hpp"

using namespace gddeploy;

//
int ImageRetrievalDecodeOutputNCHW(std::vector<gddeploy::BufSurfWrapperPtr> out_data, 
                                    gddeploy::InferResult &result,  
                                    gddeploy::any param,  
                                    std::vector<gddeploy::FrameInfo> frame_info, 
                                    gddeploy::ModelPtr model_ptr)
{   
    float threshold = any_cast<float>(param);

    auto output_shape = model_ptr->OutputShape(0);
    int feature_len     = output_shape[1];
    int batch_size  = output_shape[0];

    for (size_t b = 0; b < out_data.size(); b++) {
        float *data_ptr = static_cast<float*>(out_data[0]->GetHostData(0, b));

        cv::Mat L2_norm(1, feature_len, CV_32FC1, data_ptr);
        cv::normalize(L2_norm, L2_norm, 1, 0, cv::NORM_L2);

        ImageRetrieval image_retrieval_img;
        image_retrieval_img.img_id = 0;
        image_retrieval_img.feature.resize(feature_len);
        memcpy(image_retrieval_img.feature.data(), L2_norm.data, feature_len * sizeof(float));

        result.result_type.emplace_back(GDD_RESULT_TYPE_IMAGE_RETRIEVAL);
        result.image_retrieval_result.batch_size++;
        result.image_retrieval_result.image_retrieval_imgs.emplace_back(image_retrieval_img);
    }
    
    return 0;
}

int ImageRetrievalDecodeOutputNHWC(std::vector<gddeploy::BufSurfWrapperPtr> out_data, 
                                    gddeploy::InferResult &result,  
                                    gddeploy::any param,  
                                    std::vector<gddeploy::FrameInfo> frame_info, 
                                    gddeploy::ModelPtr model_ptr)
{   
    float threshold = any_cast<float>(param);

    auto output_shape = model_ptr->OutputShape(0);
    int feature_len     = output_shape[1];
    int batch_size  = output_shape[0];

    for (size_t b = 0; b < out_data.size(); b++) {
        float *data_ptr = static_cast<float*>(out_data[0]->GetHostData(0, b));

        cv::Mat L2_norm(1, feature_len, CV_32FC1, data_ptr);
        cv::normalize(L2_norm, L2_norm, 1, 0, cv::NORM_L2);

        ImageRetrieval image_retrieval_img;
        image_retrieval_img.img_id = 0;
        image_retrieval_img.feature.resize(feature_len);
        memcpy(image_retrieval_img.feature.data(), L2_norm.data, feature_len * sizeof(float));

        result.result_type.emplace_back(GDD_RESULT_TYPE_IMAGE_RETRIEVAL);
        result.image_retrieval_result.batch_size++;
        result.image_retrieval_result.image_retrieval_imgs.emplace_back(image_retrieval_img);
    }
    
    return 0;

    return 0;
}
