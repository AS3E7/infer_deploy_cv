#include "postproc_face_retrieval.h"

#include <math.h>

#include "opencv2/opencv.hpp"

using namespace gddeploy;

//
int FaceRetrievalDecodeOutputNCHW(std::vector<gddeploy::BufSurfWrapperPtr> out_data, 
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

        FaceRetrieval face_retrieval_img;
        face_retrieval_img.img_id = 0;
        face_retrieval_img.feature.resize(feature_len);
        memcpy(face_retrieval_img.feature.data(), L2_norm.data, feature_len * sizeof(float));

        result.result_type.emplace_back(GDD_RESULT_TYPE_FACE_RETRIEVAL);
        result.face_retrieval_result.batch_size++;
        result.face_retrieval_result.face_retrieval_imgs.emplace_back(face_retrieval_img);
    }
    
    return 0;
}

int FaceRetrievalDecodeOutputNHWC(std::vector<gddeploy::BufSurfWrapperPtr> out_data, 
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

        FaceRetrieval face_retrieval_img;
        face_retrieval_img.img_id = 0;
        face_retrieval_img.feature.resize(feature_len);
        memcpy(face_retrieval_img.feature.data(), L2_norm.data, feature_len * sizeof(float));

        result.result_type.emplace_back(GDD_RESULT_TYPE_FACE_RETRIEVAL);
        result.face_retrieval_result.batch_size++;
        result.face_retrieval_result.face_retrieval_imgs.emplace_back(face_retrieval_img);
    }

    return 0;
}
