#include "app/runner_dataset.h"
#include <memory>
#include <fstream>
#include <sys/stat.h>

#include "api/global_config.h"
#include "api/infer_api.h"
#include "core/mem/buf_surface_util.h"
#include "core/result_def.h"
#include "common/type_convert.h"
#include "result_handle.h"
#include "common/json.hpp"
using namespace nlohmann;

#include "opencv2/opencv.hpp"

namespace gddeploy {

class DatasetRunnerPriv;
class DatasetRunnerPriv{
public:
    DatasetRunnerPriv()=default;

    int Init(const std::string config, std::string model_path, std::string license_paths = "");
    int Init(const std::string config, std::vector<std::string> model_paths, std::vector<std::string> license_paths);

    int InferSync(std::string anno_path, std::string pic_path, std::string result_file, std::string save_path="", bool is_draw=false);  
private:
    int DatasetClassify(std::string anno_path, std::string pic_path, std::string result_file, std::string save_path="", bool is_draw=false);
    int DatasetDetect(std::string anno_path, std::string pic_path, std::string result_file, std::string save_path="", bool is_draw=false);
    int DatasetPose(std::string anno_path, std::string pic_path, std::string result_file, std::string save_path="", bool is_draw=false);
    int DatasetSeg(std::string anno_path, std::string pic_path, std::string result_file, std::string save_path="", bool is_draw=false);
    int DatasetImageRetrieval(std::string anno_path, std::string pic_path, std::string result_file, std::string save_path, bool is_draw);
    int DatasetFaceRetrieval(std::string anno_path, std::string pic_path, std::string result_file, std::string save_path, bool is_draw);
    int DatasetOcrRetrieval(std::string anno_path, std::string pic_path, std::string result_file, std::string save_path, bool is_draw);
    int DatasetAction(std::string anno_path, std::string pic_path, std::string result_file, std::string save_path, bool is_draw);

    int InferSync(std::string pic_path, gddeploy::PackagePtr& out);

    InferAPI infer_api_;
    std::vector<InferAPI> infer_api_v_;
};

} // namespace gddeploy

using namespace gddeploy;

int DatasetRunnerPriv::InferSync(std::string pic_path, gddeploy::PackagePtr &out)
{                  
    cv::Mat in_mat = cv::imread(pic_path);
    if (in_mat.empty()) {
        std::cout << "read image failed: " << pic_path << std::endl;
        return -1;
    }
    
    // 2. 推理
    gddeploy::BufSurfWrapperPtr surf;
    
#if WITH_BM1684
    bm_image img;
    cv::bmcv::toBMI(in_mat, &img, true);

    convertBmImage2BufSurface(img, surf, false);
#else
    convertMat2BufSurface(in_mat, surf, true);
#endif

    gddeploy::PackagePtr in = gddeploy::Package::Create(1);

    in->data[0]->Set(surf);

    infer_api_v_[0].InferSync(in, out);

    return 0;
}

int DatasetRunnerPriv::DatasetClassify(std::string anno_path, std::string pic_path, std::string result_file, std::string save_path, bool is_draw)
{
    //读取json文件，并且解析反序列化
    std::ifstream ifs(anno_path);
    json j ;
    ifs >> j;
    auto images = j["images"];
    int img_num = images.size();

    std::string result_s;
    auto result_array = json::array();
    std::ofstream out(result_file);

    int index = 0;

    for (int i = 0; i < img_num; i++) {
 
        auto item_img = images[i];
        int img_id = item_img["id"];
        std::string file_name = item_img["file_name"];
        std::string file_path = pic_path+file_name;
        std::string file_save_path = save_path+file_name;

        gddeploy::PackagePtr out = gddeploy::Package::Create(1);
        InferSync(file_path, out);

        for (uint32_t i = 0; i < out->data.size(); i++){
            const gddeploy::InferResult& postproc_results = out->data[0]->GetMetaData<gddeploy::InferResult>();  
            
            for (auto result_type : postproc_results.result_type){
                if (result_type == GDD_RESULT_TYPE_DETECT){

                    for (auto &obj : postproc_results.classify_result.detect_imgs[0].detect_objs) {
                                    std::cout << "Classify result: "
                                        << "   score: " << obj.score 
                                        << "   class id: " << obj.class_id << std::endl;
                                
                        json result;
                        result["image_id"] = img_id;
                        result["category_id"] = obj.class_id;
                        result["score"] = obj.score;

                        result_array[index] = result;   
                        index++;
                    }
                }
            }
        }
    }

    result_s = result_array.dump();
    out << result_s;

    return 0;
}


int DatasetRunnerPriv::DatasetDetect(std::string anno_path, std::string pic_path, std::string result_file, std::string save_path, bool is_draw)
{
    //读取json文件，并且解析反序列化
    std::ifstream ifs(anno_path);
    json j ;
    ifs >> j;
    auto images = j["images"];
    int img_num = images.size();

    std::string result_s;
    auto result_array = json::array();
    std::ofstream out(result_file);

    int index = 0;

    for (int i = 0; i < img_num; i++) {
 
        auto item_img = images[i];
        int img_id = item_img["id"];
        std::string file_name = item_img["file_name"];
        std::string file_path = pic_path+file_name;
        std::string file_save_path = save_path+file_name;
        printf("file path:%s\n", file_path.c_str());

        gddeploy::PackagePtr out = gddeploy::Package::Create(1);
        if (-1 == InferSync(file_path, out)){
            continue;
        }

        for (uint32_t i = 0; i < out->data.size(); i++){
            if (out->data[0]->HasMetaValue() == false)
                continue;
            const gddeploy::InferResult& postproc_results = out->data[0]->GetMetaData<gddeploy::InferResult>();  
            
            for (auto result_type : postproc_results.result_type){
                if (result_type == GDD_RESULT_TYPE_DETECT){

                    for (auto &obj : postproc_results.detect_result.detect_imgs[0].detect_objs) {

                        std::cout << "detect result: " << "box[" << obj.bbox.x \
                                << ", " << obj.bbox.y << ", " << obj.bbox.w << ", " \
                                << obj.bbox.h << "]" \
                                << "   score: " << obj.score << std::endl;

                        json result;
                        result["image_id"] = img_id;
                        result["image_name"] = file_name;
                        result["category_id"] = obj.class_id;
                        result["bbox"] = {obj.bbox.x, obj.bbox.y, obj.bbox.w, obj.bbox.h};
                        result["score"] = obj.score;

                        result_array[index] = result;   
                        index++;
                    }
                }
            }
        }
    }

    result_s = result_array.dump();
    out << result_s;

    return 0;
}


int DatasetRunnerPriv::DatasetPose(std::string anno_path, std::string pic_path, std::string result_file, std::string save_path, bool is_draw)
{
    //读取json文件，并且解析反序列化
    std::ifstream ifs(anno_path);
    json j ;
    ifs >> j;
    auto images = j["images"];
    int img_num = images.size();

    std::string result_s;
    auto result_array = json::array();
    std::ofstream out(result_file);

    int index = 0;

    for (int i = 0; i < img_num; i++) {
 
        auto item_img = images[i];
        int img_id = item_img["id"];
        std::string file_name = item_img["file_name"];
        std::string file_path = pic_path+file_name;
        std::string file_save_path = save_path+file_name;

        gddeploy::PackagePtr out = gddeploy::Package::Create(1);
        InferSync(file_path, out);

        for (uint32_t i = 0; i < out->data.size(); i++){
            const gddeploy::InferResult& postproc_results = out->data[0]->GetMetaData<gddeploy::InferResult>();  
            
            for (auto result_type : postproc_results.result_type){
                if (result_type == GDD_RESULT_TYPE_DETECT_POSE){

                    for (auto &obj : postproc_results.detect_pose_result.detect_imgs[0].detect_objs) {
                        std::cout << "detect result: " << "box[" << obj.bbox.x \
                                << ", " << obj.bbox.y << ", " << obj.bbox.w << ", " \
                                << obj.bbox.h << "]" \
                                << "   score: " << obj.score << std::endl;

                        json result;
                        result["image_id"] = img_id;
                        result["image_name"] = file_name;
                        result["category_id"] = 1;
                        result["bbox"] = {(int)obj.bbox.x, (int)obj.bbox.y, (int)obj.bbox.w, (int)obj.bbox.h};
                        result["score"] = obj.score;

                        int keypoint_num = obj.point.size();
                        std::vector<float> keypoints_data;
                        keypoints_data.resize(keypoint_num*3);
                        for (int i = 0; i < keypoint_num; i++){
                            keypoints_data[i*3+0] = (int)obj.point[i].x;
                            keypoints_data[i*3+1] = (int)obj.point[i].y;
                            keypoints_data[i*3+2] = 2;
                            
                        }
                        result["keypoints"] = keypoints_data;

                        result_array[index] = result;   
                        index++;
                    }
                }
            }
        }
    }

    result_s = result_array.dump();
    out << result_s;

    return 0;
}

int DatasetRunnerPriv::DatasetSeg(std::string anno_path, std::string pic_path, std::string result_file, std::string save_path, bool is_draw)
{
    //读取json文件，并且解析反序列化
    std::ifstream ifs(anno_path);
    json j ;
    ifs >> j;
    int img_num = j.size();
    
    for (int i = 0; i < img_num; i++) {
 
        auto item_img = j[i];        
        std::string file_name = item_img["filename"];
        std::string file_path = pic_path+file_name;
        std::string file_save_path = save_path+file_name;

        gddeploy::PackagePtr out = gddeploy::Package::Create(1);
        InferSync(file_path, out);

        const gddeploy::InferResult& postproc_results = out->data[0]->GetMetaData<gddeploy::InferResult>();
        const std::vector<SegImg>& seg_imgs = postproc_results.seg_result.seg_imgs;

        // 这里只需要把结果保存为.npy文件即可
        std::string filenamepure = file_name.substr(0, file_name.rfind("."));
        std::string pic_name = result_file + filenamepure + ".bin";

        std::ofstream ofile(pic_name);
        if(ofile.is_open()==false){
            std::cout << strerror(errno) << std::endl;
            continue;
        }
        ofile.write((char*)seg_imgs[0].seg_map.data(), (seg_imgs[0].map_h * seg_imgs[0].map_w)*sizeof(uint8_t));
        ofile.close();
    }
    return 0;
}

static void split(const std::string & s, std::vector<std::string>& tokens, const std::string & delimiters = " ")
{
    std::string::size_type lastPos = s.find_first_not_of(delimiters, 0);
    std::string::size_type pos = s.find_first_of(delimiters, lastPos);
    while (std::string::npos != pos || std::string::npos != lastPos){
        tokens.push_back(s.substr(lastPos, pos - lastPos));
        lastPos = s.find_first_not_of(delimiters, pos);
        pos = s.find_first_of(delimiters, lastPos);
    }
}

static std::string GetFilename(std::string path)
{
    std::string::size_type iPos = path.find_last_of('/') + 1;
    std::string filename = path.substr(iPos, path.length() - 1);
    return filename;
}

static std::string GetPureFilename(const std::string & s)
{
    std::vector<std::string> tokens;
    split(s, tokens, ".");
    return tokens[0];
}

static std::string GetImgId(const std::string & s)
{
    std::vector<std::string> tokens;
    split(s, tokens, "_");
    return tokens[1];
}

int DatasetRunnerPriv::DatasetImageRetrieval(std::string anno_path, std::string pic_path, std::string result_file, std::string save_path, bool is_draw)
{
    //读取json文件，并且解析反序列化
    std::ifstream ifs(anno_path);
    json j ;
    ifs >> j;
    int img_num = j.size();

    // 把所有的特征值保存到dataset中，行是图片数量，底库数量，列是特征值长度512
    cv::Mat mat_database = cv::Mat(img_num, 512, CV_32FC1);
    std::vector<std::string> category_id_list;
    int index = 0;
    for (int i = 0; i < img_num; i++) {
        auto item_img = j[i];
        std::string category_id = item_img["category_id"];
        // std::string embedding = item_img["embedding"];

        category_id_list.push_back(category_id);

        auto embedding = item_img["embedding"].get<std::vector<float>>();
        memcpy(mat_database.data + index * 512 * sizeof(float), embedding.data(), 512 * sizeof(float));
        index++;
    }
    
    index = 0;
    int bingo_num = 0;
    std::vector<std::string> imgs_path;
    struct stat s_buf;
    stat(pic_path.c_str(), &s_buf);
    if (S_ISDIR(s_buf.st_mode)){
        struct dirent * ptr;
        DIR *dir = opendir((char *)pic_path.c_str()); //打开一个目录
        while((ptr = readdir(dir)) != NULL) //循环读取目录数据
        {
            if (std::string(ptr->d_name) == "." || std::string(ptr->d_name) == "..")
                continue;
            
            if (-1 == std::string(ptr->d_name).find(".jpg")
                && -1 == std::string(ptr->d_name).find(".png")) {   //目录
                continue;
            }
            // TODO: 判断后缀是否为可支持的图片格式，排除非图片文件
            imgs_path.push_back(pic_path+std::string(ptr->d_name));
        }
    }else{
        imgs_path.push_back(pic_path);
    }

    // 循环异步推理图片数据
    for (auto img_path : imgs_path){
        std::string img_save_path = save_path + GetFilename(img_path);
 
        // auto item_img = j[i];        
        // std::string img_id = GetPureFilename(item_img["image_id"]);
        std::string file_name = GetFilename(img_path);
        auto category_id_tmp = GetPureFilename(file_name);
        auto category_id = GetImgId(category_id_tmp);
        
        std::string file_path = img_path;
        std::string file_save_path = save_path+file_name;

        gddeploy::PackagePtr out = gddeploy::Package::Create(1);
        InferSync(file_path, out);

        const gddeploy::InferResult& postproc_results = out->data[0]->GetMetaData<gddeploy::InferResult>();
        const std::vector<ImageRetrieval>& image_retrieval_imgs = postproc_results.image_retrieval_result.image_retrieval_imgs;

        for (auto & image_retrieval_img : image_retrieval_imgs){
            // image_retrieval_img.img_id = img_id;

            cv::Mat output(1, image_retrieval_img.feature.size(), CV_32FC1, (float *)image_retrieval_img.feature.data());

            cv::Mat res = mat_database * output.t();

            float score = -FLT_MAX;
            int category_index = -1;
            for (int i = 0; i < res.rows; i++)
            {
                if (res.at<float>(i, 0) > score)
                {
                    score = res.at<float>(i, 0);
                    category_index = i;
                }
            }
            std::string result_category = category_id_list[category_index];
            std::cout << "file_name: " << file_name << ", label category: " << category_id << ", category: " << result_category << std::endl;
            if (category_id == result_category){
                bingo_num++;
            }

            index++;
        }
    }

    std::cout << "Finaly score:" << (float)bingo_num / imgs_path.size() << std::endl;

    return 0;
}


int DatasetRunnerPriv::DatasetFaceRetrieval(std::string anno_path, std::string pic_path, std::string result_file, std::string save_path, bool is_draw)
{
    //读取json文件，并且解析反序列化
    // std::ifstream ifs(anno_path);
    // json j ;
    // ifs >> j;
    // int img_num = j.size();

    // // 把所有的特征值保存到dataset中，行是图片数量，底库数量，列是特征值长度512
    // cv::Mat mat_database = cv::Mat(img_num, 512, CV_32FC1);
    // std::vector<std::string> category_id_list;
    // int index = 0;
    // for (int i = 0; i < img_num; i++) {
    //     auto item_img = j[i];
    //     std::string category_id = item_img["category_id"];
    //     // std::string embedding = item_img["embedding"];

    //     category_id_list.push_back(category_id);

    //     auto embedding = item_img["embedding"].get<std::vector<float>>();
    //     memcpy(mat_database.data + index * 512 * sizeof(float), embedding.data(), 512 * sizeof(float));
    //     index++;
    // }
    
    int index = 0;
    int bingo_num = 0;
    std::vector<std::string> imgs_path;
    struct stat s_buf;
    stat(pic_path.c_str(), &s_buf);
    if (S_ISDIR(s_buf.st_mode)){
        struct dirent * ptr;
        DIR *dir = opendir((char *)pic_path.c_str()); //打开一个目录
        while((ptr = readdir(dir)) != NULL) //循环读取目录数据
        {
            if (std::string(ptr->d_name) == "." || std::string(ptr->d_name) == "..")
                continue;
            
            if (-1 == std::string(ptr->d_name).find(".jpg")
                && -1 == std::string(ptr->d_name).find(".png")) {   //目录
                continue;
            }
            // TODO: 判断后缀是否为可支持的图片格式，排除非图片文件
            imgs_path.push_back(pic_path+std::string(ptr->d_name));
        }
    }else{
        imgs_path.push_back(pic_path);
    }

    auto result_array = json::array();

    // 循环异步推理图片数据
    for (auto img_path : imgs_path){
        std::string img_save_path = save_path + GetFilename(img_path);
 
        // auto item_img = j[i];        
        // std::string img_id = GetPureFilename(item_img["image_id"]);
        std::string file_name = GetFilename(img_path);
        // auto category_id_tmp = GetPureFilename(file_name);
        // auto category_id = GetImgId(category_id_tmp);
        
        std::string file_path = img_path;
        std::string file_save_path = save_path+file_name;

        gddeploy::PackagePtr out = gddeploy::Package::Create(1);
        cv::Mat in_mat = cv::imread(file_path);
        
        // 2. 推理
        gddeploy::BufSurfWrapperPtr surf;
        
    #if WITH_BM1684
        bm_image img;
        cv::bmcv::toBMI(in_mat, &img, true);
        convertBmImage2BufSurface(img, surf, false);
    #else
        convertMat2BufSurface(in_mat, surf, true);
    #endif

        gddeploy::PackagePtr in = gddeploy::Package::Create(1);

        in->data[0]->Set(surf);

        // InferSync(file_path, out);
        if (infer_api_v_.size() > 0){
            for (auto &infer_api : infer_api_v_){
                infer_api.InferSync(in, out);
                if (false == out->data[0]->HasMetaValue())
                    break;
                
                gddeploy::InferResult result = out->data[0]->GetMetaData<gddeploy::InferResult>();

                PrintResult(result);
                if (result.detect_pose_result.detect_imgs.size() == 0){
                    break;
                }
                if (result.detect_pose_result.detect_imgs[0].detect_objs.size() == 0){
                    break;
                }
                    
            }
        } else {
            infer_api_.InferSync(in, out);
        }

        const gddeploy::InferResult& postproc_results = out->data[0]->GetMetaData<gddeploy::InferResult>();
        const std::vector<FaceRetrieval>& face_retrieval_imgs = postproc_results.face_retrieval_result.face_retrieval_imgs;

        for (auto & face_retrieval_img : face_retrieval_imgs){
            // face_retrieval_img.img_id = img_id;
            float feature[512] = {0};
            memcpy(feature, face_retrieval_img.feature.data(), 512 * sizeof(float));
            json result;
            result["image_name"] = file_name;
            result["embedding"] = feature;

            result_array.push_back(result);

            // cv::Mat output(1, face_retrieval_img.feature.size(), CV_32FC1, (float *)face_retrieval_img.feature.data());

            // cv::Mat res = mat_database * output.t();

            // float score = -FLT_MAX;
            // int category_index = -1;
            // for (int i = 0; i < res.rows; i++)
            // {
            //     if (res.at<float>(i, 0) > score)
            //     {
            //         score = res.at<float>(i, 0);
            //         category_index = i;
            //     }
            // }
            // std::string result_category = category_id_list[category_index];
            // std::cout << "file_name: " << file_name << ", label category: " << category_id << ", category: " << result_category << std::endl;
            // if (category_id == result_category){
            //     bingo_num++;
            // }

            index++;
        }
    }

    std::string result_s = result_array.dump();
    std::ofstream out(result_file);
    out << result_s;

    std::cout << "Finaly score:" << (float)bingo_num / imgs_path.size() << std::endl;

    return 0;
}

std::string GetPureCarLicense(std::string license)
{
    std::string pure_car_license = license;
    pure_car_license.erase(std::remove(pure_car_license.begin(), pure_car_license.end(), '.'), pure_car_license.end());
    pure_car_license.erase(std::remove(pure_car_license.begin(), pure_car_license.end(), '-'), pure_car_license.end());

    std::string chars = "·";
    pure_car_license.erase(remove_if(pure_car_license.begin(), pure_car_license.end(),
                        [&chars](const char &c) {
                            return chars.find(c) != std::string::npos;
                        }),
                        pure_car_license.end());
    
    return pure_car_license;
}

int DatasetRunnerPriv::DatasetOcrRetrieval(std::string anno_path, std::string pic_path, std::string result_file, std::string save_path, bool is_draw)
{
    std::ifstream input_file(anno_path);
    if (!input_file.is_open()) {
        std::cerr << "Could not open the file - '" << anno_path << "'" << std::endl;
        return EXIT_FAILURE;
    }

    std::string line;
    std::map<std::string, std::string> labels;
    while (getline(input_file, line)){
        std::vector<std::string> tokens;
        std::istringstream ss(line);
        std::string word;
        while(ss >> word) {
            tokens.push_back(word);
        }

        std::string pure_file_name = GetFilename(tokens[0]);
        labels[pure_file_name] = tokens[1];
    }

    int labels_num = labels.size();
    int bingo_num = 0;

    for (auto &iter : labels){
        std::string pure_file_name = iter.first;
        std::string label = iter.second;

        std::string file_path = pic_path + pure_file_name;

        gddeploy::PackagePtr out = gddeploy::Package::Create(1);
        InferSync(file_path, out);

        for (uint32_t i = 0; i < out->data.size(); i++){
            const gddeploy::InferResult& postproc_results = out->data[0]->GetMetaData<gddeploy::InferResult>();  
            
            for (auto result_type : postproc_results.result_type){
                if (result_type == GDD_RESULT_TYPE_OCR_RETRIEVAL){
                    for (auto &obj : postproc_results.ocr_rec_result.ocr_rec_imgs[0].ocr_rec_objs) {
                        std::string pure_car_license = GetPureCarLicense(obj.chars_str);
                        std::cout << "label: " << label << ", result: " << pure_car_license << std::endl;
                        if (pure_car_license == label){
                            bingo_num++;
                        }
                    }
                }
            }
        }
    }

    std::cout << "Finaly score:" << (float)bingo_num / labels_num << std::endl;

    return 0;
}

int DatasetRunnerPriv::DatasetAction(std::string anno_path, std::string pic_path, std::string result_file, std::string save_path, bool is_draw)
{
    std::ifstream ifs(anno_path);

    json result_arrays ;
    ifs >> result_arrays;
    json result_array = result_arrays[0];

    int right_num = 0;
    int dataset_num = 0;
    int img_id = 0;
    for (auto result : result_array){
        img_id++;
        auto keypoints = result["keypoint"];
        int keypoint_num = keypoints.size()/(17*2);

        auto category_id = 0;
        auto keypoint_score = result["keypoint_score"];
        
        auto label = result["label"];
        auto img_w_h = result["img_shape"];

        int keypoint_num_min = std::max(21, keypoint_num);


        std::vector<DetectPoseObject> objs;   

        bool isRight = false;
        for (int k = 0; k < keypoint_num_min; k++){
            int k_index = k; 
            if (keypoint_num < 21 && k >= keypoint_num){
                k_index = (k - keypoint_num) % keypoint_num;
            }
            //-----------------------------------------------------------------------------//
            
            DetectPoseObject obj;
            obj.class_id = category_id;
            obj.score = 0.9;
            
            obj.point.resize(17);
            for (int k = 0; k < 17; k++){
                obj.point[k].x = (int)keypoints[k_index*17*2+k*2+0];
                obj.point[k].y = (int)keypoints[k_index*17*2+k*2+1];
                obj.point[k].score = 1;
                obj.point[k].number = k;
            }

            auto min_bbox_x_iter = std::min_element(obj.point.begin(), obj.point.end(), [](PoseKeyPoint a, PoseKeyPoint b){
                return (a.x < b.x);
            });
            obj.bbox.x = (*min_bbox_x_iter).x;
            
            auto min_bbox_y_iter = std::min_element(obj.point.begin(), obj.point.end(), [](PoseKeyPoint a, PoseKeyPoint b){
                return (a.y < b.y);
            });
            obj.bbox.y = (*min_bbox_y_iter).y;

            auto max_bbox_x_iter = std::max_element(obj.point.begin(), obj.point.end(), [](PoseKeyPoint a, PoseKeyPoint b){
                return (a.x < b.x);
            });
            obj.bbox.w = (*max_bbox_x_iter).x - obj.bbox.x;

            auto max_bbox_y_iter = std::max_element(obj.point.begin(), obj.point.end(), [](PoseKeyPoint a, PoseKeyPoint b){
                return (a.y < b.y);
            });
            obj.bbox.h = (*max_bbox_y_iter).y - obj.bbox.y;

            objs.emplace_back(obj);
            
            if (objs.size() < 20){
                continue;
            }

            DetectPoseImg detect_img;
            detect_img.img_id = img_id;
            detect_img.img_w = img_w_h[0];
            detect_img.img_h = img_w_h[1];
            detect_img.detect_objs = objs;

            InferResult infer_result;
            infer_result.result_type.emplace_back(GDD_RESULT_TYPE_DETECT_POSE);
            infer_result.detect_pose_result.detect_imgs.emplace_back(detect_img);
            gddeploy::PackagePtr in = gddeploy::Package::Create(1);
            in->data[0]->Set(std::move(infer_result));
            in->tag = "no img";

            gddeploy::PackagePtr out = gddeploy::Package::Create(1);

            infer_api_.InferSync(in, out);

            if (out->data.size() == 0)
                continue;

            const gddeploy::InferResult& postproc_results = out->data[0]->GetMetaData<gddeploy::InferResult>();
            for (auto &obj : postproc_results.classify_result.detect_imgs[0].detect_objs) {

                std::cout << "Result img id: " << img_id << ", class id:" << obj.class_id << ", score:" << obj.score << ", label: " <<  int(label) << std::endl;

                if (int(label) == obj.class_id){
                    std::cout << "bingo!!!!!!!!!!!!!!!!!!" << std::endl;
                    right_num++;
                    isRight = true;
                    break;
                }
            }

            if (isRight)
                break;

            if (objs.size() >= 20){
                objs.erase(objs.begin(), objs.begin()+1);
                // std::cout << "obj size: " << objs.size() << std::endl;

            }
        }
        //-----------------------------------------------------------------------------//
        dataset_num++;
    }
    float final_score = (float)right_num / dataset_num;
    std::cout << "right_num :" << right_num << ", final score: " << final_score << std::endl;

    return 0;
}

int DatasetRunnerPriv::Init(const std::string config, std::string model_path, std::string license_path)
{
    gddeploy_init("");
    
    infer_api_.Init(config, model_path, license_path);
    return 0;
}

int DatasetRunnerPriv::Init(const std::string config, std::vector<std::string> model_paths, std::vector<std::string> license_paths)
{
    gddeploy_init("");

    for (int i = 0; i < model_paths.size(); i++){
        std::string model_path = model_paths[i];
        std::string license_path = license_paths[i];
        
        InferAPI infer_api;
        infer_api.Init(config, model_path, license_path);

        infer_api_v_.push_back(infer_api);
    }
    return 0;
}

int DatasetRunnerPriv::InferSync(std::string anno_path, std::string pic_path, std::string result_file, std::string save_path, bool is_draw)
{
    std::string model_type;
     if (infer_api_v_.size() > 0){
        for (auto &infer_api : infer_api_v_){
            model_type = infer_api.GetModelType();
        }
    } else {
        model_type = infer_api_.GetModelType();
    }
    if (model_type == "classification"){
        DatasetClassify(anno_path, pic_path, result_file, save_path, is_draw);
    } else if (model_type == "detection"){
        DatasetDetect(anno_path, pic_path, result_file, save_path, is_draw);
    } else if (model_type == "pose"){
        DatasetPose(anno_path, pic_path, result_file, save_path, is_draw);
    } else if (model_type == "segmentation"){
        DatasetSeg(anno_path, pic_path, result_file, save_path, is_draw);
    } else if (model_type == "action"){
        DatasetAction(anno_path, pic_path, result_file, save_path, is_draw);
    } else if (model_type == "image-retrieval"){
    //     DatasetImageRetrieval(anno_path, pic_path, result_file, save_path, is_draw);
    // } else if (model_type == "image-retrieval"){
        DatasetFaceRetrieval(anno_path, pic_path, result_file, save_path, is_draw);
    } else if (model_type == "ocr"){
        DatasetOcrRetrieval(anno_path, pic_path, result_file, save_path, is_draw);
    }

    return 0;
}

DatasetRunner::DatasetRunner()
{
    priv_ = std::make_shared<DatasetRunnerPriv>();
}

int DatasetRunner::Init(const std::string config, std::string model_path, std::string license_path)
{
    priv_->Init(config, model_path, license_path);

    return 0;
}

int DatasetRunner::Init(const std::string config, std::vector<std::string> model_paths, std::vector<std::string> license_paths)
{
    priv_->Init(config, model_paths, license_paths);

    return 0;
}

int DatasetRunner::InferSync(std::string anno_path, std::string pic_path, std::string result_file, std::string save_path, bool is_draw)
{
    priv_->InferSync(anno_path, pic_path, result_file, save_path, is_draw);

    return 0;
}