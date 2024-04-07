#include "hikvison_isapi_v2.h"
#include "spdlog/spdlog.h"
#include <json.hpp>
#include <opencv2/imgcodecs.hpp>
#include <string>

namespace gddi {
namespace nodes {

HikvisonISAPI_v2::~HikvisonISAPI_v2() {}

static size_t write_callback(void *contents, size_t size, size_t nmemb, std::string *response) {
    size_t totalSize = size * nmemb;
    response->append(static_cast<char *>(contents), totalSize);
    return totalSize;
}

bool send_request(CURL *curl, const std::string &camera_ip, const int preset_id, const int staing_time, std::string &response) {
    response.clear();
    curl_easy_setopt(curl, CURLOPT_CUSTOMREQUEST, "PUT");
    curl_easy_setopt(
        curl, CURLOPT_URL,
        ("http://" + camera_ip + "/ISAPI/PTZCtrl/channels/1/presets/" + std::to_string(preset_id) + "/goto").c_str());
    auto res = curl_easy_perform(curl);
    if (res == CURLE_OK) {
        long response_code;
        curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &response_code);
        if (response_code != 200) {
            spdlog::error("{}", response.c_str());
            return false;
        }

        std::this_thread::sleep_for(std::chrono::seconds(staing_time));

        response.clear();
        curl_easy_setopt(curl, CURLOPT_CUSTOMREQUEST, "GET");
        curl_easy_setopt(curl, CURLOPT_URL, ("http://" + camera_ip + "/ISAPI/Streaming/channels/1/picture").c_str());

        res = curl_easy_perform(curl);
        if (res == CURLE_OK) {
            long response_code;
            curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &response_code);
            if (response_code != 200) {
                spdlog::error("{}", response.c_str());
                return false;
            }

            spdlog::info("image-size: {}", response.size());
            return true;
        } else {
            spdlog::error("{}", curl_easy_strerror(res));
        }
    } else {
        spdlog::error("{}", curl_easy_strerror(res));
    }

    return false;
}

void HikvisonISAPI_v2::on_setup() {
    std::this_thread::sleep_for(std::chrono::seconds(5));
    spdlog::info("username:password: {}", username_password_);

    auto obj = nlohmann::json::parse(str_presets_);
    for (auto it = obj.begin(); it != obj.end(); ++it) {
        std::vector<std::vector<int>> values;
        for (auto &item : it.value()) {
            std::vector<int> value;
            for (auto &i : item) { value.push_back(i); }
            values.push_back(value);
        }
        presets_.insert(std::make_pair(it.key(), values));
    }

    curl_ = std::shared_ptr<CURL>(curl_easy_init(), [](CURL *curl) { curl_easy_cleanup(curl); });

    curl_easy_setopt(curl_.get(), CURLOPT_CUSTOMREQUEST, "PUT");

    curl_easy_setopt(curl_.get(), CURLOPT_HTTPAUTH, CURLAUTH_DIGEST);
    curl_easy_setopt(curl_.get(), CURLOPT_USERPWD, username_password_.c_str());

    std::string response;
    curl_easy_setopt(curl_.get(), CURLOPT_WRITEFUNCTION, write_callback);
    curl_easy_setopt(curl_.get(), CURLOPT_WRITEDATA, &response);

    for (const auto &[key, values] : presets_) {
        int preset_num = values.size();
        for (int i = 0; i < preset_num - 1; i++) {
            for (int j = values[i][0]; j <= values[i][1]; j++) {
                if (send_request(curl_.get(), camera_ip_, j, staing_time_, response)) {
#if defined(WITH_JETSON)
                    auto mem_obj = mem_pool_.alloc_mem_detach(0, 0);
#elif defined(WITH_BM1684)
                    auto mem_obj =
                        mem_pool_.alloc_mem_detach<std::shared_ptr<AVFrame>>(nullptr, FORMAT_YUV420P, 0, 0, false);
#elif defined(WITH_MLU220) || defined(WITH_MLU270)
                    auto mem_obj =
                        mem_pool_.alloc_mem_detach<std::shared_ptr<AVFrame>>(nullptr, CNCODEC_PIX_FMT_NV12, 0, 0);
#elif defined(WITH_MLU370)
                    auto mem_obj = mem_pool_.alloc_mem_detach();
#else
                    auto mem_obj = mem_pool_.alloc_mem_detach(0, 0);
#endif
                    mem_obj->data = image_wrapper::image_jpeg_dec((uint8_t *)response.data(), response.size());
                    auto frame = std::make_shared<msgs::cv_frame>(task_name_, TaskType::kCamera, 1);
                    frame->frame_info = std::make_shared<FrameInfo>(++frame_idx_, mem_obj);
                    output_image_(frame);
                }

                // FILE *fp = fopen("test.jpg", "rb");
                // fseek(fp, 0, SEEK_END);
                // response.resize(ftell(fp));
                // fseek(fp, 0, SEEK_SET);
                // fread(response.data(), 1, response.size(), fp);
                // fclose(fp);
                // auto mem_obj =
                //     mem_pool_.alloc_mem_detach<std::shared_ptr<AVFrame>>(nullptr, CNCODEC_PIX_FMT_NV12, 0, 0);
                // mem_obj->data = image_wrapper::image_jpeg_dec((uint8_t *)response.data(), response.size());
                // auto frame = std::make_shared<msgs::cv_frame>(task_name_, TaskType::kCamera, 1);
                // frame->frame_info = std::make_shared<FrameInfo>(++frame_idx_, mem_obj);
                // output_image_(frame);
                // std::this_thread::sleep_for(std::chrono::seconds(1));
            }
        }
        if (send_request(curl_.get(), camera_ip_, values.back()[0], staing_time_, response)) {
#if defined(WITH_JETSON)
                    auto mem_obj = mem_pool_.alloc_mem_detach(0, 0);
#elif defined(WITH_BM1684)
                    auto mem_obj =
                        mem_pool_.alloc_mem_detach<std::shared_ptr<AVFrame>>(nullptr, FORMAT_YUV420P, 0, 0, false);
#elif defined(WITH_MLU220) || defined(WITH_MLU270)
                    auto mem_obj =
                        mem_pool_.alloc_mem_detach<std::shared_ptr<AVFrame>>(nullptr, CNCODEC_PIX_FMT_NV12, 0, 0);
#elif defined(WITH_MLU370)
                    auto mem_obj = mem_pool_.alloc_mem_detach();
#else
                    auto mem_obj = mem_pool_.alloc_mem_detach(0, 0);
#endif
            mem_obj->data = image_wrapper::image_jpeg_dec((uint8_t *)response.data(), response.size());
            auto frame = std::make_shared<msgs::cv_frame>(task_name_, TaskType::kCamera, 1);
            frame->frame_info = std::make_shared<FrameInfo>(++frame_idx_, mem_obj);
            frame->check_report_callback_ = [this](const std::vector<FrameExtInfo> &) { return FrameType::kReport; };
            output_image_(frame);
        }

        // FILE *fp = fopen("test.jpg", "rb");
        // fseek(fp, 0, SEEK_END);
        // response.resize(ftell(fp));
        // fseek(fp, 0, SEEK_SET);
        // fread(response.data(), 1, response.size(), fp);
        // fclose(fp);
        // auto mem_obj = mem_pool_.alloc_mem_detach<std::shared_ptr<AVFrame>>(nullptr, CNCODEC_PIX_FMT_NV12, 0, 0);
        // mem_obj->data = image_wrapper::image_jpeg_dec((uint8_t *)response.data(), response.size());
        // auto frame = std::make_shared<msgs::cv_frame>(task_name_, TaskType::kCamera, 1);
        // frame->frame_info = std::make_shared<FrameInfo>(++frame_idx_, mem_obj);
        // frame->check_report_callback_ = [this](const std::vector<FrameExtInfo> &) { return FrameType::kReport; };
        // output_image_(frame);
    }
}

}// namespace nodes
}// namespace gddi