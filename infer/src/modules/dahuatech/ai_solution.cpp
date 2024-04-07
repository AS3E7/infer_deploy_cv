#include "ai_solution.h"
#include "version.h"
#include <cstdio>

#if defined(WITH_MLU220)
#include "cncv.h"
#include "gdd_api.h"
#include "gdd_result_type.h"

#define FFALIGN(x, a) (((x) + (a)-1) & ~((a)-1))

struct InferHandle {
    std::unique_ptr<gdd::GddInfer> algo_;
    std::function<void(const int64_t, const gdd::InferResult &)> callback_;

    std::vector<ai_result_s> results_;
};

static infer_server::video::PixelFmt cvt_image_format(const uint8_t color) {
    switch (color) {
        case AI_CS_YUV420: return infer_server::video::PixelFmt::I420;
        case AI_CS_NV12: return infer_server::video::PixelFmt::NV12;
        case AI_CS_NV21: return infer_server::video::PixelFmt::NV21;
        case AI_CS_Y: throw std::runtime_error("Unsupported image format");
    }
}

int32_t ai_init() {
    cnrtInit(0);
    return 0;
}

int32_t ai_get_version(ai_verion_s *pVersion) {
    pVersion->protocol_version = 0;
    sprintf(pVersion->version, PROJECT_VERSION);

    return 0;
}

int32_t ai_get_capacities(ai_capacity_s *pCaps) {
    pCaps->max_result_obj = UINT32_MAX;
    pCaps->schedule_mode = AI_SCHEDULE_SYNC | AI_SCHEDULE_ASYNC;
    pCaps->max_image_width = 7680;
    pCaps->max_image_height = 4320;

    return 0;
}

int32_t ai_create(void **ppHandle, ai_create_s *pCreate) {
    *ppHandle = new InferHandle;
    auto handle = reinterpret_cast<InferHandle *>(*ppHandle);

    handle->algo_ = std::make_unique<gdd::GddInfer>();
    handle->algo_->Init(0, 0, "");
    if (handle->algo_->LoadModel(pCreate->config_file, "") != 0) {
        printf("Failed to load model: %s\n", pCreate->config_file);
        return -1;
    }

    return 0;
}

int32_t ai_register_callback(void *pHandle, uint32_t cbType, void *pCbFunc, void *pCbParam) {
    auto handle = reinterpret_cast<InferHandle *>(pHandle);
    if (cbType == AI_CB_TYPE_PROCESS_DONE) {
        handle->callback_ = [pCbFunc, pCbParam](const int64_t frame_id, const gdd::InferResult &infer_result) {
            auto &detect_img = infer_result.detect_result.detect_imgs[0];
            ai_result_s result;
            result.frame_id = frame_id;
            result.result_length = 1;
            result.obj_num = detect_img.detect_objs.size();
            result.objs = new ai_obj_s[result.obj_num];

            int index = 0;
            for (auto &obj : detect_img.detect_objs) {
                (result.objs + index)->base.id = 0;
                (result.objs + index)->base.confidence = obj.score;
                (result.objs + index)->base.rect =
                    ai_rect_s{(int32_t)obj.bbox.x, (int32_t)obj.bbox.y, (uint32_t)obj.bbox.w, (uint32_t)obj.bbox.h};
                printf("id: %d, x: %.2f, y: %.2f, w: %.2f, h: %.2f, score: %.2f\n", 0, obj.bbox.x, obj.bbox.y,
                       obj.bbox.w, obj.bbox.h, obj.score);
                ++index;
            }

            delete[] result.objs;
        };
    }

    return 0;
}

int32_t ai_get_result(void *pHandle, uint32_t *num, ai_result_s *pResult[]) {
    auto handle = reinterpret_cast<InferHandle *>(pHandle);
    *num = handle->results_.size();
    for (int i = 0; i < *num; i++) { pResult[i] = &handle->results_[i]; }

    return 0;
}

int32_t ai_release_result(void *pHandle, uint32_t num, ai_result_s *pResult[]) {
    auto handle = reinterpret_cast<InferHandle *>(pHandle);
    handle->results_.clear();

    return 0;
}

int32_t ai_process(void *pHandle, ai_input_s *pInput, uint32_t num, ai_result_s *pSyncResult[]) {
    auto handle = reinterpret_cast<InferHandle *>(pHandle);
    infer_server::video::VideoFrame video_frame;

    if (pInput->frames->frame_type == AI_INPUT_TYPE_IMAGE) {
        auto image = reinterpret_cast<ai_input_image_s *>(pInput->frames->frame);

        int stride_width = FFALIGN(image->width, 128);
        int stride_height = FFALIGN(image->height, 16);

        if (image->colorspace == AI_CS_YUV420) {

            video_frame.width = stride_width;
            video_frame.height = stride_height;
            video_frame.format = cvt_image_format(image->colorspace);
            video_frame.stride[0] = stride_width;
            video_frame.stride[1] = stride_width / 4;
            video_frame.stride[2] = stride_width * stride_height / 4;

            uint8_t *plane[MAX_PLANE_NUM];
            cnrtMalloc((void **)&plane[0], stride_width * stride_height);
            cnrtMalloc((void **)&plane[1], stride_width * stride_height / 4);
            cnrtMalloc((void **)&plane[2], stride_width * stride_height / 4);
            cnrtMemcpy(plane[0], (uint8_t *)image->main_data, image->width * image->height,
                       CNRT_MEM_TRANS_DIR_HOST2DEV);
            cnrtMemcpy(plane[1], (uint8_t *)image->main_data + image->width * image->height,
                       image->width * image->height / 4, CNRT_MEM_TRANS_DIR_HOST2DEV);
            cnrtMemcpy(plane[2], (uint8_t *)image->main_data + image->width * image->height * 5 / 4,
                       image->width * image->height / 4, CNRT_MEM_TRANS_DIR_HOST2DEV);

            video_frame.plane[0] = infer_server::Buffer(plane, stride_width * stride_height, nullptr, 0);
            video_frame.plane[1] = infer_server::Buffer(plane + stride_width * stride_height,
                                                        stride_width * stride_height / 4, nullptr, 0);
            video_frame.plane[2] = infer_server::Buffer(plane + stride_width * stride_height * 5 / 4,
                                                        stride_width * stride_height / 4, nullptr, 0);
        } else if (image->colorspace == AI_CS_NV12) {
            video_frame.width = image->width;
            video_frame.height = image->height;
            video_frame.format = cvt_image_format(image->colorspace);
            video_frame.stride[0] = stride_width;
            video_frame.stride[1] = stride_width;
            video_frame.plane_num = 2;

            uint8_t *plane[MAX_PLANE_NUM];
            cnrtMalloc((void **)&plane[0], stride_width * stride_height);
            cnrtMalloc((void **)&plane[1], stride_width * stride_height / 2);
            cnrtMemcpy(plane[0], (uint8_t *)image->main_data, image->width * image->height,
                       CNRT_MEM_TRANS_DIR_HOST2DEV);
            cnrtMemcpy(plane[1], (uint8_t *)image->main_data + image->width * image->height,
                       image->width * image->height / 2, CNRT_MEM_TRANS_DIR_HOST2DEV);

            video_frame.plane[0] = infer_server::Buffer(plane[0], stride_width * stride_height, nullptr, 0);
            video_frame.plane[1] = infer_server::Buffer(plane[1], stride_width * stride_height / 2, nullptr, 0);
        }

        auto in_packet = infer_server::Package::Create(1);
        in_packet->data[0]->Set(video_frame);
        in_packet->data[0]->SetUserData(image->frame_id);

        handle->algo_->InferAsync(
            in_packet,
            [handle](infer_server::Status status, infer_server::PackagePtr packet, infer_server::any user_data) {
                for (auto &batch_data : packet->data) {
                    if (batch_data) {
                        int index = 0;
                        auto &result = batch_data->GetLref<gdd::InferResult>();
                        // auto frame_idx = batch_data->GetUserData<int64_t>();
                        if (handle->callback_) { handle->callback_(0, result); };
                    }
                }
            });
    }

    return 0;
}

int32_t ai_config(void *pHandle, const ai_config_s *pConfig) { return 0; }

int32_t ai_destory(void *pHandle) {
    delete (InferHandle *)pHandle;
    return 0;
}

int32_t ai_deinit() {
    cnrtDestroy();
    return 0;
}

#endif