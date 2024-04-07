// #include "transform.h"

// namespace gddeploy {
// namespace transform {
// // rga_buffer_t Normalize(const rga_buffer_t &mat, float mean, float std)
// // {
// //     rga_buffer_t matf;
// //     if (mat.type() != CV_32FC3) 
// //         mat.convertTo(matf, CV_32FC3);
// //     else 
// //         matf = mat; // reference
// //     return (matf - mean) * std;
// // }

// // rga_buffer_t Normalize(const rga_buffer_t &mat, const float *mean, const float *std)
// // {
// //     rga_buffer_t mat_copy;
// //     if (mat.type() != CV_32FC3) 
// //         mat.convertTo(mat_copy, CV_32FC3);
// //     else 
// //         mat_copy = mat.clone();

// //     for (unsigned int i = 0; i < mat_copy.rows; ++i)
// //     {
// //         cv::Vec3f *p = mat_copy.ptr<cv::Vec3f>(i);
// //         for (unsigned int j = 0; j < mat_copy.cols; ++j)
// //         {
// //             p[j][0] = (p[j][0] - mean[0]) * std[0];
// //             p[j][1] = (p[j][1] - mean[1]) * std[1];
// //             p[j][2] = (p[j][2] - mean[2]) * std[2];
// //         }
// //     }
// //     return mat_copy;
// // }

// // int Normalize(const rga_buffer_t &in_img, rga_buffer_t &out_img, const float *mean, const float *std)
// // {
// //     if (out_img.type() != CV_32FC3) 
// //         return -1;

// //     for (unsigned int i = 0; i < in_img.rows; ++i)
// //     {
// //         auto out_p = out_img.ptr<cv::Vec3f>(i);
// //         auto in_p = in_img.ptr<cv::Vec3f>(i);
// //         for (unsigned int j = 0; j < in_img.cols; ++j)
// //         {
// //             out_p[j][0] = (in_p[j][0] - mean[0]) * std[0];
// //             out_p[j][1] = (in_p[j][1] - mean[1]) * std[1];
// //             out_p[j][2] = (in_p[j][2] - mean[2]) * std[2];
// //         }
// //     }
// //     return 0;
// // }

// // int Normalize(const rga_buffer_t &in_img, rga_buffer_t &out_img, ComposeNormalizeParam &param)
// // {
// //     return Normalize(in_img, out_img, param.mean, param.std);
// // }

// int Resize(const rga_buffer_t &in_img, rga_buffer_t &out_img, int in_w, int in_h, int out_w, int out_h, ResizeProcessType type, int padding_num)
// {
//     imcvtcolor(rga_buffer_t src,
//                      rga_buffer_t dst,
//                      int sfmt,
//                      int dfmt,
//                      int mode = IM_COLOR_SPACE_DEFAULT,
//                      int sync = 1,
//                      int *release_fence_fd = NULL);
// }

// int Resize(const rga_buffer_t &in_img, rga_buffer_t &out_img, int in_w, int in_h, int out_w, int out_h, ResizeProcessType type, int padding_num)
// {
//     if (type == RESIZE_PT_LEFT_TOP){
//         int w, h, x, y;
//         float r_w = out_h / (in_w * 1.0);
//         float r_h = out_w / (in_h * 1.0);

//         if (r_h > r_w)
//         {
//             w = out_h;
//             h = r_w * in_h;
//             x = 0;
//             y = 0;
//         }
//         else
//         {
//             w = r_h * in_w;
//             h = out_w;
//             x = 0;
//             y = 0;
//         }
//         rga_buffer_t re(h, w, CV_8UC3);
//         rga_buffer_t re_padding(out_w, out_h, CV_8UC3, cv::Scalar(padding_num, padding_num, padding_num));

//         cv::resize(in_img, re, re.size(), 0, 0, cv::INTER_LINEAR);

//         re.copyTo(re_padding(cv::Rect(x, y, re.cols, re.rows)));

//         std::memcpy((uint8_t *)out_img.data, re_padding.data, 3 * out_w * out_h * sizeof(uint8_t));

//     } else if (type == RESIZE_PT_CENTER){
//         int w, h, x, y;
//         float r_w = out_h / (in_w * 1.0);
//         float r_h = out_w / (in_h * 1.0);

//         if (r_h > r_w)
//         {
//             w = out_h;
//             h = r_w * in_h;
//             x = 0;
//             y = (out_h - h) / 2;;
//         }
//         else
//         {
//             w = r_h * in_w;
//             h = out_w;
//             x = (out_w - w) / 2;
//             y = 0;
//         }
//         rga_buffer_t re(h, w, CV_8UC3);
//         rga_buffer_t re_padding(out_w, out_h, CV_8UC3, cv::Scalar(padding_num, padding_num, padding_num));

//         cv::resize(in_img, re, re.size(), 0, 0, cv::INTER_LINEAR);

//         re.copyTo(re_padding(cv::Rect(x, y, re.cols, re.rows)));

//         std::memcpy((uint8_t *)out_img.data, re_padding.data, 3 * out_w * out_h * sizeof(uint8_t));

//     } else if (type == RESIZE_PT_CROP){
//         int w, h, x, y;
//         float r_w = out_h / (in_w * 1.0);
//         float r_h = out_w / (in_h * 1.0);

//         if (r_h > r_w)
//         {
//             w = out_h;
//             h = r_w * in_h;
//             x = 0;
//             y = (out_h - h) / 2;;
//         }
//         else
//         {
//             w = r_h * in_w;
//             h = out_w;
//             x = (out_w - w) / 2;
//             y = 0;
//         }
//         rga_buffer_t re(h, w, CV_8UC3);
//         rga_buffer_t re_padding(out_w, out_h, CV_8UC3, cv::Scalar(padding_num, padding_num, padding_num));

//         cv::resize(in_img, re, re.size(), 0, 0, cv::INTER_LINEAR);

//         re.copyTo(re_padding(cv::Rect(x, y, re.cols, re.rows)));
//         std::memcpy((uint8_t *)out_img.data, re_padding.data, 3 * out_w * out_h * sizeof(uint8_t));

//     } else {
//         rga_buffer_t re(out_h, out_w, CV_8UC3, out_img.data);

//         cv::resize(in_img, re, re.size(), 0, 0, cv::INTER_LINEAR);
//         std::memcpy((uint8_t *)out_img.data, re.data, 3 * out_w * out_h * sizeof(uint8_t));
//     }
    

//     return 0;
// }

// int Resize(const rga_buffer_t &in_img, rga_buffer_t &out_img, ComposeResizeParam param)
// {
//     return Resize(in_img, out_img, param.in_w, param.in_h, param.out_w, param.out_h, param.type, param.padding_num);
// }

// int Compose(BufSurfaceParams *in_surf_param, BufSurfaceParams *out_surf_param, std::vector<std::pair<std::string, gddeploy::any>> ops)
// {
//     rga_buffer_t src, dst;
// #if RKNN_RGA_VEWSION_1_3
//     src = wrapbuffer_virtualaddr(src_buf, src_width, src_height, src_format);
//     dst = wrapbuffer_virtualaddr(dst_buf, dst_width, dst_height, dst_format);
//     if(src.width == 0 || dst.width == 0) {
//         printf("wrapbuffer_virtualaddr error, %s, %s\n", __FUNCTION__, imStrError());
//         goto release_buffer;
//     }

//     ret = imresize(src, dst);
//     if (ret != IM_STATUS_SUCCESS) {
//         printf("unning failed, %s\n", imStrError((IM_STATUS)ret));
//         goto release_buffer;
//     }


// #elif RKNN_RGA_VEWSION_1_9
//     src = importbuffer_virtualaddr(src_buf, src_buf_size);
//     dst = importbuffer_virtualaddr(dst_buf, dst_buf_size);
//     if (src == 0 || dst == 0) {
//         printf("importbuffer failed!\n");
//         goto release_buffer;
//     }
//     src_img = wrapbuffer_handle(src, src_width_align, src_height, src_format);
//     dst_img = wrapbuffer_handle(dst, dst_width_align, dst_height, dst_format);
// #endif

//     int ret = imcheck(src, dst);
//     if (IM_STATUS_NOERROR != ret) {
//         printf("%d, check error! %s", __LINE__, imStrError((IM_STATUS)ret));
//         return -1;
//     }

//     rga_buffer_t in = in_img;
//     rga_buffer_t out;
//     for (auto &iter : ops){
//         if (&iter == &ops.back()){
//             out = out_img;
//         }
//         std::string op = iter.first;
//         gddeploy::any param = iter.second;

//         if ( op == "resize"){
//             ComposeResizeParam resize_param = gddeploy::any_cast<ComposeResizeParam>(param);
//             Resize(in, out, resize_param);

//         } else if ( op == "normalize"){
//             ComposeNormalizeParam norn_param = gddeploy::any_cast<ComposeNormalizeParam>(param);
//             Normalize(in, out, norn_param);

//         } else if ( op == "bgr2rgb"){
//             bool is_open = gddeploy::any_cast<bool>(param);
//             if (is_open == false)
//                 continue;

//             cv::cvtColor(in, out, cv::COLOR_BGR2RGB);
//         } else if ( op == "nv122rgb"){
//             bool is_open = gddeploy::any_cast<bool>(param);
//             if (is_open == false)
//                 continue;

//             in.convertTo(out, CV_32FC3, 1.0);
//         } else if ( op == "hwc2chw"){
//             bool is_open = gddeploy::any_cast<bool>(param);
//             if (is_open == false)
//                 continue;

//             int img_w = in.cols;
//             int img_h = in.rows;

//             std::vector<rga_buffer_t> bgrChannels;
//             bgrChannels.emplace_back(img_w, img_h, CV_32F);
//             bgrChannels.emplace_back(img_w, img_h, CV_32F);
//             bgrChannels.emplace_back(img_w, img_h, CV_32F);
//             cv::split(in, bgrChannels);

//             out = rga_buffer_t(img_h, img_w, CV_32F);

//             for (int i = 0; i < 3; i++)
//             {
//                 std::memcpy((float *)out.data + i * img_w * img_h, bgrChannels[i].data, img_w * img_h * sizeof(float));
//             }
//         }
//         in = out;
//     }

//     return 0;
// }

// }
// }