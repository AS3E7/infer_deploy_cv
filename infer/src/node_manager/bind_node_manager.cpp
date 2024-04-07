//
// Created by cc on 2021/11/26.
//
#include "node_manager.hpp"

/**
* 这个原文件只有一个功能，就是把所有可以使用的节点在这里进行注册，不要引入其它的任何对象
*/

#include "nodes/ffvcodec/feature_library.h"
#include "nodes/ffvcodec/hikvison_isapi_v2.h"
#include "nodes/postprocess/aggregation_node_v2.h"
#include "nodes/postprocess/box_filter_node_v2.h"
#include "nodes/postprocess/box_scaler_node_v2.h"
#include "nodes/postprocess/camera_status_node_v2.h"
#include "nodes/postprocess/check_moving_node_v2.h"
#include "nodes/postprocess/crop_image_node_v2.h"
#include "nodes/postprocess/cross_counter_node_v2.h"
#include "nodes/postprocess/delay_timer_node_v2.h"
#include "nodes/postprocess/dial_plate_node_v2.h"
#include "nodes/postprocess/label_counter_condition_node_v2.h"
#include "nodes/postprocess/label_logic_interpreter_node_v2.h"
#include "nodes/postprocess/label_stay_node_v2.h"
#include "nodes/postprocess/logic_gate_node_v2.h"
#include "nodes/postprocess/mosaic_node_v2.h"
#include "nodes/postprocess/msg_subscribe_node_v2.h"
#include "nodes/postprocess/prob_filter_node_v2.h"
#include "nodes/postprocess/regional_counter_node_v2.h"
#include "nodes/postprocess/roi_crop_node_v2.h"
#include "nodes/postprocess/roi_filter_node_v2.h"
#include "nodes/postprocess/roi_perspective_transform_v2.h"
#include "nodes/postprocess/sequence_of_events_statistic_node_v2.h"
#include "nodes/postprocess/target_counter_node_v2.h"
#include "nodes/postprocess/target_cover_logic_node_v2.h"
#include "nodes/postprocess/target_status_node_v2.h"
#include "nodes/postprocess/connected_components_node_v2.h"
#include "nodes/postprocess/regular_expression_v2.h"
#include "nodes/postprocess/status_update_node_v2.h"
#include "nodes/postprocess/edge_detect_node_v2.h"
#include "nodes/postprocess/target_time_judge_v2_node_v2.h"
#include "nodes/postprocess/target_status_node_v2.h"
#include "nodes/postprocess/connected_components_node_v2.h"
#include "nodes/postprocess/regular_expression_v2.h"
#include "nodes/postprocess/color_judge_node_v2.h"

#if defined(WITH_RV1126)
#include "nodes/postprocess/conn_detector_node_v2.h"
#include "nodes/postprocess/vehicle_license_parser_node_v2.h"
#include "nodes/postprocess/vehicle_license_spliter_node_v2.h"

#endif

#if defined(WITH_FFMPEG)
#include "nodes/ffvcodec/decoder_node_v1.hpp"
#include "nodes/ffvcodec/decoder_node_v1_1.hpp"
#include "nodes/ffvcodec/decoder_node_v2.hpp"
#include "nodes/ffvcodec/demuxer_node_v1.hpp"
#include "nodes/ffvcodec/demuxer_node_v1_1.hpp"
#include "nodes/ffvcodec/demuxer_node_v2.hpp"
#include "nodes/ffvcodec/encoder_node_v2.hpp"
#include "nodes/ffvcodec/media_decoder_node_v2.h"
#include "nodes/ffvcodec/remuxer_node_v2.hpp"
#include "nodes/message_ref_time.hpp"
#endif

#if defined(WITH_OPENCV)
#include "nodes/ffvcodec/image_server_node_v2.h"

// #include "OpenCV/cv_imshow_node.hpp"
// #include "OpenCV/cv_imshow_v2.hpp"
#include "OpenCV/cv_to_cvmat.hpp"
#include "nodes/ffvcodec/opencv_decoder_node_v2.hpp"
// #include "nodes/postprocess/tracking_node_v2.hpp"
#include "nodes/video_capter_node.hpp"

#include "nodes/algorithm/classifier_node_v2.h"
#include "nodes/algorithm/detection_node_v2.h"
#include "nodes/algorithm/inference_node_v2.h"
#include "nodes/postprocess/action_counter_node_v2.h"
#include "nodes/postprocess/action_pair_node_v2.h"
#include "nodes/postprocess/direction_node_v2.h"
#include "nodes/postprocess/regional_alert_node_v2.h"
#include "nodes/postprocess/seg_calculation_node_v2.h"

#endif

#include "nodes/postprocess/target_tracker_node_v2.h"

#include "nodes/postprocess/graphics_node_v2.h"
#include "nodes/postprocess/report_node_v2.h"

#if defined(WITH_OPENCV)
#include "modules/zmq_cv_post/cppzmq_cvmat_push_node.hpp"
#include "nodes/jpeg_previewer_node.hpp"
#endif

void gddi::NodeManager::bind_all_node_to_node_manager() {

    bind_node_creator<gddi::nodes::BoxFilter_v2>("目标框过滤组件");
    bind_node_creator<gddi::nodes::CropImage_v2>("扣图组件");
    bind_node_creator<gddi::nodes::BoxFilter_v2>("目标框过滤组件");
    bind_node_creator<gddi::nodes::LogicGate_v2>("逻辑门组件");
    bind_node_creator<gddi::nodes::RoiFilter_v2>("ROI 组件");
    bind_node_creator<gddi::nodes::MsgSubscribe_v2>("消息订阅组件");
    bind_node_creator<gddi::nodes::ProbFilter_v2>("置信度过滤组件");
    bind_node_creator<gddi::nodes::RoiCrop_v2>("ROI扣图组件");
    bind_node_creator<gddi::nodes::CrossCounter_v2>("越界计数组件");
    bind_node_creator<gddi::nodes::CameraStatus_v2>("摄像头状态组件");
    bind_node_creator<gddi::nodes::TargetCounter_v2>("目标计数组件");
    bind_node_creator<gddi::nodes::DialPlate_v2>("表盘读数组件");
    bind_node_creator<gddi::nodes::Aggregation_v2>("聚合组件");
    bind_node_creator<gddi::nodes::FeatureLibrary_v2>("特征建库组件");
    bind_node_creator<gddi::nodes::RoiPerspectiveTransform_v2>("透视变换组件");
    bind_node_creator<gddi::nodes::HikvisonISAPI_v2>("海康 ISAPI 组件");
    bind_node_creator<gddi::nodes::CheckMoving_v2>("运动检测组件");
    bind_node_creator<gddi::nodes::LabelCounterCondition_v2>("标签计数比较组件");
    bind_node_creator<gddi::nodes::SequenceOfEventsStatistic_v2>("事件序列统计组件");
    bind_node_creator<gddi::nodes::RegionalCounter_v2>("区域目标计数组件");
    bind_node_creator<gddi::nodes::DelayTimer_v2>("延迟计时器组件");
    bind_node_creator<gddi::nodes::TargetStatus_v2>("目标移动检测组件");
    bind_node_creator<gddi::nodes::LabelLogicInterpreter_v2>("标签逻辑解释器组件");
    bind_node_creator<gddi::nodes::LabelStay_v2>("标签停留判读组件");
    bind_node_creator<gddi::nodes::TargetCoverLogic_v2>("目标覆盖逻辑组件");
    bind_node_creator<gddi::nodes::BoxScaler_v2>("目标框缩放组件");
    bind_node_creator<gddi::nodes::ConnectedComponents_v2>("连通域组件");
    bind_node_creator<gddi::nodes::RegularExpression_v2>("正则表达式组件");
    bind_node_creator<gddi::nodes::StatusUpdate_v2>("目标状态更新组件");
    bind_node_creator<gddi::nodes::EdgeDetect_v2>("目标边缘检测组件");
    bind_node_creator<gddi::nodes::TargetTimeJudge_v2>("目标消失时长组件");
    bind_node_creator<gddi::nodes::BoxScaler_v2>("目标框缩放组件");
    bind_node_creator<gddi::nodes::ConnectedComponents_v2>("连通域组件");
    bind_node_creator<gddi::nodes::RegularExpression_v2>("正则表达式组件");
    bind_node_creator<gddi::nodes::ColorJudge_v2>("颜色判断组件");
#if defined(WITH_RV1126)
    bind_node_creator<gddi::nodes::ConnDetector_v2>("连通域检测器组件");
    bind_node_creator<gddi::nodes::VehicleLicenseSpliter_v2>("驾驶证正反面识别组件");
    bind_node_creator<gddi::nodes::VehicleLicenseParser_v2>("驾驶证字段解析组件");
#endif

#if defined(WITH_FFMPEG)
    bind_node_creator<gddi::nodes::lib_av::Demuxer_v1>("码流解包组件");
    bind_node_creator<gddi::nodes::lib_av::Demuxer_v1_1>("码流解包组件");
    bind_node_creator<gddi::nodes::Decoder_v1>("解码组件");
    bind_node_creator<gddi::nodes::Decoder_v1_1>("解码组件");
    bind_node_creator<gddi::nodes::MessageRefTime_v1>("消息时间分析打印(debug)");

    bind_node_creator<gddi::nodes::Demuxer_v2>("码流解包组件");
    bind_node_creator<gddi::nodes::Decoder_v2>("音视频解码组件");
    bind_node_creator<gddi::nodes::Encoder_v2>("音视频编码组件");
    bind_node_creator<gddi::nodes::Remuxer_v2>("码流封包组件");
    bind_node_creator<gddi::nodes::MediaDecoder_v2>("多媒体解码组件");
#endif

#if defined(WITH_OPENCV)
    // bind_node_creator<gddi::nodes::Tracking_v2>("目标跟踪组件");

    bind_node_creator<gddi::nodes::cv_nodes::AvToCvMat_v1>("AVFrame to cv::Mat，sws方法");
    bind_node_creator<gddi::nodes::cv_nodes::AvToCvMat_v1_1>("AVFrame to cv::Mat，sws方法, imencode");
    bind_node_creator<gddi::nodes::cv_nodes::AvToCvMat_opencv2>("AVFrame to cv::Mat，cvtColor方法");
    bind_node_creator<gddi::nodes::cv_nodes::AvToCvMat_yuv2rgb>("AVFrame to cv::Mat，github yuv2rgb std");

#ifdef __x86_64
    // bind_node_creator<cv_nodes::CvImShow_v1>("opencv2图片显示");
    // bind_node_creator<cv_nodes::CvImShow_v2>("opencv2图片显示");
#endif

    bind_node_creator<gddi::nodes::VideoCaptureNode>("视频读取组件");
    bind_node_creator<gddi::nodes::CropImage_v2>("扣图组件");
    bind_node_creator<gddi::nodes::Direction_v2>("方向计算组件");
    bind_node_creator<gddi::nodes::RegionalAlert_v2>("区域告警组件");
    bind_node_creator<gddi::nodes::SegCalculation_v2>("分割计算组件");
    bind_node_creator<gddi::nodes::ActionCounter_v2>("动作计数组件");
    bind_node_creator<gddi::nodes::ActionPair_v2>("动作匹配组件");
    bind_node_creator<gddi::nodes::Mosaic_v2>("马赛克脱敏组件");
#endif

#if defined(WITH_OPENCV)
    bind_node_creator<gddi::nodes::Graphics_v2>("图像目标框信息绘制组件");
    bind_node_creator<gddi::nodes::Report_v2>("目标框信息上报组件");
#endif

    bind_node_creator<gddi::nodes::TargetTracker_v2>("目标跟踪组件");

#if defined(WITH_OPENCV)
    bind_node_creator<gddi::nodes::Detection_v2>("检测推理组件");
    bind_node_creator<gddi::nodes::Classifier_v2>("分类推理组件");
    bind_node_creator<gddi::nodes::Inference_v2>("推理组件");
    bind_node_creator<gddi::nodes::ImageServer_v2>("常驻图像检测组件");
#endif

#if defined(WITH_OPENCV)
    bind_node_creator<gddi::nodes::CppZmqCvMatPusher_v1>("ZMQ cv::Mat图片推送");
    bind_node_creator<gddi::nodes::JpegPreviewer_v2>("ZMQ cv::Mat图片推送");
#endif
}
