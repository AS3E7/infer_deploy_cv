#include "core/alg.h"
#include <cstddef>
#include "common/logger.h"

using namespace gddeploy;

AlgManager *AlgManager::pInstance_ = nullptr;


#include "register/alg/detect/alg_detect.h"
#include "register/alg/pose/alg_pose.h"
#include "register/alg/seg/alg_seg.h"
#include "register/alg/classify/alg_classify.h"
#include "register/alg/image_retrieval/alg_image_retrieval.h"
#include "register/alg/face_retrieval/alg_face_retrieval.h"
#include "register/alg/ocr_detect/alg_ocr_detect.h"
#include "register/alg/ocr_retrieval/alg_ocr_retrieval.h"
#include "register/alg/action/alg_action.h"

namespace gddeploy
{
int register_alg_module()
{
    AlgManager* algmgr = AlgManager::Instance();
    GDDEPLOY_INFO("[Register] register alg module");

    DetectAlgCreator *alg_detect_creator = new DetectAlgCreator();
    algmgr->RegisterAlg("detection", "yolo", alg_detect_creator);
    algmgr->RegisterAlg("detection", "yolov6", alg_detect_creator);

    PoseAlgCreator *alg_pose_creator = new PoseAlgCreator();
    algmgr->RegisterAlg("pose", "yolox", alg_pose_creator);
    algmgr->RegisterAlg("pose", "rtmpose", alg_pose_creator);

    SegAlgCreator *alg_seg_creator = new SegAlgCreator();
    algmgr->RegisterAlg("segmentation", "OCRNet", alg_seg_creator);

    ClassifyAlgCreator *alg_classify_creator = new ClassifyAlgCreator();
    algmgr->RegisterAlg("classification", "ofa", alg_classify_creator);

    ImageRetrievalAlgCreator *alg_image_retrieval_creator = new ImageRetrievalAlgCreator();
    algmgr->RegisterAlg("image-retrieval", "dolg", alg_image_retrieval_creator);

    FaceRetrievalAlgCreator *alg_face_retrieval_creator = new FaceRetrievalAlgCreator();
    algmgr->RegisterAlg("image-retrieval", "arcface", alg_face_retrieval_creator);

    OcrDetectAlgCreator *alg_ocr_det_creator = new OcrDetectAlgCreator();
    algmgr->RegisterAlg("ocr", "ocr_det", alg_ocr_det_creator);

    OcrRecAlgCreator *alg_ocr_rec_creator = new OcrRecAlgCreator();
    algmgr->RegisterAlg("ocr", "ocr_rec", alg_ocr_rec_creator);
    algmgr->RegisterAlg("ocr", "resnet31v2ctc", alg_ocr_rec_creator);

    ActionAlgCreator *alg_action_creator = new ActionAlgCreator();
    algmgr->RegisterAlg("action", "tsn_gddi", alg_action_creator);

    return 0;
}
}