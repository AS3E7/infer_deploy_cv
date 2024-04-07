import sys
import os
import argparse
import json
import cv2
import numpy as np

import torch

sys.path.append('./yolox-pose')
from yolox.data.data_augment import ValTransform
from yolox.data.datasets import COCO_CLASSES
from yolox.exp import get_exp
from yolox.utils import fuse_model, get_model_info, postprocess, vis


def preprocess(img, new_shape=(640, 640)):
  shape = img.shape[:2]
  if isinstance(new_shape, int):
      new_shape = (new_shape, new_shape)

  r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
  ratio = r, r
  new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
  dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
  dw, dh = dw / 2, dh / 2
  if shape[::-1] != new_unpad:
      img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
  top, bottom = int(round(dh-0.1)), int(round(dh+0.1))
  left, right = int(round(dw-0.1)), int(round(dw+0.1))
  img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
  return img[:, :, ::-1].transpose([2, 0, 1]) / 255, ratio[0], (top, left)

def _nms(dets, scores, prob_threshold):
    #import pdb; pdb.set_trace()
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
 
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
 
    score_index = np.argsort(scores)[::-1]
 
    keep = []
 
    while score_index.size > 0:
        max_index = score_index[0]
        # 最大的肯定是需要的框
        keep.append(max_index)
        xx1 = np.maximum(x1[max_index], x1[score_index[1:]])
        yy1 = np.maximum(y1[max_index], y1[score_index[1:]])
        xx2 = np.minimum(x2[max_index], x2[score_index[1:]])
        yy2 = np.minimum(y2[max_index], y2[score_index[1:]])
 
        width = np.maximum(0.0, xx2 - xx1 + 1)
        height = np.maximum(0.0, yy2 - yy1 + 1)
 
        union = width * height
 
        iou = union / (areas[max_index] + areas[score_index[1:]] - union)
        ids = np.where(iou < prob_threshold)[0]
        # 以为算iou的时候没把第一个参考框索引考虑进来，所以这里都要+1
        score_index = score_index[ids+1]
    return keep

def sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s

def pose_postprocess(prediction, num_classes=1, conf_thre=0.7, nms_thre=0.45, class_agnostic=False):
    # box_corner = prediction.new(prediction.shape)
    prediction = prediction['16']
    # prediction = sigmoid(prediction)
    box_corner = prediction.copy()
    box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
    box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
    box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
    box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
    prediction[:, :, :4] = box_corner[:, :, :4]

    output = [None for _ in range(len(prediction))]

    for i, image_pred in enumerate(prediction):

        # If none are remaining => process next image
        if not image_pred.size:
            continue
        # Get score and class with highest confidence
        # class_conf, class_pred = torch.max(image_pred[:, 5: 5 + num_classes], 1, keepdim=True)
        class_conf = image_pred[:, 5:6].max(1, keepdims=True)
        class_pred = image_pred[:, 5:6].argmax(1).reshape(-1, 1)

        conf_mask = (class_conf.squeeze() >= conf_thre).squeeze()
        # conf_mask = (image_pred[:, 4] * class_conf.squeeze() >= conf_thre).squeeze()
        # Detections ordered as (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
        # force to be the same format
        keypoints = image_pred[:, 6:]

        # detections = torch.cat((image_pred[:, :5], class_conf, class_pred.float()), 1)
        # detections = np.concatenate((image_pred[:, :5], class_conf, class_pred.astype(np.float32)), 1)[conf_mask]
        detections = np.concatenate((image_pred[:, :5], class_conf, class_pred.astype(np.float32)), 1)
        detections = detections[conf_mask]
        keypoints = keypoints[conf_mask]

        if not detections.size:
            continue

        
        nms_out_index = _nms(detections[:, :4], detections[:, 4] * detections[:, 5], nms_thre)

        detections = detections[nms_out_index]
        keypoints = keypoints[nms_out_index]

        # should cat detections and keypoints here
        # results = torch.cat((detections, keypoints), axis=1)
        results = np.concatenate((detections, keypoints), axis=1)

        # TODO: revise the following
        if output[i] is None:
            output[i] = results 
        else:
            # output[i] = torch.cat((output[i], results))
            output[i] = np.concatenate((output[i], results))

    return output

_COLORS = np.array(
    [
        0.000, 0.447, 0.741,
        0.850, 0.325, 0.098,
        0.929, 0.694, 0.125,
        0.494, 0.184, 0.556,
        0.466, 0.674, 0.188,
        0.301, 0.745, 0.933,
        0.635, 0.078, 0.184,
        0.300, 0.300, 0.300,
        0.600, 0.600, 0.600,
        1.000, 0.000, 0.000,
        1.000, 0.500, 0.000,
        0.749, 0.749, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 1.000,
        0.667, 0.000, 1.000,
        0.333, 0.333, 0.000,
        0.333, 0.667, 0.000,
        0.333, 1.000, 0.000,
        0.667, 0.333, 0.000,
        0.667, 0.667, 0.000,
        0.667, 1.000, 0.000,
        1.000, 0.333, 0.000,
        1.000, 0.667, 0.000,
        1.000, 1.000, 0.000,
        0.000, 0.333, 0.500,
        0.000, 0.667, 0.500,
        0.000, 1.000, 0.500,
        0.333, 0.000, 0.500,
        0.333, 0.333, 0.500,
        0.333, 0.667, 0.500,
        0.333, 1.000, 0.500,
        0.667, 0.000, 0.500,
        0.667, 0.333, 0.500,
        0.667, 0.667, 0.500,
        0.667, 1.000, 0.500,
        1.000, 0.000, 0.500,
        1.000, 0.333, 0.500,
        1.000, 0.667, 0.500,
        1.000, 1.000, 0.500,
        0.000, 0.333, 1.000,
        0.000, 0.667, 1.000,
        0.000, 1.000, 1.000,
        0.333, 0.000, 1.000,
        0.333, 0.333, 1.000,
        0.333, 0.667, 1.000,
        0.333, 1.000, 1.000,
        0.667, 0.000, 1.000,
        0.667, 0.333, 1.000,
        0.667, 0.667, 1.000,
        0.667, 1.000, 1.000,
        1.000, 0.000, 1.000,
        1.000, 0.333, 1.000,
        1.000, 0.667, 1.000,
        0.333, 0.000, 0.000,
        0.500, 0.000, 0.000,
        0.667, 0.000, 0.000,
        0.833, 0.000, 0.000,
        1.000, 0.000, 0.000,
        0.000, 0.167, 0.000,
        0.000, 0.333, 0.000,
        0.000, 0.500, 0.000,
        0.000, 0.667, 0.000,
        0.000, 0.833, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 0.167,
        0.000, 0.000, 0.333,
        0.000, 0.000, 0.500,
        0.000, 0.000, 0.667,
        0.000, 0.000, 0.833,
        0.000, 0.000, 1.000,
        0.000, 0.000, 0.000,
        0.143, 0.143, 0.143,
        0.286, 0.286, 0.286,
        0.429, 0.429, 0.429,
        0.571, 0.571, 0.571,
        0.714, 0.714, 0.714,
        0.857, 0.857, 0.857,
        0.000, 0.447, 0.741,
        0.314, 0.717, 0.741,
        0.50, 0.5, 0
    ]
).astype(np.float32).reshape(-1, 3)


COCO_CLASSES = (
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
)


def pose_vis(img, boxes, keypoints, scores, cls_ids, conf=0.5, class_names=None):
    
    for i in range(len(boxes)):
        box = boxes[i]
        cls_id = int(cls_ids[i])
        score = scores[i]
        kpts = keypoints[i]

        if score < conf:
            continue
        x0 = int(box[0])
        y0 = int(box[1])
        x1 = int(box[2])
        y1 = int(box[3])

        color = (_COLORS[cls_id] * 255).astype(np.uint8).tolist()
        text = '{}:{:.1f}%'.format(class_names[cls_id], score * 100)
        txt_color = (0, 0, 0) if np.mean(_COLORS[cls_id]) > 0.5 else (255, 255, 255)
        font = cv2.FONT_HERSHEY_SIMPLEX

        txt_size = cv2.getTextSize(text, font, 0.4, 1)[0]
        cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)

        txt_bk_color = (_COLORS[cls_id] * 255 * 0.7).astype(np.uint8).tolist()
        cv2.rectangle(
            img,
            (x0, y0 + 1),
            (x0 + txt_size[0] + 1, y0 + int(1.5*txt_size[1])),
            txt_bk_color,
            -1
        )
        cv2.putText(img, text, (x0, y0 + txt_size[1]), font, 0.4, txt_color, thickness=1)

        # kpts = kpts.view(-1, 2)
        # for k in kpts:
        #     x, y = k
        #     x = int(x)
        #     y = int(y)
        #     cv2.circle(img, (x, y), 2, (0, 0, 255), 2)

    return img

def visual(output, img_info, cls_conf=0.35):
    ratio = img_info["ratio"]
    img = img_info["raw_img"]
    if output is None:
        return img

    bboxes = output[:, 0:4]
    keypoints = output[:, 7:]

    # preprocessing: resize
    bboxes /= ratio
    keypoints /= ratio

    cls = output[:, 6]
    # scores = output[:, 4] * output[:, 5]
    scores = output[:, 5]

    vis_res = pose_vis(img, bboxes, keypoints, scores, cls, cls_conf, COCO_CLASSES)
    return vis_res
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='for sail det_yolovx-pose py test')
    parser.add_argument('--pt', default='', required=True)
    parser.add_argument('--input', default='', required=True)
    args = parser.parse_args()

    conf_thresh = 0.7
    nms_thresh = 0.45

    # model = torch.jit.load(args.pt, torch.device('cpu'))
    exp = get_exp('exps/my_exps/yolox_s_kpt_head.py', None)
    model = exp.get_model()
    # model = torch.load(args.pt, map_location=torch.device('cpu'))
    model = model.eval().float()
    # net = sail.Engine(args.bmodel, 0, sail.IOMode.SYSIO)
    # graph_name = net.get_graph_names()[0]
    # input_name = net.get_input_names(graph_name)[0]
    
    img_info = {"id": 0}
    img_info["file_name"] = os.path.basename(args.input)
    img = cv2.imread(args.input)

    height, width = img.shape[:2]
    img_info["height"] = height
    img_info["width"] = width
    img_info["raw_img"] = img

    img1, ratio, (top, left) = preprocess(img)
    img_info["ratio"] = ratio
    # img1 = {input_name: np.array([img1], dtype=np.float32)}

    # outputs = net.process(graph_name, img1)
    outputs = model(torch.from_numpy(img1))

    outputs = pose_postprocess(
                outputs, 1, conf_thresh,
                nms_thresh, class_agnostic=True
            )
    
    result_image = visual(outputs[0], img_info, 0.5)
    cv2.imwrite('./pic/pre.jpg', result_image)


print('over')