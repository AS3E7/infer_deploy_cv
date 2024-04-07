""" Copyright 2016-2022 by Bitmain Technologies Inc. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""
from __future__ import division

import argparse
import json
import os
import time

import cv2
import numpy as np
import onnxruntime
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tqdm import tqdm


def preprocess(img, new_shape):
    """ Preprocessing of a frame.

    Args:
      image : np.array, input image
      detected_size : list, yolov3 detection input size

    Returns:
      Preprocessed data.
    """

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

ANCHORS = np.array([
    [10,  13, 16,  30,  33,  23],
    [30,  61, 62,  45,  59,  119],
    [116, 90, 156, 198, 373, 326]
])

ANCHOR_GRID = ANCHORS.reshape(3, -1, 2).reshape(3, 1, -1, 1, 1, 2)
STRIDES = [8, 16, 32]
CONF_THR = 0.6
IOU_THR = 0.5


def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = np.zeros_like(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s


def make_grid(nx, ny):
    z = np.stack(np.meshgrid(np.arange(nx), np.arange(ny)), 2)
    return z.reshape(1, 1, ny, nx, 2).astype(np.float32)


def predict_preprocess(x):
    for i in range(len(x)):
        bs, no, ny, nx,  = x[i].shape
        na = 3
        no = int(no/3)
        # x[i] = x[i].transpose(0, 2, 3, 1).reshape(bs, na, ny, nx, no)
        x[i] = x[i].reshape(bs, na, no, ny, nx).transpose(0, 1, 3, 4, 2)
        grid = make_grid(nx, ny)
        # x[i] = sigmoid(x[i])
        x[i][..., 0:2] = (x[i][..., 0:2] * 2. - 0.5 + grid) * STRIDES[i]
        x[i][..., 2:4] = (x[i][..., 2:4] * 2) ** 2 * ANCHOR_GRID[i]
        x[i] = x[i].reshape(bs, -1, no)
    return np.concatenate(x, 1)


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


def non_max_suppression(prediction, conf_thres=0.1, iou_thres=0.6, classes=None):
    """Performs Non-Maximum Suppression (NMS) on inference results

    Returns:
         detections with shape: nx6 (x1, y1, x2, y2, conf, cls)
    """
#    prediction = [prediction['147'], prediction['148'], prediction['149']]
    prediction = predict_preprocess(prediction)
    nc = prediction[0].shape[1] - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    # Settings
    min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
    max_det = 300  # maximum number of detections per image
    time_limit = 10.0  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label = nc > 1  # multiple labels per box (adds 0.5ms/img)

    t = time.time()
    output = [None] * prediction.shape[0]
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            # i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
            i, j = np.stack((x[:, 5:] > conf_thres).nonzero())
            x = np.concatenate((box[i], x[i, j + 5, None], j[:, None].astype(np.float32)), 1)
        else:  # best class only
            # conf, j = x[:, 5:].max(1, keepdim=True)
            conf = x[:, 5:].max(1, keepdims=True)
            j = x[:, 5:].argmax(1).reshape(-1, 1)
            x = np.concatenate((box, conf, j.astype(np.float32)), 1)[conf.reshape(-1) > conf_thres]

        # Filter by class
        if classes:
            x = x[(x[:, 5:6] == np.array(classes)).any(1)]

        # Apply finite constraint
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]

        # If none remain process next image
        n = x.shape[0]  # number of boxes
        if not n:
            continue

        # Sort by confidence
        # x = x[x[:, 4].argsort(descending=True)]

        # Batched NMS
        c = x[:, 5:6] * max_wh  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = _nms(boxes, scores, iou_thres)
        if len(i) > max_det:  # limit detections
            i = i[:max_det]

        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            break  # time limit exceeded

    return output



IMAGES_NUM = 100
BASE_PATH = '/home/linaro/work/face/images/valid/'
if __name__ == '__main__':
    """ A YOLOv3 example.
    """
    parser = argparse.ArgumentParser(description='for sail det_yolov3 py test')
    parser.add_argument('--onnx', default='', required=True)
    parser.add_argument('--input', default='', required=False)
    
    args = parser.parse_args()
    
    sess = onnxruntime.InferenceSession(args.onnx)


    detected_size = (640, 640)
    threshold = 0.1
    nms_threshold = 0.5

    input_path = args.input
    input_name = sess.get_inputs()[0].name
    import os
    import json
    with open(input_path) as g:
        js = json.load(g)
    results = []

    for img in js['images']:
        img_f_path = img['file_name']
        img_f_path = os.path.join('/volume1/gddi-data/lgy/dataset/mask2/0/images/train/', img_f_path)

        img = cv2.imread(img_f_path)
        # preprocess
        # cv2.imwrite("/home/linaro/tmp/pic/mask3.jpg", img)
        data, ratio, (top, left) = preprocess(img, detected_size)
        data = data.astype(np.float32)

        # pre_data = np.fromfile('../cambricon/code/gddi-cambricon/data/pic/pre_data.bin',dtype=np.float32).reshape((3,640,640))
        # compare = (np.round(data,4) == np.round(pre_data,4))

        # data.tofile("helmet3_raw.bin")
        input_data = {input_name: np.array([data], dtype=np.float32)}
        output = sess.run(None, input_data)
        # postprocess
        prediction = non_max_suppression(output, conf_thres=threshold, iou_thres=nms_threshold, classes=None)

        for i in prediction:
            if i is None:
                continue
            for j in i:
                bbox = j[:4]
                bbox[0::2] -= left
                bbox[1::2] -= top
                bbox /= ratio
                prob = float(j[4])
                clse = int(j[5])
                results.append(dict(
                    image_id=img['id'],
                    category_id=clse,
                    bbox=[float(bbox[0]), float(bbox[1]), float(bbox[2]-bbox[0]), float(bbox[3]-bbox[1])],
                    score=prob))
    
    with open('mask_predictions.json', 'w') as f:
      json.dump(results, f, indent=4)
