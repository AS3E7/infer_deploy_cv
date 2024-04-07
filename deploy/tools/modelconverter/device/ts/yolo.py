import os
import time
import torch
import pickle
import argparse
import numpy as np
from torch import nn
from torch.autograd import Variable

import cv2
import numpy as np
import onnxruntime
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tqdm import tqdm

from lib.modules import *
from lib.quant_utils.utils import get_quant_mode
from custom_model.models import QuantModel
from custom_model.models.gru import load_cmd_data

sys.path.append('/root/thirdparty/yolov5/')
# sys.path.append('/volume1/gddi-data/lgy/cambricon/thirdparty/')
from models.experimental import attempt_download, attempt_load
from models.yolo import Model, Detect
from models.common import Conv
# from det_yolov5 import preprocess, non_max_suppression

def preprocess(img, new_shape = (640, 640)):
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
    # return img[:, :, ::-1].transpose([2, 0, 1]) / 255, ratio[0], top, left
    return img[:, :, ::-1].transpose([2, 0, 1]), ratio[0], top, left


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
        # bs, na, ny, nx, no = x[i].shape
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
    

img_dir =  '/gddi_data/0/images/train/'
anno_file = '/gddi_data/0/annotation/train/anno.json'
# img_dir =  '/volume1/gddi-data/lgy/dataset/jinggai'
# anno_file = '/volume1/gddi-data/lgy/models/yolo/jinggai/gddi_valid_data.json'
def infer(model, **kwargs):
    threshold = 0.01
    nms_threshold = 0.6

    coco = COCO(anno_file)
    preds = []
    infer_num = 0
    mode = get_quant_mode()
    for img in tqdm(coco.dataset['images']):
        
        if mode == "IR":
            if infer_num > 1:
                break
        else: 
            if mode != 'infer':
                if infer_num > 500:
                    break
        infer_num = infer_num + 1

        img_id = img['id']
        if not os.path.exists(os.path.join(img_dir, img['file_name'])):
            continue

        input_path = os.path.join(img_dir, img['file_name'])
        img = cv2.imread(input_path)
        data, ratio, top, left = preprocess(img, (640, 640))


        # if mode == "IR":
        #     data = torch.from_numpy(data.astype(np.float32)).unsqueeze(0)
        #     data = torch.vstack((data,data))
        # else:
        data = torch.from_numpy(data.astype(np.float32)).unsqueeze(0)
        
        outputs = model(data, augment=False)
        
        if mode == 'infer':
            pred = [a.to('cpu').detach().numpy() for a in outputs]
            pred = non_max_suppression(pred, conf_thres=0.5, iou_thres=0.5)

            for pp in pred:
                if pp is None:
                    continue
                for p in pp:
                    bbox = p[:4]
                    bbox[0::2] -= left
                    bbox[1::2] -= top
                    bbox /= ratio
                    prob = float(p[4])
                    clse = int(p[5])
                    preds.append(dict(
                        image_id=img_id,
                        category_id=clse,
                        bbox=[float(bbox[0]), float(bbox[1]), float(bbox[2]-bbox[0]), float(bbox[3]-bbox[1])],
                        score=prob))
                        #         cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 4)
        # cv2.imwrite("/Knight/TS-Quantize/Pytorch/preds/{}".format(input_path.split('/')[-1]), img)
    if mode == 'infer':
        with open('mask_predictions_onnx2.json', 'w') as f:
            json.dump(preds, f, indent=4)
        coco_dt = coco.loadRes('mask_predictions_onnx2.json')
        cocoeval = COCOeval(coco, coco_dt, 'bbox')
        cocoeval.evaluate()
        cocoeval.accumulate()
        cocoeval.summarize()

        print('******final top1:{}'.format(cocoeval.stats[0]))
        print('******final top5:{}'.format(cocoeval.stats[1]))
        return cocoeval.stats[0], cocoeval.stats[1]
    return 0

# def yolov5(weight=None):
#     # if weight:
#     model_base_path = os.path.dirname(weight)
#     model_file = os.path.join(model_base_path, "model.yaml")
#     weight_file = os.path.join(model_base_path, 'gddi_model_weight.pth')
#     net = Model(model_file)
#     state_dict = torch.load(weight_file, map_location=torch.device('cpu'))
#     net.load_state_dict(state_dict, strict=False)

#     for m in net.modules():
#         t = type(m)
#         print(t)

#         if t in (nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU, Detect, Model):
#             m.inplace = True  # torch 1.7.0 compatibility
#             if t is Detect:
#                 if not isinstance(m.anchor_grid, list):  # new Detect Layer compatibility
#                     delattr(m, 'anchor_grid')
#                     setattr(m, 'anchor_grid', [torch.zeros(1)] * m.nl)
#         if t is Conv:
#             m._non_persistent_buffers_set = set()  # torch 1.6.0 compatibility
#         elif t is nn.Upsample and not hasattr(m, 'recompute_scale_factor'):
#             m.recompute_scale_factor = None  # torch 1.11.0 compatibility
#     return net, (1, 3, 640, 640)

class _yolov5(QuantModel):
    def __init__(self, model_name):
        super(_yolov5, self).__init__()
        self.model_name = model_name

    def model(self, weight=None):
        model = Model('/tmp/model_ts.yaml')

        if weight:
            model_base_path = os.path.dirname(weight)
            model_file = os.path.join(model_base_path, "model.yaml")
            weight_file = os.path.join(model_base_path, 'gddi_model_weight.pth')
            # model = Model(model_file)
            state_dict = torch.load(weight_file, map_location=torch.device('cpu'))
            model.load_state_dict(state_dict['model'].state_dict(), strict=False)

            w0 = torch.Tensor([1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.,
                           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.,
                           0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                           1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.,
                           0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.,
                           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.,
                           0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                           0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.])
            model.model[0].conv0.weight.data.copy_(w0.reshape(12, 3, 2, 2))
            
            for m in model.modules():
               t = type(m)

               if t in (nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU, Detect, Model):
                   m.inplace = True  # torch 1.7.0 compatibility
                   if t is Detect:
                       if not isinstance(m.anchor_grid, list):  # new Detect Layer compatibility
                           delattr(m, 'anchor_grid')
                           setattr(m, 'anchor_grid', [torch.zeros(1)] * m.nl)
               if t is Conv:
                   m._non_persistent_buffers_set = set()  # torch 1.6.0 compatibility
               elif t is nn.Upsample and not hasattr(m, 'recompute_scale_factor'):
                   m.recompute_scale_factor = None  # torch 1.11.0 compatibility
        return model

    def infer(self):
        return infer
