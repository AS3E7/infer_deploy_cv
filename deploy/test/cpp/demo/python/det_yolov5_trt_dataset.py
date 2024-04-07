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

import torch
import sys
sys.path.append('./yolov5')
from utils.general import scale_coords

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
    return img[:, :, ::-1].transpose([2, 0, 1])/255, ratio[0], (top, left)

def plot_one_box(x, img, color=None, label=None, line_thickness=None):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [np.random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)


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

from onnx.tools import update_model_dims
import onnx
def changeOnnx2Dynamic(onnx_file):

    model = onnx.load(onnx_file)
    
    input_shape = [-1, 3, 640, 640]
    input_dims = {}
    for input in model.graph.input:
        input_dims[input.name] = list(input_shape)
    # keep output_dims
    output_dims = {}
    for output in model.graph.output:
        dim = []
        tensor_type = output.type.tensor_type
        if tensor_type.HasField("shape"):
            for d in tensor_type.shape.dim:
                if d.HasField("dim_value"):
                    dim.append(d.dim_value)
                elif d.HasField("dim_param"):
                    dim.append(d.dim_param)
                else:
                # sys.exit("error: unknown dimension")
                    continue
            output_dims[output.name] = dim

    update_model_dims.update_inputs_outputs_dims(model, input_dims, output_dims)
    onnx.save(model, './tmp_dynamic.onnx')

    return './tmp_dynamic.onnx'

    
from models.yolo import Model

IMAGES_NUM = 100

from cuda import cudart
import tensorrt as trt
import calibrator


if __name__ == '__main__':
    """ A YOLOv5 example.
    """
    parser = argparse.ArgumentParser(description='for sail det_yolov5 py test')
    parser.add_argument('--onnx', default='', required=True)
    parser.add_argument('--trt', default='', required=True)
    parser.add_argument('--input', default='', required=False)
    parser.add_argument('--calib_imgs_path', default='', required=False)
    
    args = parser.parse_args()
    
    onnxFile = changeOnnx2Dynamic(args.onnx)
    trtFile = args.trt
    input_path = args.input
    calibrationDataPath = args.calib_imgs_path
    calibrationCount = 500
    batch = 4
    imageHeight = 640
    imageWidth = 640
    cacheFile = "./int8.cache"

    logger = trt.Logger(trt.Logger.ERROR)

    if os.path.isfile(trtFile):
        with open(trtFile, 'rb') as f:
            engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())
        if engine == None:
            print("Failed loading engine!")
            exit()
        print("Succeeded loading engine!")
    else:
        builder = trt.Builder(logger)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        profile = builder.create_optimization_profile()
        config = builder.create_builder_config()
        config.flags = 1 << int(trt.BuilderFlag.INT8)
        config.int8_calibrator = calibrator.MyCalibrator(calibrationDataPath, calibrationCount, (batch, 3, imageHeight, imageWidth), cacheFile)
        config.max_workspace_size = 3 << 30
        parser = trt.OnnxParser(network, logger)
        if not os.path.exists(onnxFile):
            print("Failed finding onnx file!")
            exit()
        print("Succeeded finding onnx file!")
        with open(onnxFile, 'rb') as model:
            if not parser.parse(model.read()):
                print("Failed parsing .onnx file!")
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                exit()
            print("Succeeded parsing .onnx file!")

        inputTensor = network.get_input(0)
        profile.set_shape(inputTensor.name, (1, 3, 640, 640),  (8, 3, 640, 640),  (16, 3, 640, 640))
        config.add_optimization_profile(profile)

        # network.unmark_output(network.get_output(0))  # 去掉输出张量 'y'
        engineString = builder.build_serialized_network(network, config)
        if engineString == None:
            print("Failed building engine!")
            exit()
        print("Succeeded building engine!")
        with open(trtFile, 'wb') as f:
            f.write(engineString)
        engine = trt.Runtime(logger).deserialize_cuda_engine(engineString)

    context = engine.create_execution_context()
    context.set_binding_shape(0, [1, 3, 640, 640])
    _, stream = cudart.cudaStreamCreate()
    print("EngineBinding0->", engine.get_binding_shape(0), engine.get_binding_dtype(0))
    print("EngineBinding1->", engine.get_binding_shape(1), engine.get_binding_dtype(1))
    print("EngineBinding2->", engine.get_binding_shape(2), engine.get_binding_dtype(2))
    print("EngineBinding3->", engine.get_binding_shape(3), engine.get_binding_dtype(3))

    device = torch.device('cuda:0')

    
    # H2D
    inputH0 = np.empty(context.get_binding_shape(0), dtype=trt.nptype(engine.get_binding_dtype(0)))
    _, inputD0 = cudart.cudaMallocAsync(inputH0.nbytes, stream)

    outputH0 = np.empty(context.get_binding_shape(1), dtype=trt.nptype(engine.get_binding_dtype(1)))
    _, outputD0 = cudart.cudaMallocAsync(outputH0.nbytes, stream)
    outputH1 = np.empty(context.get_binding_shape(2), dtype=trt.nptype(engine.get_binding_dtype(2)))
    _, outputD1 = cudart.cudaMallocAsync(outputH1.nbytes, stream)
    outputH2 = np.empty(context.get_binding_shape(3), dtype=trt.nptype(engine.get_binding_dtype(3)))
    _, outputD2 = cudart.cudaMallocAsync(outputH2.nbytes, stream)


    import os
    import json
    with open(input_path) as g:
        js = json.load(g)
    results = []

    for img in js['images']:
        input_path = img['file_name']
        input_path = os.path.join('/volume1/gddi-data/lgy/dataset/helmet/0/images/valid/', input_path)

        input = cv2.imread(input_path)
        data, ratio, (top, left) = preprocess(input)
        data = data.astype(np.float32)

        inputH0 = np.ascontiguousarray(data.reshape(-1))
        cudart.cudaMemcpyAsync(inputD0, inputH0.ctypes.data, inputH0.nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, stream)

        context.execute_async_v2([int(inputD0), int(outputD0), int(outputD1), int(outputD2)], stream)

        # D2H
        cudart.cudaMemcpyAsync(outputH0.ctypes.data, outputD0, outputH0.nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, stream)
        cudart.cudaMemcpyAsync(outputH1.ctypes.data, outputD1, outputH1.nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, stream)
        cudart.cudaMemcpyAsync(outputH2.ctypes.data, outputD2, outputH2.nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, stream)
        cudart.cudaStreamSynchronize(stream)


        pred = [outputH0, outputH1, outputH2]
        # pred = [a.detach().numpy() for a in pred[1]]
        pred = non_max_suppression(pred, conf_thres=0.5, iou_thres=0.5)

        print(pred)
        for i in pred:
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
        #         cv2.rectangle(input, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 4)
        # cv2.imwrite("./preds/{}".format(img_f_path.split('/')[-1]), input)

    with open('mask_predictions.json', 'w') as f:
      json.dump(results, f, indent=4)


    cudart.cudaStreamDestroy(stream)
    cudart.cudaFree(inputD0)
    cudart.cudaFree(outputH0)
    cudart.cudaFree(outputH1)
    cudart.cudaFree(outputH2)

    print('over')
