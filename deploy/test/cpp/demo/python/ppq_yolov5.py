from ppq.api import export_ppq_graph, quantize_torch_model
from ppq.api import quantize_onnx_model
from ppq import TargetPlatform, QuantizationSettingFactory
from ppq.quantization.analyise.layerwise import layerwise_error_analyse, parameter_analyse

import torch
from torch.utils.data import DataLoader

import numpy as np
import cv2
from glob import glob

DEVICE = 'cuda'
BATCHSIZE = 8
INPUT_SHAPE = [3, 640, 640]

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
    return img[:, :, ::-1].transpose([2, 0, 1])/255


imageList = glob('/volume1/gddi-data/lgy/dataset/helmet/0/images/train/' + "*.jpg")[:1000]
def load_calibration_dataset():
    # res = [torch.rand(size=INPUT_SHAPE) for _ in range(32)]
    # return res
    
    subImageList = np.random.choice(imageList, 16, replace=False)
    # res = [torch.zeros(size=INPUT_SHAPE) for _ in range(16)]
    res = []
    for i in range(16):
        data = preprocess(cv2.imread(subImageList[i]).astype(np.float32))
        res.append(torch.from_numpy(data.copy()))
    return res

    # res = []
    # for i in range(16):
    #     data = cv2.imread(subImageList[i]).astype(np.float32)
    #     data, _, (_, _) = preprocess(data)
    #     res.append(data.deepcopy())
    # return res

calibration_dataset = load_calibration_dataset()
calibration_dataloader = DataLoader(
    dataset=calibration_dataset, 
    batch_size=BATCHSIZE, shuffle=True)

quant_setting = QuantizationSettingFactory.default_setting()
# quant_setting.equalization = True # use layerwise equalization algorithm.
# quant_setting.dispatcher   = 'conservative' # dispatch this network in conservertive way.
# quant_setting.blockwise_reconstruction              = True    # turn on pass
# quant_setting.lsq_optimization_setting.lr   = 1e-4    # adjust learning rate
# quant_setting.lsq_optimization_setting.mode = 'local' # mode of lsq optimization
# quant_setting.advanced_optimization = True
# quant_setting.advanced_optimization_setting.auto_check = True

def collate_fn(batch: torch.Tensor):
    return batch.to(DEVICE)


model = './models/gddi_model.onnx'
quantized = quantize_onnx_model(
    onnx_import_file=model, calib_dataloader=calibration_dataloader,
    calib_steps=32, input_shape=(1, 3, 640, 640),
    setting=quant_setting, collate_fn=collate_fn,
    platform=TargetPlatform.TRT_INT8,
    device=DEVICE, verbose=0)


# import sys
# sys.path.append('./yolov5')
# from models.experimental import attempt_download, attempt_load
# model = './models/gddi_model_weight.pth'
# model = torch.load(model, map_location='cuda')
# model = model['model'].to('cuda')
# quantized = quantize_torch_model(
#     model=model, calib_dataloader=calibration_dataloader,
#     calib_steps=32, input_shape=(1, 3, 640, 640), 
#     setting=quant_setting, collate_fn=collate_fn, platform=TargetPlatform.TRT_INT8,
#     onnx_export_file='./models/gddi_model_quantized.onnx', device=DEVICE, verbose=0)

# export quantized graph with another line:
export_ppq_graph(
    graph=quantized, platform=TargetPlatform.TRT_INT8,
    graph_save_to='./models/gddi_model_quantized.trt',
    config_save_to='./models/gddi_model_quantized.json')

# reports = layerwise_error_analyse(
#     graph=quantized, running_device=DEVICE, collate_fn=collate_fn,
#     dataloader=calibration_dataloader, interested_outputs=['output', '430', '432'])

# # WITH PPQ 0.6 or newer, you can invoke parameter_analyse to get a more detailed report.
# parameter_analyse(graph=quantized)
