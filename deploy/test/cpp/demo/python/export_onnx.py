import torch
import sys

sys.path.append('./yolov5')

input = torch.randn(1,3,640,640)
model = torch.load('./models/gddi_model_helmet.pth', map_location=torch.device('cpu'))['model']
model = model.eval().float()

torch.onnx.export(model, input, "gddi_model.onnx", 
        export_params=True, opset_version=12, do_constant_folding=True, input_names=['data'], output_names=['out0','out1','out2'])