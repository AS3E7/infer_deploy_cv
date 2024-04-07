import sys
import os
import argparse
import ast

import torch

import onnx 
from onnxsim import simplify

import pdb

def jit_export_onnx(pt_file, onnx_file, shape, input_name, output_name, dynamic=False):
    input = torch.randn(shape)
    model = torch.jit.load(pt_file, map_location=torch.device('cpu'))
    model = model.eval().float()

    dummy_output = model(input)
    torch.onnx.export(model, input, '/tmp/tmp.onnx', 
            export_params=True, opset_version=12, do_constant_folding=False, input_names=input_name, output_names=output_name)
            #export_params=True, opset_version=12, do_constant_folding=True, input_names=input_name, output_names=output_name)

    model_onnx = onnx.load('/tmp/tmp.onnx')

    # convert model
    model_simp, check = simplify(model_onnx)

    assert check, "Simplified ONNX model could not be validated"

    onnx.save(model_simp, onnx_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='''Export onnx file from jit model, for example: 
        python3 tools/modelconverter/jit_export_onnx.py --pt resnet50.pth --onnx resnet50.onnx --shape "[1,3,224,224]" --input_name "['input']" --output_name "['out']" 
    ''')
    parser.add_argument('--pt',type = str,default=None, help='torch jit save model file path')
    parser.add_argument('--onnx',type = str,default=None, help='save onnx file path')
    parser.add_argument('--shape',type = str,default=None, help='model input shape, please set -1 to batch size if need dynamic shape, for example:[-1, 3, 640, 640]')
    parser.add_argument('--input_name',type = str,default=None, help="onnx model input name list, for example: ['input']")
    parser.add_argument('--output_name',type = str,default=None, help="onnx model output name list, for example: ['out0', 'out1', 'out2']")
    parser.add_argument('--dynamic',type = bool,default=False, help='save onnx file as dynamic shape')

    args,unknown = parser.parse_known_args()

    # TODO: 判断一下pt、onnx路径是否正确，shape格式是否对

    # TODO: 判断一下jit是否jit trace后的模型

    shape       = ast.literal_eval(args.shape)
    input_name  = ast.literal_eval(args.input_name)
    output_name = ast.literal_eval(args.output_name)
    #tmp = args.output_name

    #print(tmp)
    #output_name = tmp[1:-1].split(',')
    #print(type(output_name))
    #print(len(output_name))
    jit_export_onnx(args.pt, args.onnx, shape, input_name, output_name, args.dynamic)
