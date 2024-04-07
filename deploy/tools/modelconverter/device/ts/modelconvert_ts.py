import os
import sys
import subprocess

cmd = ['cp', '/gddi_output/model.yaml', '/tmp/model_ts.yaml']
print(cmd)
subprocess.call(cmd , env=os.environ)

cmd = ['sed', '-i', 's/nn.Upsample/Upsample/g', '/tmp/model_ts.yaml']
print(cmd)
subprocess.call(cmd , env=os.environ)

cmd = [
'Knight', '--chip', 'TX5368A', 'quant', 'pytorch',
'-w', '/gddi_output/gddi_model_weight.pth', 
'-m', '_yolov5', 
'--input-shapes', '1', '3', '640', '640',
'-qm', 'min_max', 
'-r', 'all'
]
subprocess.call(cmd , env=os.environ)

cmd = [
'Knight', '--chip', 'TX5368A', 'rne-compile',
'--net', '/TS-Knight-output/quant_pytorch/_yolov5/_yolov5_8.prototxt', 
'--weight', '/TS-Knight-output/quant_pytorch/_yolov5/_yolov5_8.weight', 
'--outpath', '/TS-Knight-output/quant_pytorch/_yolov5', 
'--opt-group', '1'
]
subprocess.call(cmd , env=os.environ)

cmd = [
'/root/combine', '/TS-Knight-output/quant_pytorch/_yolov5/_yolov5_8_r.tsm', '/TS-Knight-output/quant_pytorch/_yolov5/_yolov5_8_r.cfg', '/TS-Knight-output/quant_pytorch/_yolov5/_yolov5_8_r.weight'
]
# cmd = [
# Knight --chip TX5368A rne-profiling --config /ts/code/data/models/detect/small_helmet/_yolov5/_yolov5_8_r.cfg --weight /ts/code/data/models/detect/small_helmet/_yolov5/_yolov5_8_r.weight --outpath /ts/code/data/models/detect/small_helmet/_yolov5/
# ]
subprocess.call(cmd , env=os.environ)


cmd = ['mv', '/TS-Knight-output/quant_pytorch/_yolov5/_yolov5_8_r.tsm', '/gddi_output/gddi_model.tsm']
print(cmd)
subprocess.call(cmd , env=os.environ)