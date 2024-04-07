import os
import os.path as osp
import argparse
import subprocess
import requests
import json
import time
import sys
import glob

import torch

INPUT_DIR = '/gddi_input_dir'
OUTPUT_DIR = '/gddi_output'
MODEL_WEIGHTS = osp.join(OUTPUT_DIR,'gddi_model_weight.pth')
PTH_WEIGHTS = glob.glob('/gddi_output/*_jit.pth')
TENSORRT_WEIGHTS = osp.join(OUTPUT_DIR,'gddi_model.trt')
TRNSORTFLITE_WEIGHTS = osp.join(OUTPUT_DIR,'gddi_model.tflite')
MNN_WEIGHTS = osp.join(OUTPUT_DIR,'gddi_model.mnn')

def popen_process(command):
    try:
        p = subprocess.Popen(command, shell=True)
    except subprocess.CalledProcessError as e:
        raise ValueError('gddi command: {} exec error {}'.format(command,str(e)))

    stdout, stderr = p.communicate()
    if p.returncode != 0:
        raise RuntimeError('gddi task # {} # exec error'.format(command))

def parse_properties(pro_file):
    result = {}
    with open(pro_file, 'r', encoding='UTF-8') as f:
        for line in f.readlines():
            line = line.strip()
            if not len(line):
                continue
            result[line.split('=')[0]] = line.split('=')[1]
    return result

def change_shape(model_type, shape):
    if model_type == 'detection' or model_type == 'pose':
        shape = (512, 512)
    return shape

def launch(is_quantization):
    property = parse_properties("/gddi_output/gddi_model.properties")
    size = property['input_size']
    model_type = property['model_type']

    if '(' in size or ')' in size:
        size = [int(i) for i in (property['input_size'][1:-1].split(','))]
    else:
        size = (int(size), int(size))

    pth_file_path = '/gddi_output/gddi_model_jit.pth'
    if len(PTH_WEIGHTS) == 0:
        print("Can't find pt file")
        return 
    if not osp.exists(PTH_WEIGHTS[0]):
        print('gddi model convert error: onnx model {} convert fail'.format(PTH_WEIGHTS[0]))
        sys.stdout.flush()
        return

    try:
        cmd = 'python3 /root/modelconvert_ts.py'
        os.system(cmd)

    except Exception as err:
        print(str(err))
        sys.exit(1)

    sys.stdout.flush()

def callback(url,result):
    try:
        r = requests.post(url, data=result)
    except:
        time.sleep(3)
        try:
            r = requests.post(url, data=result)
        except:
            time.sleep(3)
            try:
                r = requests.post(url, data=result)
            except:
                print('call back fail: {}'.format(str(result)))
                return 
    print('http response',r.content)

def is_quantization():
    with os.open("/gddi_output/gddi_model.properties") as f:
        for line in f.readlines():
            line = line.strip()
            if not len(line):
                continue
            result[line.split('=')[0]] = line.split('=')[1]
    if result['net_type'] == 'yolo':
        return 0
    else:
        return 1

if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser(description='hey launch')
        parser.add_argument('--task_type',type = str,default='train', help='task type')
        parser.add_argument('--task_id',type = str,default=None, help='task id')
        parser.add_argument('--url',type = str,default=None, help='task url')
        parser.add_argument('--sub_task_state',type = int,default=None, help='sub task state')
        #parser.add_argument('--is_quantization',help='is quantization', required=False)
        args,unknown = parser.parse_known_args()
        # result = {'id':int(args.task_id),'status':True,"callback_type":"notice","task_type": args.task_type,"remove_action":False,"result":{"progress": 0.05}}
        # result = json.dumps(result,indent=4)
        # callback(args.url,result)

        launch(1)

        result = {'id':int(args.task_id),'status':True,"callback_type":"notice","task_type": args.task_type,"remove_action":False,"result":{"progress": 0.95}}
        result = json.dumps(result,indent=4)
        callback(args.url,result)

        result = {'id':int(args.task_id),'status':True,"callback_type":"result","task_type":args.task_type,"remove_action":True,"message":"The task {} complete".format(args.task_type)}
        result = json.dumps(result,indent=4)
        r = requests.post(args.url, data=result)

        print('http response',r.content)

    except Exception as err:
        print('gddi model convert error :{}'.format(str(err)))
        result = {'id':int(args.task_id),'status':True,"callback_type":"result","task_type":args.task_type,"remove_action":False,"message":"The task {} failed".format(args.task_type)}
        result = json.dumps(result,indent=4)
        r = requests.post(args.url, data=result)
        print('http response',r.content)
        os._exit(1)

