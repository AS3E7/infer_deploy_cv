import os
import sys
import time
import yaml
import json
import requests
import subprocess


def popen_process(command):
    try:
        p = subprocess.Popen(command,
                             shell=True,
                             stdout=sys.stdout,
                             stderr=subprocess.PIPE,
                             close_fds=True,
                             universal_newlines=True)
    except subprocess.CalledProcessError as e:
        raise ValueError('gddi command: {} exec error {}'.format(command,str(e)))
    res = p.communicate()
    sys.stdout.flush()
    if p.poll() != 0:
        raise RuntimeError('subprocess task {} exec failed: {}'.format(command, str(res[-1][:])))


class Yolov5SubnetSelectTask(object):
    def __init__(self, task_id, url, args):
        self.task = 'toposearchv2'
        self.task_id = task_id
        self.url = url
        self.args = args
        self.fps = None
        self.max_flops = 25

    def prepare(self):
        os.makedirs('/gddi_log/tensorboard/', exist_ok=True)
        # get max flops
        with open('/gddi_config/config.json') as f:
            cfg = json.load(f)
        with open('plat/device_capacity.json') as f:
            dev = json.load(f)
        env = cfg['args']['application']['deployment_env']
        fps = cfg['args']['search']['fps']
        for i, _ in enumerate(dev):
            if _['value'] == env[0]:
                break
        else:
            raise ValueError('No deployment_env {} found'.format(env[0]))
        dev = dev[i]['children']

        for i, _ in enumerate(dev):
            if _['value'] == env[1]:
                break
        else:
            raise ValueError('No deployment_env {} {} found'.format(*env[:2]))
        dev = dev[i]['children']

        for i, _ in enumerate(dev):
            if _['value'] == env[2]:
                break
        else:
            raise ValueError('No deployment_env {} {} {} found'.format(*env))
        dev = dev[i]['attachment']
        try:
            self.max_flops = dev['detection']['yolo'][str(fps)] / 1000
        except:
            self.max_flops = dev['detection']['yolo'][str(20)] / 1000

        # coco data to yolo data
        try:
            os.makedirs('/gddi_data/0/labels/valid', exist_ok=True)
            valid_names = self.cocotoyolo('/gddi_data/0/annotation/valid/anno.json')
            os.makedirs('/gddi_data/0/labels/train', exist_ok=True)
            train_names = self.cocotoyolo('/gddi_data/0/annotation/train/anno.json')
            assert train_names == valid_names
            with open('/gddi_data/data.yaml', 'w') as f:
                data = dict(
                    train='/gddi_data/0/images/train',
                    val='/gddi_data/0/images/valid',
                    nc=len(train_names),
                    names=train_names
                )
                yaml.dump(data, f, sort_keys=False)
        except:
            os.makedirs('/gddi_data/0/labels/train', exist_ok=True)
            train_names = self.cocotoyolotxt('/gddi_output/gddi_train_data.json')
            valid_names = self.cocotoyolotxt('/gddi_output/gddi_valid_data.json')
            assert train_names == valid_names
            with open('/gddi_data/data.yaml', 'w') as f:
                data = dict(
                    train='/gddi_data/0/labels/_train.txt',
                    val='/gddi_data/0/labels/_valid.txt',
                    nc=len(train_names),
                    names=train_names
                )
                yaml.dump(data, f, sort_keys=False)
        result = {
            "status": True,
            "callback_type": "notice",
            "remove_action": False,
            "result": {"progress": 0.10}
        }
        self.callback(**result)

    def exec(self):
        max_flops = self.max_flops
        min_flops = min(max_flops - 2, 22)
        cmd = 'python fast_finetune_onebyone.py --data {} --stop-epoch {}'
        cmd = cmd + ' --max-flops {} --min-flops {}'
        cmd = cmd + ' --logdir {}'
        cmd = cmd + ' --callback 40 70 98'
        cmd = cmd + ' --task-id {} --task-url {} --task-type {}'
        # cmd = cmd + ' --img-size 320 --batch-size 16 --epochs 1'
        cmd = cmd.format('/gddi_data/data.yaml', 5,
                         max_flops, min_flops,
                         '/gddi_log', self.task_id, self.url, self.task)
        popen_process(cmd)
        result = {
            "status": True,
            "callback_type": "notice",
            "remove_action": False,
            "result": {"progress": 0.99}
        }
        self.callback(**result)
        
    def clear(self):
        from shutil import rmtree
        try:
            rmtree('/gddi_data/0/labels')
            os.remove('/gddi_data/data.yaml')
        except:
            pass

    def callback(self, status, result, remove_action=True, callback_type="result"):
        result = {
            'id': int(self.task_id),
            'status': status,
            "callback_type": callback_type,
            "task_type": self.task,
            "remove_action": remove_action,
            "result": result}
        result = json.dumps(result, indent=4)
        for i in range(3):
            try:
                r = requests.post(self.url, data=result)
                print('http response', r.content)
                break
            except:
                time.sleep(5)
        else:
            print('call back fail: {}'.format(str(result)))

    def cocotoyolo(self, json_path):
        print("load", json_path)
        with open(json_path) as f:
            data = json.load(f)

        print("preprocess", json_path)
        names = {_['id']: (i, _['name']) for i, _ in enumerate(data['categories'])}

        images = {}
        for img in data['images']:
            images[img['id']] = img 
            images[img['id']]['anns'] = []

        for ann in data['annotations']:
            img_id = ann['image_id']
            if img_id not in images:
                continue
            iw = images[img_id]['width']
            ih = images[img_id]['height']

            x1, y1, w, h = ann['bbox']
            x2, y2 = x1 + w, y1 + h 
            x1 = max(x1, 0)
            y1 = max(y1, 0)
            x2 = min(x2, iw) 
            y2 = min(y2, ih) 
            c = names[ann['category_id']][0]
            images[img_id]['anns'].append([c, (x1+x2)/2/iw, (y1+y2)/2/ih, (x2-x1)/iw, (y2-y1)/ih])

        names = [names[_][1] for _ in names]

        print("Convert", json_path)
        if 'train' in json_path:
            prefix = '/gddi_data/0/labels/train'
        else:
            prefix = '/gddi_data/0/labels/valid'
        for k, v in images.items():
            ann_path = '.'.join(v['file_name'].split('.')[:-1])+'.txt'
            with open(os.path.join(prefix, ann_path), 'w') as f:
                for ann in v['anns']:
                    f.write(' '.join([str(_) for _ in ann])+'\n')
        return names

    def cocotoyolotxt(self, json_path):
        print("load", json_path)
        with open(json_path) as f:
            data = json.load(f)

        print("preprocess", json_path)
        names = {_['id']: (i, _['name']) for i, _ in enumerate(data['categories'])}

        images = {}
        for img in data['images']:
            images[img['id']] = img 
            images[img['id']]['anns'] = []

        for ann in data['annotations']:
            img_id = ann['image_id']
            if img_id not in images:
                continue
            iw = images[img_id]['width']
            ih = images[img_id]['height']

            x1, y1, w, h = ann['bbox']
            x2, y2 = x1 + w, y1 + h 
            x1 = max(x1, 0)
            y1 = max(y1, 0)
            x2 = min(x2, iw) 
            y2 = min(y2, ih) 
            c = names[ann['category_id']][0]
            images[img_id]['anns'].append([c, (x1+x2)/2/iw, (y1+y2)/2/ih, (x2-x1)/iw, (y2-y1)/ih])

        names = [names[_][1] for _ in names]

        print("Convert", json_path)
        prefix = '/gddi_data/0/labels/train'
        for k, v in images.items():
            ann_path = '.'.join(v['file_name'].split('.')[:-1])+'.txt'
            with open(os.path.join(prefix, ann_path), 'w') as f:
                for ann in v['anns']:
                    f.write(' '.join([str(_) for _ in ann])+'\n')
        prefix = '/gddi_data/0/images/train/'
        if 'train' in json_path:
            with open('/gddi_data/0/labels/_train.txt', 'w') as f:
                for k, v in images.items():
                    f.write(prefix+v['file_name']+'\n')
        else:
            with open('/gddi_data/0/labels/_valid.txt', 'w') as f:
                for k, v in images.items():
                    f.write(prefix+v['file_name']+'\n')
        return names
