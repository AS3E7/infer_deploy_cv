import os
import json
import argparse
import threading
import subprocess

from agent import *

def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('--config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    # parser.add_argument(
    #     '--amp',
    #     action='store_true',
    #     default=False,
    #     help='enable automatic-mixed-precision training')
    # parser.add_argument(
    #     '--resume',
    #     nargs='?',
    #     type=str,
    #     const='auto',
    #     help='If specify checkpoint path, resume from it, while if not '
    #     'specify, try to auto resume from the latest checkpoint '
    #     'in the work directory.')
    # parser.add_argument(
    #     '--cfg-options',
    #     nargs='+',
    #     action=DictAction,
    #     help='override some settings in the used config, the key-value pair '
    #     'in xxx=yyy format will be merged into config file. If the value to '
    #     'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
    #     'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
    #     'Note that the quotation marks are necessary and that no white space '
    #     'is allowed.')
    # parser.add_argument(
    #     '--launcher',
    #     choices=['none', 'pytorch', 'slurm', 'mpi'],
    #     default='none',
    #     help='job launcher')
    # parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()

    return args

'''
"cambricon_mlu220_aarch64": {
            "product": "cambricon",
            "chip": "mlu220",
            "arch": "aarch64",
            "communication": {
                "type": "ssh",  //ssh、serial、adb
                "ip": "10.11.0.202",
                "username": "root",
                "password": "gdd"
            }, 
            "model_suffix": ".cambricon",
            "device_env": "LD_LIBRARY_PATH=",
            "speed_pre_cmd": "",    //需要预先执行的命令
            "speed_cmd": "",
            "docker_image": "",
            "docker_contrainer": "",
            "docker_contrainer_cmd": ""
        }
'''


# 加密模型
def modcrypt_model(model, properties, sn, model_cry):
    cmd = [
        '/data/tools/modcrypt/gencryptor',
        '-t', 'offline', '-c', properties,
        '-m', model,
        '-sn', sn,
        '-o', "/tmp"
    ]
    subprocess.call(cmd, env=os.environ)

    cmd = [
        'mv', '/tmp/model.gem', model_cry,
    ]
    subprocess.call(cmd, env=os.environ)

def coco_map(anno_path, result_path):
    cmd = [
        'python3', 
        '/data/tools/map/coco_map.py',
        '--real', anno_path, 
        '--pred', result_path
    ]
    subprocess.call(cmd, env=os.environ)

# 获取测速命令行
def device_cmd_speed(dev_info, sdk_info, model_path):
    cmd_str = []
    # 3. 拼接各个命令
    dev_speed_cmd = [dev_info['device_env'],  
    dev_info['speed_pre_cmd'], 
    ' && ', 'SDK_PATH='+sdk_info['sdk_path'], sdk_info['sdk_env'],
    dev_info['speed_cmd'], 
    model_path, 
    dev_info['speed_cmd_args'] 
    ]
    return ' '.join(dev_speed_cmd)

def device_cmd_docker(dev_info):
    docker_contrainer_cmd = ''

    # 1. 判断镜像是否有，是否需要启动容器，容器需要挂载什么目录
    if dev_info['docker_image'] != '' and dev_info['docker_contrainer'] != '':
        test_docker_images_cmd = 'docker images | grep {}'.format(dev_info['docker_image']) 
        print(test_docker_images_cmd)
        # agent.run(test_docker_images_cmd)

        #docker run -it --name gddeploy_ubuntu20_v0.1 --network=host --privileged=true -v /volume1/gddi-data/:/volume1/gddi-data/ -v /data/gddeploy/:/gddeploy -v /system/:/system arm64v8/ubuntu:20.04 /bin/bash
        docker_contrainer_cmd = dev_info['docker_contrainer_cmd']

    return docker_contrainer_cmd

# 获取测map命令行
def device_cmd_map(sdk_info, dataset, model_path, model_properties, work_dir):
    # dev_info, model_path, dataset_anno_path, dataset_path, result_save_path, result_pic_path

    # 加密模型， 先获取SN号，再加密
    model_cry = model_path+'.gem'
    modcrypt_model(model_path, model_properties, '05ca7685-8c58-a285-366f-64d051310356', model_cry)

    dataset_cmd = [os.path.join(sdk_info['sdk_path'], 'bin', sdk_info['sdk_cmd_dataset']),
         '--model ', model_cry,
         '--anno-file', dataset['anno_file'],
         '--pic-path', dataset['pic_path'],
         '--result-path', os.path.join(work_dir, 'mask_predictions.json'),
         '--save-pic', os.path.join(work_dir, 'save_pic')
    ]

    dev_map_cmd = ['SDK_PATH='+sdk_info['sdk_path'],
        sdk_info['sdk_env'],
        ' '.join(dataset_cmd)
    ]

    return ' '.join(dev_map_cmd)

# 解耦：设备，任务，模型，通信方式
class taskThread (threading.Thread):
    def __init__(self, thread_id, name, task_type, dev_info, config, work_dir, log_path):
        threading.Thread.__init__(self)
        self.thread_id = thread_id
        self.name = name
        self.work_dir = work_dir
        self.log_path = log_path

        self.config = config
        self.task_type = task_type
        self.dev_info = dev_info
        self.sdk_info = config['sdk']
        self.model = config['models']
        
    def run(self, model_path = '', model_properties = ''):
        for model in self.model:     #不同的任务类型不能混用多线程，比如测速任务不能有别的进程在跑
            task_cmd = ''

            if model_path == '':
                model_path = model['model_path']
            
            if model_properties == '':
                model_properties = model['model_properties']
             
            # 和硬件通信方式
            agent = get_agent(self.dev_info['communication'])

            # TODO: 扩展更多任务
            if self.task_type == 'speed':
                task_cmd = device_cmd_speed(self.dev_info, self.sdk_info, model_path)
            elif self.task_type == 'map':
                task_cmd = device_cmd_map(self.sdk_info, model['dataset'], model_path, model_properties, self.work_dir)
            elif self.task_type == 'video':
                pass
            elif self.task_type == 'test':  # 运行gddeploy test
                pass

            if self.dev_info['docker_image'] != '' and self.dev_info['docker_contrainer'] != '':
                docker_contrainer_cmd = device_cmd_docker(self.dev_info)
                task_cmd = docker_contrainer_cmd + '\"' + task_cmd + '\"'

            # 4. 执行操作，TODO:记录到log文件
            print(task_cmd)
            stdout = agent.run(task_cmd)

            for i in stdout.readlines():
                print(i)

            if self.task_type == 'map':
                coco_map(model['dataset']['anno_file'], os.path.join(self.work_dir, 'mask_predictions.json'))

def main():
    args = parse_args()

    with open(args.config) as f:
        config = json.load(f)

    # 1. 修改配置文件
    # 1.1 拷贝需要测试的文件到临时目标目录
    # 1.2 修改配置文件
    # 1.2.1 修改batch
    # 1.2.2 修改分辨率

    # 2. docker转换模型和保存模型

    # 3. 发送到盒子进行测试命令和获取结果返回，开启多线程，每个线程执行一个硬件操作     
    work_dir = args.work_dir
    tasks = config['tasks']

    for task in tasks['task_type']:     #不同的任务类型不能混用多线程，比如测速任务不能有别的进程在跑
        
        task_threads = []

        dev_infos = {}
        if isinstance(tasks['target_device'], str) and tasks['target_device'] == 'all':
            dev_infos = config['dev_infos']
        elif isinstance(tasks['target_device'], list):
            for dev in tasks['target_device']:
                if dev in config['dev_infos'].keys():
                    dev_infos[dev] = config['dev_infos'][dev]

        for dev, dev_info in dev_infos.items():
            log_path = 'benchmark_{}_{}.log'.format(task, dev)

            thread = taskThread(1, "Thread-{}".format(dev), task, dev_info, config, work_dir, log_path)
            thread.start()

            task_threads.append(thread)

            # TODO:获取硬件资源

        # 等待所有线程完成
        for t in task_threads:
            t.join()


    # 4. 解析结果和输出报表，记得容错


if __name__ == '__main__':
    main()
