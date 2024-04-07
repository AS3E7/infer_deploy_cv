import argparse
import json
import glob
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from tqdm import tqdm

from models.experimental import attempt_load
from utils.datasets import create_dataloader
from utils.general import (
    check_dataset, check_file, check_img_size, non_max_suppression,
    clip_coords, xywh2xyxy, box_iou, ap_per_class, set_logging,
    scale_coords, xyxy2xywh, coco80_to_coco91_class)
from utils.torch_utils import select_device, time_synchronized
from flops_counter import get_model_complexity_info
from utils.dynamic_general import bn_calibration_init

def test(model,
         dataloader,
         conf_thres=0.001,
         iou_thres=0.65,
         augment=False,
         merge=False,
         half=False,
         use_coco_api=False):
    # Configure
    iouv = torch.linspace(0.5, 0.95, 10).to(device)
    niou = iouv.numel()

    model.eval()

    seen = 0
    s = ('%12s' * 4) % ('P', 'R', 'mAP@.5', 'mAP@.5:.95')
    coco91class = coco80_to_coco91_class()
    p, r, f1, mp, mr, map50, map, t0, t1 = 0., 0., 0., 0., 0., 0., 0., 0., 0.
    loss = torch.zeros(3, device=device)
    jdict, stats, ap, ap_class = [], [], [], []

    for batch_i, (img, targets, paths, shapes) in enumerate(tqdm(dataloader, desc=s)):
        img = img.to(device, non_blocking=True)
        img = img.half() if half else img.float()
        img /= 255.0
        targets = targets.to(device)
        nb, _, height, width = img.shape
        whwh = torch.Tensor([width, height, width, height]).to(device)

        # Disable gradients
        with torch.no_grad():
            inf_out, train_out = model(img, augment=augment)

            # Run NMS
            output = non_max_suppression(inf_out,
                                         conf_thres=conf_thres, 
                                         iou_thres=iou_thres,
                                         merge=merge)

        # Statistics per image
        for si, pred in enumerate(output):
            labels = targets[targets[:, 0] == si, 1:]
            nl = len(labels)
            tcls = labels[:, 0].tolist() if nl else []  # target class
            seen += 1

            if pred is None:
                if nl:
                    stats.append((torch.zeros(0, niou, dtype=torch.bool),
                                 torch.Tensor(), 
                                 torch.Tensor(),
                                 tcls))
                continue

            # Clip boxes to image bounds
            clip_coords(pred, (height, width))

            # if COCO
            if use_coco_api:
                image_id = Path(paths[si]).stem
                box = pred[:, :4].clone()  # xyxy
                scale_coords(img[si].shape[1:], box, shapes[si][0], shapes[si][1])
                box = xyxy2xywh(box)
                box[:, :2] -= box[:, 2:] / 2
                for p, b in zip(pred.tolist(), box.tolist()):
                    jdict.append({'image_id': int(image_id) \
                        if image_id.isnumeric() else image_id,
                                  'category_id': coco91class[int(p[5])],
                                  'bbox': [round(x, 3) for x in b],
                                  'score': round(p[4], 5)})               

            # Assign all predictions as incorrect
            correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool, device=device)
            if nl:
                detected = []
                tcls_tensor = labels[:, 0]

                # target boxes
                tbox = xywh2xyxy(labels[:, 1:5]) * whwh

                # Per target class
                for cls in torch.unique(tcls_tensor):
                    ti = (cls == tcls_tensor).nonzero(as_tuple=False).view(-1)
                    pi = (cls == pred[:, 5]).nonzero(as_tuple=False).view(-1)

                    # Search for detections
                    if pi.shape[0]:
                        # Prediction to target ious
                        ious, i = box_iou(pred[pi, :4], tbox[ti]).max(1)

                        # Append detections
                        detected_set = set()
                        for j in (ious > iouv[0]).nonzero(as_tuple=False):
                            d = ti[i[j]]  # detected target
                            if d.item() not in detected_set:
                                detected_set.add(d.item())
                                detected.append(d)
                                correct[pi[j]] = ious[j] > iouv
                                if len(detected) == nl:
                                    break

            # Append statistics (correct, conf, pcls, tcls)
            stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls))


    # Compute statistics
    stats = [np.concatenate(x, 0) for x in zip(*stats)]
    if len(stats) and stats[0].any():
        p, r, ap, f1, ap_class = ap_per_class(*stats)
        p, r, ap50, ap = p[:, 0], r[:, 0], ap[:, 0], ap.mean(1)
        mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()

    # Print results
    pf = '%12.3g' * 4  # print format
    print(pf % (mp, mr, map50, map))

    # COCO API
    if use_coco_api:
        f = 'coco_detections.json' 
        print('\nCOCO mAP with pycocotools... saving %s...' % f)
        with open(f, 'w') as file:
            json.dump(jdict, file)

        try:
            from pycocotools.coco import COCO
            from pycocotools.cocoeval import COCOeval

            imgIds = [int(Path(x).stem) for x in dataloader.dataset.img_files]
            cocoGt = COCO(glob.glob('./datasets/coco/annotations/instances_val*.json')[0])
            cocoDt = cocoGt.loadRes(f)
            cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')
            cocoEval.params.imgIds = imgIds
            cocoEval.evaluate()
            cocoEval.accumulate()
            cocoEval.summarize()
            map, map50 = cocoEval.stats[:2]
        except Exception as e:
            print('ERROR: pycocotools unable to run: %s' % e)
       

def calibrate_bn(model,
                 calib_loader,
                 iters):
    """Function to calibrate BN stats of a given model.
    """
    model.train()
    model.apply(lambda m: bn_calibration_init(m))
    cnt = 0
    while True:
        pbar = enumerate(calib_loader)
        num_batches = len(calib_loader)
        pbar = tqdm(pbar, total=num_batches)
        for i, (imgs, targets, paths, _) in pbar: 
            imgs = imgs.to(device, non_blocking=True).float() / 255.0
            model(imgs)
            cnt += 1
            if cnt > iters:
                break
        if cnt > iters:
            break
    return 

def reset_bn_params(model,
                    momentum,
                    eps):
    """Reset bn momentum and eps
    """
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.momentum = momentum
            m.eps = eps
    return
     

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='eval.py')
    parser.add_argument('--weights', 
                        type=str,
                        required=True,
                        help='Weights to be loaded')
    parser.add_argument('--data',
                        type=str, 
                        default='data/voc.yaml',
                        help='Data config path')
    parser.add_argument('--hyp',
                        type=str,
                        default='data/hyp.scratch.yaml',
                        help='Hyperparameter config path')
    parser.add_argument('--calib-iters',
                        type=int,
                        default=200,
                        help='Iterations for BN calibration')
    parser.add_argument('--width-mode',
                        type=str,
                        default='min',
                        help='Width mode in subnet sampling')
    parser.add_argument('--depth-mode',
                        type=str,
                        default='min',
                        help='Depth mode in subnet sampling')
    parser.add_argument('--batch-size',
                        type=int, 
                        default=64, 
                        help='Test batch size')
    parser.add_argument('--img-size',
                        type=int, 
                        default=512, 
                        help='Test image size')
    parser.add_argument('--conf-thres', 
                        type=float, 
                        default=0.001, 
                        help='Score threshold')
    parser.add_argument('--iou-thres', 
                        type=float, 
                        default=0.65, 
                        help='IOU threshold for NMS')
    parser.add_argument('--device', 
                        type=str,
                        default='0', 
                        help='CUDA device ID')
    parser.add_argument('--augment',
                        action='store_true',
                        help='Whether use TTA')
    parser.add_argument('--merge',
                        action='store_true',
                        help='Whether use merge NMS')
    parser.add_argument('--half',
                        action='store_true',
                        help='Whether use float16 test')
    parser.add_argument('--single-cls',
                        action='store_true',
                        help='Deprecate arg')
    parser.add_argument('--reset-bn-params',
                        action='store_true',
                        help='Whether reset BN params')
    parser.add_argument('--bn-momentum',
                        type=float,
                        default=0.1,
                        help='BN momentum')
    parser.add_argument('--bn-eps',
                        type=float,
                        default=1e-5,
                        help='BN eps')
    parser.add_argument('--use-coco-api',
                        action='store_true',
                        help='Whether use COCO evaluation API')
    opt = parser.parse_args()

#    print(opt)
    opt.data = check_file(opt.data)
    set_logging()
    with open(opt.hyp) as f:
        hyp = yaml.load(f, Loader=yaml.FullLoader)

    # load model
    device = select_device(opt.device, batch_size=opt.batch_size)
    model = attempt_load(opt.weights, map_location=device)
    if opt.half:
        model.half()
    
    # get dataloader
    with open(opt.data) as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
    check_dataset(data)
    img_size = check_img_size(opt.img_size, s=model.stride.max())
    train_path = data['train']
    val_path = data['val']
   
    calib_loader = create_dataloader(train_path,
                                     img_size,
                                     opt.batch_size,
                                     model.stride.max(),
                                     opt,
                                     hyp=hyp,
                                     augment=True,
                                     cache=False,
                                     rect=False)[0]

    val_loader = create_dataloader(val_path,
                                   img_size,
                                   opt.batch_size,
                                   model.stride.max(),
                                   opt,
                                   hyp=[],
                                   augment=False,
                                   cache=False,
                                   pad=0.5,
                                   rect=True)[0]
      
    if opt.reset_bn_params:
        reset_bn_params(model, opt.bn_momentum, opt.bn_eps)

    # calibrate BN
    width = model.sample_width(mode=opt.width_mode)   
    depth = model.sample_depth(mode=opt.depth_mode)

    model.set_active_subnet(width=width, depth=depth)

    if opt.calib_iters > 0:
        calibrate_bn(model, calib_loader, opt.calib_iters) 

    # make test 
    test(model,
         val_loader,
         conf_thres=opt.conf_thres,
         iou_thres=opt.iou_thres,
         augment=opt.augment,
         merge=opt.merge,
         half=opt.half,
         use_coco_api=opt.use_coco_api)

    # subnet FLOPs
    macs, params = get_model_complexity_info(model, 
        (3, opt.img_size, opt.img_size),
        print_per_layer_stat=False,
        as_strings=False)
    print('FLOPs = {:.1f}G'.format(macs * 2/ 1e9)) 
