from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import json
import sys
import argparse
import os
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='make valid dataset')
    parser.add_argument('--real', default='', required=True)
    parser.add_argument('--pred', default='', required=True)
    args = parser.parse_args()

    coco = COCO(args.real)
    coco_dt = coco.loadRes(args.pred)
    # coco_dt = coco.loadRes('/root/host/tmp/mask_predictions_pt2.json')
    cocoeval = COCOeval(coco, coco_dt, 'keypoints')
    cocoeval.params.kpt_oks_sigmas = np.array([0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25])
    cocoeval.evaluate()
    cocoeval.accumulate()
    cocoeval.summarize()
