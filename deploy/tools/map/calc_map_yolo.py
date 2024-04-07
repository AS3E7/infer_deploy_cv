import argparse
import json
import numpy as np
import torch


def clip_coords(boxes, img_shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    boxes[:, 0].clamp_(0, img_shape[1])  # x1
    boxes[:, 1].clamp_(0, img_shape[0])  # y1
    boxes[:, 2].clamp_(0, img_shape[1])  # x2
    boxes[:, 3].clamp_(0, img_shape[0])  # y2


def ap_per_class(tp, conf, pred_cls, target_cls):
    """ Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
    # Arguments
        tp:    True positives (nparray, nx1 or nx10).
        conf:  Objectness value from 0-1 (nparray).
        pred_cls: Predicted object classes (nparray).
        target_cls: True object classes (nparray).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """

    # Sort by objectness
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes
    unique_classes = np.unique(target_cls)

    # Create Precision-Recall curve and compute AP for each class
    pr_score = 0.1  # score to evaluate P and R https://github.com/ultralytics/yolov3/issues/898
    s = [unique_classes.shape[0], tp.shape[1]]  # number class, number iou thresholds (i.e. 10 for mAP0.5...0.95)
    ap, p, r = np.zeros(s), np.zeros(s), np.zeros(s)
    for ci, c in enumerate(unique_classes):
        i = pred_cls == c
        n_gt = (target_cls == c).sum()  # Number of ground truth objects
        n_p = i.sum()  # Number of predicted objects

        if n_p == 0 or n_gt == 0:
            continue
        else:
            # Accumulate FPs and TPs
            fpc = (1 - tp[i]).cumsum(0)
            tpc = tp[i].cumsum(0)

            # Recall
            recall = tpc / (n_gt + 1e-16)  # recall curve
            r[ci] = np.interp(-pr_score, -conf[i], recall[:, 0])  # r at pr_score, negative x, xp because xp decreases

            # Precision
            precision = tpc / (tpc + fpc)  # precision curve
            p[ci] = np.interp(-pr_score, -conf[i], precision[:, 0])  # p at pr_score

            # AP from recall-precision curve
            for j in range(tp.shape[1]):
                ap[ci, j] = compute_ap(recall[:, j], precision[:, j])

    # Compute F1 score (harmonic mean of precision and recall)
    f1 = 2 * p * r / (p + r + 1e-16)

    return p, r, ap, f1, unique_classes.astype('int32')


def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rbgirshick/py-faster-rcnn.
    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """

    # Append sentinel values to beginning and end
    mrec = np.concatenate(([0.], recall, [min(recall[-1] + 1E-3, 1.)]))
    mpre = np.concatenate(([0.], precision, [0.]))

    # Compute the precision envelope
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))

    # Integrate area under curve
    method = 'interp'  # methods: 'continuous', 'interp'
    if method == 'interp':
        x = np.linspace(0, 1, 101)  # 101-point interp (COCO)
        ap = np.trapz(np.interp(x, mrec, mpre), x)  # integrate
    else:  # 'continuous'
        i = np.where(mrec[1:] != mrec[:-1])[0]  # points where x axis (recall) changes
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])  # area under curve

    return ap


def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def box_iou(box1, box2):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.T)
    area2 = box_area(box2.T)

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    return inter / (area1[:, None] + area2 - inter)  # iou = inter / (area1 + area2 - inter)


height = width = 640
whwh = torch.Tensor([640, 640, 640, 640])
iouv = torch.linspace(0.5, 0.95, 10)  # iou vector for mAP@0.5:0.95
niou = iouv.numel()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred')
    parser.add_argument('--real')
    args = parser.parse_args()

    with open(args.pred) as f:
        pred = json.load(f)
    with open(args.real) as f:
        real = json.load(f)
        id2wh = {i['id']: (i['width'], i['height']) for i in real['images']}

    id2real = {}
    new_shape = (640, 640)
    for r in real['annotations']:
        if r['image_id'] not in id2real:
            id2real[r['image_id']] = []

        width = 0
        height = 0
        for img in real['images']:
            if img['id'] == r['image_id']:
                height = img['height']
                width = img['width']
                break

        ratio = min(new_shape[0] / height, new_shape[1] / width)

        new_unpad = int(round(width * ratio)), int(round(height * ratio))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

        labels = r['bbox'].copy()
        # labels[0] = ratio * width  * (r['bbox'][0] - r['bbox'][2] / 2) + dw  # pad width
        # labels[1] = ratio * height * (r['bbox'][1] - r['bbox'][3] / 2) + dh  # pad height
        # labels[2] = ratio * width  * (r['bbox'][0] + r['bbox'][2] / 2) + dw
        # labels[3] = ratio * height * (r['bbox'][1] + r['bbox'][3] / 2) + dh

        labels[0] = int(round(dw - 0.1))/2 + r['bbox'][0] * ratio
        labels[1] = int(round(dh - 0.1))/2 + r['bbox'][1] * ratio
        labels[2] = int(round(dw - 0.1))/2 + (r['bbox'][0] + r['bbox'][2]) * ratio
        labels[3] = int(round(dh - 0.1))/2 + (r['bbox'][1] + r['bbox'][3]) * ratio
        # labels[2] = r['bbox'][2] * ratio
        # labels[3] = r['bbox'][3] * ratio

        # r['bbox'] = labels

        # r['bbox'][0] += r['bbox'][2] / 2
        # r['bbox'][1] += r['bbox'][3] / 2
        # w, h = id2wh[r['image_id']]
        # if w < h:
        #     r['bbox'][0] += (h - w) / 2
        # else:
        #     r['bbox'][1] += (w - h) / 2
        # r['bbox'] = [_ / max(w, h) for _ in r['bbox']]
        id2real[r['image_id']].append([r['category_id']] + labels)

    id2pred = {}
    for p in pred:
        if p['image_id'] not in id2pred:
            id2pred[p['image_id']] = []
        # p['bbox'][0] += p['bbox'][2] / 2
        # p['bbox'][1] += p['bbox'][3] / 2
        id2pred[p['image_id']].append([p['category_id']] + p['bbox'] + [p['score']])
    
    seen = 0
    stats = []
    nc = len(real['categories'])
    # Statistics per image
    for id in id2real:
        labels = torch.Tensor(id2real[id])
        nl = len(labels)
        tcls = labels[:, 0].tolist() if nl else []
        seen += 1

        pred = id2pred.get(id, [])
        if not pred and nl:
            stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
            continue
        
        pred = torch.Tensor(pred)
        clip_coords(pred, (640, 640))
        # clip_coords(pred, id2wh[id])

        correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool)

        if nl:
            detected = []  # target indices
            tcls_tensor = labels[:, 0]

            # target boxes
            # tbox = xywh2xyxy(labels[:, 1:5])
            tbox = labels[:, 1:5]
            # pbox = xywh2xyxy(pred[:, 1:5])
            pbox = pred[:, 1:5]
            # import pdb; pdb.set_trace()
            # Per target class
            for cls in torch.unique(tcls_tensor):
                ti = (cls == tcls_tensor).nonzero(as_tuple=False).view(-1)  # prediction indices
                pi = (cls == pred[:, 0]).nonzero(as_tuple=False).view(-1)  # target indices

                # Search for detections
                if pi.shape[0]:
                    # Prediction to target ious
                    ious, i = box_iou(pbox[pi], tbox[ti]).max(1)  # best ious, indices

                    # Append detections
                    detected_set = set()
                    for j in (ious > iouv[0]).nonzero(as_tuple=False):
                        d = ti[i[j]]  # detected target
                        if d.item() not in detected_set:
                            detected_set.add(d.item())
                            detected.append(d)
                            correct[pi[j]] = ious[j] > iouv  # iou_thres is 1xn
                            if len(detected) == nl:  # all targets already located in image
                                break

        # Append statistics (correct, conf, pcls, tcls)
        stats.append((correct.cpu(), pred[:, 5].cpu(), pred[:, 0].cpu(), tcls))
    stats = [np.concatenate(x, 0) for x in zip(*stats)]
    # import pdb; pdb.set_trace()
    if len(stats) and stats[0].any():
        p, r, ap, f1, ap_class = ap_per_class(*stats)
        p, r, ap50, ap = p[:, 0], r[:, 0], ap[:, 0], ap.mean(
            1)  # [P, R, AP@0.5, AP@0.5:0.95]
        mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
        nt = np.bincount(stats[3].astype(np.int64),
                         minlength=nc)  # number of targets per class

        # Print results
        pf = '%20s' + '%12.6g' * 6  # print format
        print(pf % ('all', seen, nt.sum(), mp, mr, map50, map))
