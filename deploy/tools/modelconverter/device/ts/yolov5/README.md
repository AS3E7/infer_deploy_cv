# Once-for-All YOLOV5 Networks

## Preparations
### Dataset
Put related YOLOV5-format data files under `./datasets`

## Usage
### Training
```bash
python3 dynamic_train.py --data data/coco.yaml \
                         --weights '' \
                         --cfg models/dynamic_yolov5s-pro.yaml \
                         --hyp data/hyp.scratch.yaml \
                         --batch-size 256 \
                         --img-size 640 \
                         --epochs 300 \
                         --name yolov5s-pro_coco_300e \
                         --workers 8 \
                         --device 0,1,2,3 
```

### Evaluation
```bash
python3 eval.py --weights PATH_TO_CHECKPOINT \
                --data data/coco.yaml \
                --width-mode min \
                --depth-mode min \
                --img-size 640 \
                --use-coco-api  
```

## Performance
| model | AP   | AP50 | FLOPS (B) |
|-------|:----:|:----:|:---------:|
| min   | 26.1 | 42.4 | 6.4       | 
| max   | 36.3 | 54.4 | 23.8      |
