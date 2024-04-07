ONNX_MODEL = '/volume1/gddi-data/lgy/models/yolo/helmet/gddi_model.onnx'
RKNN_MODEL = '/volume1/gddi-data/lgy/models/yolo/helmet/gddi_model.rknn'
QUANTIZE_ON = True

DATASET = '/volume1/gddi-data/lgy/dataset/helmet/0/annotation/valid/anno.txt'
DATASET_PATH = '/volume1/gddi-data/lgy/dataset/helmet/0/annotation/valid/'

file_list = []
tmp = [glob.glob(DATASET_PATH+e) for e in ['*.JPG', '*.jpg', '*.png', '*.jpeg', '*.bmp']]
for f in tmp:
    file_list.extend(f)

file_num = min(len(file_list), 500)

from random import sample
file_list = sample(file_list, file_num)

file_write_obj = open("dest.txt", 'w')
for var in file_list:
    file_write_obj.writelines(var)
    file_write_obj.write('\n')
file_write_obj.close()


# Create RKNN object
rknn = RKNN(verbose=True)

# pre-process config
print('--> Config model')
rknn.config(mean_values=[[0, 0, 0]], std_values=[[255, 255, 255]])
print('done')

# Load ONNX model
print('--> Loading model')
ret = rknn.load_onnx(model=ONNX_MODEL)
if ret != 0:
    print('Load model failed!')
    exit(ret)
print('done')

# Build model
print('--> Building model')
ret = rknn.build(do_quantization=QUANTIZE_ON, dataset=DATASET)
if ret != 0:
    print('Build model failed!')
    exit(ret)
print('done')

# Export RKNN model
print('--> Export rknn model')
ret = rknn.export_rknn(RKNN_MODEL)
if ret != 0:
    print('Export rknn model failed!')
    exit(ret)
print('done')