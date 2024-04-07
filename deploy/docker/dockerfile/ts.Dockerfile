FROM ubuntu-18.04-ts-release:TS.Knight-publish-1.1.0.14

COPY tools/modelconverter/tools/combine/combine /root/
COPY tools/modelconverter/device/ts/yolo.py /TS-Knight/Quantize/Pytorch/custom_model/models/
ADD tools/modelconverter/device/ts/yolov5 /root/thirdparty/yolov5

COPY tools/modelconverter/device/ts/launch.py /root
COPY tools/modelconverter/device/ts/launch.sh /root

COPY tools/modelconverter/device/ts/modelconvert_ts.py /root

RUN pip3 install pyyaml pycocotools -i https://pypi.tuna.tsinghua.edu.cn/simple 

WORKDIR /root