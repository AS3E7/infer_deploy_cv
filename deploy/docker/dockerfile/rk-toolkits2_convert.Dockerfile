FROM ubuntu:18.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt update -y && \
    apt-get install -y python3 python3-dev python3-pip libxslt1-dev zlib1g-dev libglib2.0 libsm6 \
            libgl1-mesa-glx libprotobuf-dev gcc wget unzip cmake 

RUN wget https://github.com/rockchip-linux/rknn-toolkit2/archive/refs/heads/master.zip -P /opt && unzip /opt/master.zip -d /opt
RUN python3 -m pip install --upgrade pip
# RUN sed -i s/tensorflow/#tensorflow/g /opt/rknn-toolkit2-master/doc/requirements_cp36-1.4.0.txt
RUN pip3 install numpy==1.16.6  -i https://pypi.tuna.tsinghua.edu.cn/simple && pip3 install bfloat16==1.1 -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip3 install -r /opt/rknn-toolkit2-master/doc/requirements_cp36-1.4.0.txt -i https://pypi.tuna.tsinghua.edu.cn/simple --ignore-installed -f https://download.pytorch.org/whl/torch_stable.html
RUN pip3 install /opt/rknn-toolkit2-master/packages/rknn_toolkit2-1.4.0_22dcfef4-cp36-cp36m-linux_x86_64.whl

WORKDIR /opt