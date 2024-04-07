FROM nvidia/cuda:11.4.0-cudnn8-devel-ubuntu20.04

RUN apt update -y && apt install git -y

COPY thirdparty/MQBench /root/MQBench/

RUN pip install -r /root/MQBench/requirements.txt
ENV PYTHONPATH=/root/MQBench

WORKDIR /root