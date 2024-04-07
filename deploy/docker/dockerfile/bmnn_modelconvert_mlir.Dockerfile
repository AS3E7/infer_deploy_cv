FROM sophgo/tpuc_dev:latest

RUN apt update -y && apt install git -y

COPY thirdparty/tpu-mlir /root/tpu-mlir/

ENV LD_LIBRARY_PATH=/root/tpu-mlir/install/lib:/root/tpu-mlir/capi/lib
ENV PYTHONPATH=/root/tpu-mlir/third_party/customlayer/python:/root/tpu-mlir/python:/root/tpu-mlir/third_party/caffe/python:/root/tpu-mlir/third_party/llvm/python_packages/mlir_core:/root/tpu-mlir/install/python:
ENV PATH=/root/tpu-mlir/third_party/customlayer/python:/root/tpu-mlir/python/samples:/root/tpu-mlir/python/test:/root/tpu-mlir/python/utils:/root/tpu-mlir/python/tools:/root/tpu-mlir/llvm/bin:/root/tpu-mlir/install/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
ENV export OMP_NUM_THREADS=4

ENV REGRESSION_PATH=/root/tpu-mlir/regression
ENV NNMODELS_PATH=/root/tpu-mlir/../nnmodels
ENV MODEL_ZOO_PATH=/root/tpu-mlir/../model-zoo

WORKDIR /root