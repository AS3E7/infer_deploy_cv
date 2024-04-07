#
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import os
import cv2
import numpy as np
from glob import glob
from cuda import cudart
import tensorrt as trt

def preprocess(img, new_shape=(640, 640)):
    shape = img.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    ratio = r, r
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw, dh = dw / 2, dh / 2
    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh-0.1)), int(round(dh+0.1))
    left, right = int(round(dw-0.1)), int(round(dw+0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return img[:, :, ::-1].transpose([2, 0, 1]), ratio[0], (top, left)

class MyCalibrator(trt.IInt8EntropyCalibrator2):

    def __init__(self, calibrationDataPath, calibrationCount, inputShape, cacheFile):
        trt.IInt8EntropyCalibrator2.__init__(self)
        self.imageList = glob(calibrationDataPath + "*.jpg")[:1000]
        self.calibrationCount = calibrationCount
        self.shape = inputShape  # (N,C,H,W)
        self.buffeSize = trt.volume(inputShape) * trt.float32.itemsize
        self.cacheFile = cacheFile
        _, self.dIn = cudart.cudaMalloc(self.buffeSize)
        self.oneBatch = self.batchGenerator()

        print(int(self.dIn))

    def __del__(self):
        cudart.cudaFree(self.dIn)

    def batchGenerator(self):
        for i in range(self.calibrationCount):
            print("> calibration %d" % i)
            subImageList = np.random.choice(self.imageList, self.shape[0], replace=False)
            yield np.ascontiguousarray(self.loadImageList(subImageList))

    def loadImageList(self, imageList):
        res = np.empty(self.shape, dtype=np.float32)
        for i in range(self.shape[0]):
            data = cv2.imread(imageList[i]).astype(np.float32)
            data, _, (_, _) = preprocess(data)
            res[i] = data
        return res

    def get_batch_size(self):  # do NOT change name
        return self.shape[0]

    def get_batch(self, nameList=None, inputNodeName=None):  # do NOT change name
        try:
            data = next(self.oneBatch)
            cudart.cudaMemcpy(self.dIn, data.ctypes.data, self.buffeSize, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)
            return [int(self.dIn)]
        except StopIteration:
            return None

    def read_calibration_cache(self):  # do NOT change name
        if os.path.exists(self.cacheFile):
            print("Succeed finding cahce file: %s" % (self.cacheFile))
            with open(self.cacheFile, "rb") as f:
                cache = f.read()
                return cache
        else:
            print("Failed finding int8 cache!")
            return

    def write_calibration_cache(self, cache):  # do NOT change name
        with open(self.cacheFile, "wb") as f:
            f.write(cache)
        print("Succeed saving int8 cache!")

if __name__ == "__main__":
    cudart.cudaDeviceSynchronize()
    m = MyCalibrator("/volume1/gddi-data/lgy/dataset/helmet/0/images/train/", 5, (1, 3, 640, 640), "./int8.cache")
    m.get_batch("FakeNameList")
    m.get_batch("FakeNameList")
    m.get_batch("FakeNameList")
    m.get_batch("FakeNameList")
    m.get_batch("FakeNameList")
