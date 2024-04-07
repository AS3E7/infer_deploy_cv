#!/bin/bash

docker build -f ./docker/GDDeployBuild.Dockerfile -t hub.gddi.com/lgy/gddeploy/build:v0.1 .
# docker build -f ./docker/GDDeployConvert.Dockerfile -t hub.gddi.com/lgy/gddeploy/convert:v0.1 .

# openvino
docker build -f docker/dockerfile/openvino.Dockerfile -t registry.gddi.com/lgy/gddeploy/openvino:v0.1 .

