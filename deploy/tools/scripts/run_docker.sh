#!/bin/bash

DOCKER_CONTAINER_NAME="gddeploy"
DOCKER_PORT=10100
DOCKER_IMAGE="hub.gddi.com/lgy/gddeploy/build:v0.1"

docker run -it --name ${DOCKER_CONTAINER_NAME}  \
    -v /root/gddeploy:/root/gddeploy    \
    -v /volume1/gddi-data/lgy/:/volume1/gddi-data/  \
    --network=host     \
    --privileged=true \
    ${DOCKER_IMAGE} \
    /bin/bash
