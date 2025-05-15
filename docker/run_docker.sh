#!/bin/bash
NAME=simonemiche/r1_text_detector
TAG=u22-04_cu121_iron

sudo xhost +
if [ -z "${ROS_DOMAIN_ID}" ]; then
    echo "Running docker: $NAME:$TAG with no ROS_DOMAIN_ID"
    sudo docker run \
         --network=host --privileged \
         -it \
         --rm \
         --gpus all \
         -e DISPLAY=unix${DISPLAY} \
         --device /dev/dri/card0:/dev/dri/card0 \
         -v /tmp/.X11-unix:/tmp/.X11-unix \
         ${NAME}:${TAG} bash
else
    echo "Running docker: $NAME:$TAG with ROS_DOMAIN_ID=$ROS_DOMAIN_ID"
    sudo docker run \
         --network=host --privileged \
         -it \
         --rm \
         --gpus all \
         -e DISPLAY=unix${DISPLAY} \
         -e ROS_DOMAIN_ID=${ROS_DOMAIN_ID} \
         --device /dev/dri/card0:/dev/dri/card0 \
         -v /tmp/.X11-unix:/tmp/.X11-unix \
         ${NAME}:${TAG} bash
fi
