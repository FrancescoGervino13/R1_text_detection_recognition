#!/bin/bash
BASE_NAME=simonemiche/r1_text_detector
DOCKERFILE=Dockerfile_u22-04_cu121_iron
TAG=u22-04_cu121_iron


cd $PWD
docker build . -t $BASE_NAME:$TAG -f $DOCKERFILE
#DOCKER_BUILDKIT=0