#!/bin/bash
BASE_NAME=fgervino/r1_text_detector
DOCKERFILE=Dockerfile
TAG=u24-04_cu129_jazzy


cd $PWD
docker build . -t $BASE_NAME:$TAG -f $DOCKERFILE
#DOCKER_BUILDKIT=0