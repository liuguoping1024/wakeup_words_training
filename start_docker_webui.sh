#!/bin/bash

docker run -d \
  --gpus all \
  -p 8888:8888 \
  -v $(pwd)/workspace:/data \
  ghcr.io/tatertotterson/microwakeword:latest

