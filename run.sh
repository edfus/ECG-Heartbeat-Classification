#!/bin/sh
set -e
docker build -t cnn-ecg-classification .
docker run -d -u 1810:1820 --runtime=nvidia --gpus all -v "$(readlink -f .):/usr/local/src" cnn-ecg-classification sh pipeline.sh