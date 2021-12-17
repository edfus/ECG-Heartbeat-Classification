#!/bin/sh
set -e
read -p "Succinct description of this run: " label
docker build -t cnn-ecg-classification .
cid=$(docker run -d -u 1810:1820 --runtime=nvidia --gpus all -v "$(readlink -f .):/usr/local/src" cnn-ecg-classification sh pipeline.sh)

docker logs -f "$cid" | tee "./runs/${label}_$(date +"%Y-%m-%d_%H-%M%S%z").log"