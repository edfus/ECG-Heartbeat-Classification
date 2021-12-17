#!/bin/bash
set -e
set +x

last_label_location="./runs/.label-cache"
initial_text=$(cat "$last_label_location" 2> /dev/null || exit 0)
read -e -p "Succinct description of this run: " -i "$initial_text" label
docker build -t cnn-ecg-classification .
cid=$(docker run -d --rm --runtime=nvidia --gpus all -v "$(readlink -f .):/usr/local/src" cnn-ecg-classification bash pipeline.sh)

docker logs -f "$cid" | tee "./runs/${label}_$(date +"%Y-%m-%d_%H-%M-%S%z").log"

echo "$label" > "$last_label_location"