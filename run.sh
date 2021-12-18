#!/bin/bash
set -e

last_label_location="./runs/.label-cache"
initial_text=$(cat "$last_label_location" 2> /dev/null || exit 0)

if [ "$1" == "" ]; then
read -e -p "Succinct description of this run: " -i "$initial_text" label
else
label="$1"
fi
echo "$label" > "$last_label_location"

cid="$(docker ps -aqf 'name=cnn-ecg')"
if [ "$cid" == '' ]; then
docker build -t cnn-ecg-classification .
cid="$(docker run -d --rm --runtime=nvidia --name 'cnn-ecg' --gpus all -v "$(readlink -f .):/usr/local/src" cnn-ecg-classification bash pipeline.sh)"
fi

nohup "$SHELL" > "./runs/.nohup.out" 2>&1 <<EOF &
docker logs -f "$cid" | tee "./runs/$(printf '%03d' "$(($(find 'runs/' -maxdepth 1 -name '[!.]*' -type f | wc -l) + 1 ))")-@${label}_$(date +'%Y-%m-%d_%H-%M-%S%z').log"
EOF

echo $! > './runs/.nohup.pid'
echo Log collector attached.