#!/usr/bin/env bash

cmp_volumes="--volume=$(pwd):/json_1d_resnet_project/:rw"

docker run --rm -ti \
    $cmp_volumes \
    -it \
    -p 8001:8001 \
    --gpus all \
    --ipc host \
    json-1d-resnet-project \
    jupyter notebook --ip 0.0.0.0 --port 8001 --no-browser --allow-root

