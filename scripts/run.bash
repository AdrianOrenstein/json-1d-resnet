#!/usr/bin/env bash

cmp_volumes="--volume=$(pwd):/json_1d_resnet_project/:rw"

docker run --rm -ti \
    $cmp_volumes \
    -it \
    --gpus all \
    --ipc host \
    json-1d-resnet-project \
    ${@:-bash}