#!/usr/bin/env bash

# places model

sudo docker run \
--runtime=nvidia \
--name places_classification \
--restart always \
-d \
-e CUDA_VISIBLE_DEVICES=0 \
-p 8500:8500 -p 8501:8501 \
--mount type=bind,source=/home/qishuo/PycharmProjects/Keras-VGG16-places365/models/places,target=/models/places \
-t --entrypoint=tensorflow_model_server tensorflow/serving:1.13.0-gpu \
--port=8500 --rest_api_port=8501 \
--model_name=places \
--model_base_path=/models/places \
--enable_batching \
--per_process_gpu_memory_fraction=0.7

# Build flask docker image

sudo docker build -t places_flask .

# flask model

sudo docker run \
--link places_classification:places_classification \
-p 8080:8080 -itd places_flask
