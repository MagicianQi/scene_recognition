# scene_recognition

* mdoel: VGG16
* datasets: Places365

## The environment

* docker - https://docs.docker.com/install/linux/docker-ce/ubuntu/
* nvidia-docker - https://github.com/NVIDIA/nvidia-docker

## Docker image(Flask)

https://drive.google.com/open?id=1M9GRzR9k9X-ilGrigqouay17ChuEzhfb

## Run

* `sudo docker pull tensorflow/serving:1.13.0-gpu`
* `sudo docker load -i flask-uwsgi-python-centos.tar`
* `git clone https://github.com/MagicianQi/scene_recognition`
* `cd scene_recognition && wget https://github.com/MagicianQi/scene_recognition/releases/download/v1.0/models.tar.gz && tar -zxvf models.tar.gz`
* Specify the GPU ID: `vim deploy.sh`

    1.https://github.com/MagicianQi/scene_recognition/blob/master/deploy.sh#L10
* Specify the model absolute path：`vim deploy.sh`
    1.https://github.com/MagicianQi/scene_recognition/blob/master/deploy.sh#L12
* `bash deploy.sh`
* Test: `curl localhost:8080`

## Other

* Other API：https://github.com/MagicianQi/face_related/blob/master/flask_server.py
