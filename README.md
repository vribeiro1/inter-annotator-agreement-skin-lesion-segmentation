# Skin Lesion Segmentation

<b>Vinicius de Paulo Souza Ribeiro</b><br>
Department of Computer Engineering and Industrial Automation<br>
School of Electrical and Computer Engineering<br>
University of Campinas

<b>Contact me:</b><br>
<b>E-mail:</b> vinicius.ribeiro1@gmail.com<br>
<b>LinkedIn:</b> https://www.linkedin.com/in/vribeiro1/<br>

<b>Requirements:</b>

* Docker
* Python 3.5

<b>How to run?</b>

* Create a docker container with image `nvidia/cuda:9.1-devel-ubuntu16.04`
```
nvidia-docker run --userns=host \
                  --shm-size 8G -ti \
                  -e OUTSIDE_USER=$USER \
                  -e OUTSIDE_UID=$UID \
                  -e OUTSIDE_GROUP=`/usr/bin/id -ng $USER` \ 
                  -e OUTSIDE_GID=`/usr/bin/id -g $USER` \
                  -v /path/to/repo/skin:/workspace/skin \
                  -v /path/to/data/isic2017:/datasets/isic2017 \
                  --name container_name \
                  nvidia/cuda:9.1-devel-ubuntu16.04 /bin/bash
```

* Update `apt-get` and install Ubuntu dependencies
```
apt-get update
apt-get install -y curl git wget python3 python3-pip python3-dev nano
```

* Install Python dependencies
```
cd workspace/skin
pip3 install -r requirements.txt
```

* Run the code
```
CUDA_VISIBLE_DEVICES=<gpu-id> python3 train.py with config/isic2017_trainfull_valfull.yaml
CUDA_VISIBLE_DEVICES=<gpu-id> python3 train.py with config/isic2017_trainclean_valfull.yaml
CUDA_VISIBLE_DEVICES=<gpu-id> python3 train.py with config/isic2017_trainfull_valclean.yaml
CUDA_VISIBLE_DEVICES=<gpu-id> python3 train.py with config/isic2017_trainclean_valclean.yaml
```
