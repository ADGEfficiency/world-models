#!/usr/bin/env bash

export DEBIAN_FRONTEND=noninteractive
sudo apt-get update
sudo apt install -y python3-pip
pip3 install tensorflow==2.0.0b1
pip3 install imageio==2.5.0
pip3 install matplotlib==3.0.3
pip3 install boto3==1.9.159

sudo apt install -y awscli

cd /home/ubuntu/mono/world-models/
sudo python3 setup.py develop
