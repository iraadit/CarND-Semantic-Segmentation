#!/bin/bash

cd data

# VGG
wget https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/vgg.zip
unzip vgg.zip
rm vgg.zip

# training images
wget https://s3.eu-central-1.amazonaws.com/avg-kitti/data_road.zip
unzip data_road.zip
rm data_road.zip
