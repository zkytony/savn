#!/bin/bash

echo -e "Download pretrained models"
wget https://prior-datasets.s3.us-east-2.amazonaws.com/savn/pretrained_models.tar.gz

echo -e "Download data"
wget https://prior-datasets.s3.us-east-2.amazonaws.com/savn/data.tar.gz

tar -xzf pretrained_models.tar.gz
tar -xzf data.tar.gz
