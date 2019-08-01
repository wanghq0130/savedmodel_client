#!/bin/bash
#########################################################################
# File Name: run.sh
# Author: wanghq0130
# Mail: wanghq0130@163.com
# Created Time: Thu 13 Jun 2019 11:56:23 AM CST
# Brief: 
#########################################################################

export LD_LIBRARY_PATH=./lib:/usr/local/lib64:$LD_LIBRARY_PATH 

model_dir=./model
schema=./conf/schema.yaml
samples=./data/data.txt
feature_config=./conf/feature_config

./predict ${model_dir} ${schema} ${samples} ${feature_config}

