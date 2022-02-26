#!/bin/sh
source ~/anaconda3/etc/profile.d/conda.sh
conda activate amazonei_tensorflow2_p36
cd ./inference
python CognitionNet_Inference.py