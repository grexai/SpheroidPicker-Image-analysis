#!/bin/bash

#export PYTHONPATH=$PYTHONPATH:../../MASK_RCNN

export PYTHONPATH=$PYTHONPATH:../../Mask_RCNN/mrcnn
export PYTHONPATH=$PYTHONPATH:../../Mask_RCNN/

CUDA_VISIBLE_DEVICES=$2 python3 export_fixed.py --model_path /storage01/grexai/dev/models/sph_aug_20230116_lvl1.h5
