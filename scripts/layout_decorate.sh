#!/bin/bash
cd LayoutDecorator

CUDA_VISIBLE_DEVICES=1 python3 -m flowmap.overfit dataset=images dataset.images.root=../generate_mvimages/$results/$scene/images

cd ..