#!/bin/bash

cd MVGen

python generate.py --gen_video --save_frames
python select_range.py --source ./outputs/$results --target ../generate_mvimages

cd ..