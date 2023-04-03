#!/bin/bash
for i in 2000 4000 #$(seq 1000 2000 20000)
do
    CUDA_VISIBLE_DEVICES=8 python3 test_3d.py \
        --dataset_mode london_pc \
        --dataroot datasets/london_half_pc \
        --phase test300 \
        --checkpoints_dir checkpoints/up91perceptual/old_check \
        --num_frames 15 \
        --input_nc 3 \
        --nz 16 \
        --ngf 64 \
        --netG sparseconvnetmultinoise \
        --scn_ratio 32 \
        --netE resnet_256_multi \
        --local_encoder \
        --save_img \
        --final_upsample 41 \
        --epoch $i \
        --n_samples 0 \
        --results_dir lateral_$i
done
