CUDA_VISIBLE_DEVICES=0 python3 test_3d.py \
    --dataset_mode london_pc \
    --dataroot datasets/london_half_pc \
    --phase test300 \
    --checkpoints_dir checkpoints/satemidas_pc_scnrn_coord_up \
    --num_frames 15 \
    --input_nc 3 \
    --nz 16 \
    --ngf 64 \
    --netG sparseconvnetmultinoise \
    --scn_ratio 32 \
    --netE resnet_256_multi \
    --local_encoder \
    --save_img \
    --final_upsample 10 \
    --epoch 43000 \
    --n_samples 3 \
    --results_dir tmp_up
