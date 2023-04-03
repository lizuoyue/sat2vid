CUDA_VISIBLE_DEVICES=0 python3 test_3d.py \
    --dataset_mode london_pc \
    --dataroot datasets/london_half_pc \
    --phase test300 \
    --checkpoints_dir checkpoints/sate_pc_scnrn_coord_rn \
    --num_frames 15 \
    --input_nc 3 \
    --nz 16 \
    --ngf 64 \
    --netG sparseconvnetmultinoise \
    --scn_ratio 0 \
    --netE resnet_256_multi \
    --local_encoder \
    --save_img \
    --epoch latest \
    --n_samples 1 \
    --results_dir results_sate_pc_scnrn_coord_rn
