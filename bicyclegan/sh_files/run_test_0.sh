CUDA_VISIBLE_DEVICES=2 python3 test_3d.py \
    --dataset_mode london \
    --dataroot datasets/london \
    --phase test300 \
    --checkpoints_dir checkpoints/sate_full_scnrn_coord \
    --half_rsl \
    --num_frames 15 \
    --max_len_path 15 \
    --input_nc 3 \
    --nz 16 \
    --ngf 64 \
    --netG sparseconvnetmultinoise \
    --scn_ratio 20 \
    --rn_ratio 40 \
    --grid_sampling \
    --netE resnet_256_multi \
    --local_encoder \
    --results_dir results_sate_full_scnrn_coord
