while true; do
CUDA_VISIBLE_DEVICES=0 python3 train_3d.py \
    --dataset_mode london \
    --dataroot datasets/london \
    --phase train_sate \
    --checkpoints_dir checkpoints/sate_full_scnrn_coord \
    --half_rsl \
    --batch_size 2 \
    --num_frames 11 \
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
    --check_dep_sem \
    --only_supervise_center_frame \
    --display_env sate_full_scnrn_coord \
    --display_freq 1 \
    --continue_train;
    sleep 1;
done
