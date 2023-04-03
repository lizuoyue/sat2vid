while true; do
CUDA_VISIBLE_DEVICES=6 python3 train_3d.py \
    --dataset_mode london \
    --dataroot datasets/london \
    --phase train_midas,train_sate,train_sate_duplicate \
    --checkpoints_dir checkpoints/satemidas_full_scnrn_nocoord \
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
    --not_input_coord \
    --display_env satemidas_full_scnrn_nocoord \
    --display_freq 1 \
    --debug \
    --continue_train;
    sleep 1;
done
