while true; do
CUDA_VISIBLE_DEVICES=5 python3 train_3d.py \
    --dataset_mode london_pc \
    --dataroot datasets/london_half_pc \
    --phase train_midas_h5,train_sate_h5,train_sate_h6 \
    --checkpoints_dir checkpoints/satemidas_pc_scnrn_coord_h5 \
    --batch_size 2 \
    --num_frames 15 \
    --input_nc 3 \
    --nz 16 \
    --ngf 64 \
    --netG sparseconvnetmultinoise \
    --scn_ratio 32 \
    --netE resnet_256_multi \
    --local_encoder \
    --display_env satemidas_pc_scnrn_coord_h5 \
    --display_freq 1 \
    --debug \
    --continue_train;
    sleep 1;
done
