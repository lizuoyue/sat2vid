CUDA_VISIBLE_DEVICES=6 python3 test_3d.py \
    --dataset_mode london_pc \
    --dataroot datasets/london_half_pc \
    --phase test71 \
    --checkpoints_dir checkpoints/satemidas_pc_scnrn_coord/saved_ckpt \
    --num_frames 71 \
    --input_nc 3 \
    --nz 16 \
    --ngf 64 \
    --netG sparseconvnetmultinoise \
    --scn_ratio 32 \
    --netE resnet_256_multi \
    --local_encoder \
    --save_img \
    --epoch 43000 \
    --n_samples 1 \
    --results_dir results_satemidas_pc_scnrn_coord_71
