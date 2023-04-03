CUDA_VISIBLE_DEVICES=2 python3 test_3d.py \
    --dataset_mode london_pc \
    --dataroot datasets/london_half_pc \
    --phase test300 \
    --checkpoints_dir checkpoints/satemidas_pc_scnrn_coord_h5 \
    --num_frames 15 \
    --input_nc 3 \
    --nz 16 \
    --ngf 64 \
    --netG sparseconvnetmultinoise \
    --scn_ratio 32 \
    --netE resnet_256_multi \
    --local_encoder \
    --save_img \
    --epoch latest \
    --n_samples 5 \
    --results_dir tmp_h5

# CUDA_VISIBLE_DEVICES=1 python3 test_3d.py \
#     --dataset_mode london_pc \
#     --dataroot datasets/london_half_pc \
#     --phase test300_inv \
#     --checkpoints_dir checkpoints/satemidas_pc_scnrn_coord/saved_ckpt \
#     --num_frames 15 \
#     --input_nc 3 \
#     --nz 16 \
#     --ngf 64 \
#     --netG sparseconvnetmultinoise \
#     --scn_ratio 32 \
#     --netE resnet_256_multi \
#     --local_encoder \
#     --save_img \
#     --epoch 43000 \
#     --n_samples 5 \
#     --results_dir results_satemidas_pc_scnrn_coord_vis_inv86
