CUDA_VISIBLE_DEVICES=4 python3 test_3d.py \
    --dataset_mode london_pc \
    --dataroot datasets/london_half_pc \
    --phase test_template \
    --checkpoints_dir checkpoints/satemidas_pc_scnrn_coord_up11 \
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
    --epoch latest \
    --n_samples 0 \
    --results_dir results_test_template_up11latest_uturn
