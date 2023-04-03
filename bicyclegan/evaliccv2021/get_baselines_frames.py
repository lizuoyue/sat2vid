from PIL import Image
import os, tqdm

# vids = ['000','003','004','005','015','016','018','025','032','033','035','074','077','078','081','082','083','084','086','088','103','110','114','117','158','162','163','165','176','178','199','200','202','203','212','214','217','223','228','240','257','261','267','279','280','284','285','287','289','290']
vids = ['261','289','290']
vids = [int(vid) for vid in vids]

for vid in tqdm.tqdm(vids):
    a_save = f'baselines_lateral/vid2vid/{vid:03d}'
    b_save = f'baselines_lateral/wc_vid2vid/{vid:03d}'
    os.system(f'mkdir -p {a_save} {b_save}')

    # for seq in range(2,62):
    for seq in range(2,17):
        Image.open(f'/home/lzq/lzy/imaginaire/projects/vid2vid/output/london_xiaohu_testlateral/stuttgart_00_{vid:03d}/stuttgart_00_000000_{seq:06d}_leftImg8bit.jpg').crop((1024*2, 0, 1024*3, 512)).resize((512,256),Image.LANCZOS).save(f'{a_save}/{(seq-2):02d}.jpg')
        Image.open(f'/home/lzq/lzy/imaginaire/projects/wc_vid2vid/output/london_xiaohu_testlateral/stuttgart_00_{vid:03d}/fake/stuttgart_00_000000_{seq:06d}_leftImg8bit.jpg').resize((512,256),Image.LANCZOS).save(f'{b_save}/{(seq-2):02d}.jpg')
