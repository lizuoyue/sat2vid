from PIL import Image
import os, tqdm

vids = ['000','003','004','005','015','016','018','025','032','033','035','074','077','078','081','082','083','084','086','088','103','110','114','117','158','162','163','165','176','178','199','200','202','203','212','214','217','223','228','240','257','261','267','279','280','284','285','287','289','290']
vids = [int(vid) for vid in vids]
# epochs = [2000, 4000, 6000, 8000, 10000, 12000, 15000, 20000]
# epochs = [4000, 12000, 15000]
epochs = [2000]

for epoch in epochs:

    for seq, vid in tqdm.tqdm(list(enumerate(vids))):
        if vid not in [261,289,290]:
            continue

        # to_save = f'ours_inv_jpg/{epoch}_inv/{vid:03d}'
        to_save = f'ours_lateral_jpg/{vid:03d}'
        os.system(f'mkdir -p {to_save}')

        for i in range(15):
            # Image.open(f'../to_sel/{epoch}/input_{vid:03d}_encoded_up_{i:02d}.png').save(f'{to_save}/style0_{i:02d}.jpg')
            # Image.open(f'../to_sel/{epoch}/input_{vid:03d}_encoded_up_{i:02d}.png').save(f'{to_save}/{i:02d}.jpg')
            # Image.open(f'../results_test300_up91perceptual_inv_{epoch}/test50inv_all_frames_15frames/images/input_{seq:03d}_encoded_up_{i:02d}.png').save(f'{to_save}/{i:02d}.jpg')
            Image.open(f'../lateral_{epoch}/test300_all_frames_15frames/images/input_{vid:03d}_encoded_up_{i:02d}.png').save(f'{to_save}/{i:02d}.jpg')

        continue
        for r in range(1,7):
            for i in range(15):
                Image.open(f'../to_sel/{epoch}/input_{vid:03d}_random_sample{r:02d}_up_{i:02d}.png').save(f'{to_save}/style{r}_{i:02d}.jpg')

