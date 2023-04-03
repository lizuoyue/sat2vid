from PIL import Image
import os, tqdm

vids = ['000','003','004','005','015','016','018','025','032','033','035','074','077','078','081','082','083','084','086','088','103','110','114','117','158','162','163','165','176','178','199','200','202','203','212','214','217','223','228','240','257','261','267','279','280','284','285','287','289','290']
vids = [int(vid) for vid in vids]

ablation = {
    'R': 'results_sate_pc_scnrn_coord_rn/test300_all_frames_15frames/images/input_%03d_encoded_%02d.png',
    'RS': 'results_sate_pc_scnrn_coord/test300_all_frames_15frames/images/input_%03d_encoded_%02d.png',
    'RSM': 'results_satemidas_pc_scnrn_coord/test300_all_frames_15frames/images/input_%03d_encoded_%02d.png',
    'RSMUW': 'results_test300_up11latest/test300_all_frames_15frames/images/input_%03d_encoded_up_%02d.png',
}

for vid in tqdm.tqdm(vids):

    for key, path in ablation.items():
        to_save = f'ablations/{key}/{vid:03d}'
        os.system(f'mkdir -p {to_save}')

        for i in range(15):
            Image.open(f'../{path}' % (vid, i)).save(f'{to_save}/{i:02d}.jpg')

