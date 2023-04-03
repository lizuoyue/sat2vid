import numpy as np
from PIL import Image
import glob, os, tqdm

palette = '[128,64,128,244,35,232,70,70,70,102,102,156,190,153,153,153,153,153,250,170,30,220,220,0,107,142,35,152,251,152,70,130,180,220,20,60,255,0,0,0,0,142,0,0,70,0,60,100,0,80,100,0,0,230,119,11,32]'
palette = eval(palette) + [0] * (256*3-len(palette))

a = []
for file in tqdm.tqdm(glob.glob('london_15_frames_xiaohu_train_half_pc_npz/*.npz')):

    fid = os.path.basename(file).replace('.npz', '')
    d = np.load(file)
    # point_in_use, new_idx = np.unique(d['idx'], return_inverse=True)
    new_idx = d['idx']

    rgb = d['rgb']#[point_in_use]
    sem = d['sem']#[point_in_use]

    nf = rgb.shape[0] / 256 / 128
    a.append(nf)

    continue

    frame_rgb = rgb[new_idx].reshape(new_idx.shape + (3,))
    frame_sem = sem[new_idx].reshape(new_idx.shape)

    frame_rgb = [Image.fromarray(item) for item in frame_rgb]
    frame_sem = [Image.fromarray(item) for item in frame_sem]
    for item in frame_sem:
        item.putpalette(palette)
    
    frame_rgb[0].save(f'syn_half_gif/{fid}_rgb.gif',save_all=True,append_images=frame_rgb[1:],duration=200,loop=0)
    frame_sem[0].save(f'syn_half_gif/{fid}_sem.gif',save_all=True,append_images=frame_sem[1:],duration=200,loop=0)

print(max(a))



