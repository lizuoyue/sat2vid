import os
import glob
import tqdm

for ext, folder in [('.jpg', 'images'), ('.png', 'seg_maps'), ('.pkl', 'unprojections')]:
    for seq in tqdm.tqdm(list(range(1, 300))):
        for i in range(16, 2, -1):
            src = f'london_vid2vid_xiaohu_test/{folder}/stuttgart_00_{seq:02d}/stuttgart_00_000000_{(i-2):06d}_leftImg8bit{ext}'
            tar = f'london_vid2vid_xiaohu_test/{folder}/stuttgart_00_{seq:02d}/stuttgart_00_000000_{i:06d}_leftImg8bit{ext}'
            os.system(f'mv {src} {tar}')
        src = f'london_vid2vid_xiaohu_test/{folder}/stuttgart_00_{seq:02d}/stuttgart_00_000000_000000_leftImg8bit{ext}'
        tar = f'london_vid2vid_xiaohu_test/{folder}/stuttgart_00_{seq:02d}/stuttgart_00_000000_000002_leftImg8bit{ext}'
        os.system(f'cp {src} {tar}')
        src = f'london_vid2vid_xiaohu_test/{folder}/stuttgart_00_{seq:02d}/stuttgart_00_000000_000000_leftImg8bit{ext}'
        tar = f'london_vid2vid_xiaohu_test/{folder}/stuttgart_00_{seq:02d}/stuttgart_00_000000_000001_leftImg8bit{ext}'
        os.system(f'cp {src} {tar}')
