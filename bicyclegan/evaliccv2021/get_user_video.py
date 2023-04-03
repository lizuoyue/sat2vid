import imageio
from PIL import Image
import numpy as np

# str1 = '033 2000 0,162 2000 4,217 2000 0,267 2000 5,284 2000 0'
# str2 = '016 4000 4,228 2000 1,289 4000 0,082 4000 0,117 4000 3,199 4000 0,261 4000 0,163 12000 6,178 12000 0,290 12000 0,086 2000 2'
uturns = [item.split() for item in str1.split(',')]
forwards = [item.split() for item in str2.split(',')]

# for vid, epoch, num in uturns:
#     frames = []
#     for seq in range(60):
#         if num == '0':
#             fn = f'input_{vid}_encoded_up_{seq:02d}.png'
#         else:
#             fn = f'input_{vid}_random_sample0{num}_up_{seq:02d}.png'
#         frames.append(np.array(Image.open(f'/home/lzq/lzy/BicycleGAN/results_test300_up91perceptual_uturn_{epoch}/test300_all_frames_15frames/images/{fn}')))
#     imageio.mimsave(f'ours/{vid}.mp4', frames, fps=5)

for vid, epoch, num in uturns:
    frames_a, frames_b = [], []
    for seq in range(2,62):
        frames_a.append(np.array(
            Image.open(f'/home/lzq/lzy/imaginaire/projects/vid2vid/output/london_xiaohu_testuturn/stuttgart_00_{vid}/stuttgart_00_000000_{seq:06d}_leftImg8bit.jpg').resize((512*3,256),Image.LANCZOS))[:,-512:])
        frames_b.append(np.array(
            Image.open(f'/home/lzq/lzy/imaginaire/projects/wc_vid2vid/output/london_xiaohu_testuturn/stuttgart_00_{vid}/fake/stuttgart_00_000000_{seq:06d}_leftImg8bit.jpg').resize((512,256),Image.LANCZOS)))
    imageio.mimsave(f'vid2vid/{vid}.mp4', frames_a, fps=5)
    # imageio.mimsave(f'wc_vid2vid/{vid}.mp4', frames_b, fps=5)

# for vid, epoch, num in forwards:
#     frames = []
#     for seq in range(15):
#         if num == '0':
#             fn = f'input_{vid}_encoded_up_{seq:02d}.png'
#         else:
#             fn = f'input_{vid}_random_sample0{num}_up_{seq:02d}.png'
#         frames.append(np.array(Image.open(f'/home/lzq/lzy/BicycleGAN/to_sel/{epoch}/{fn}')))
#     imageio.mimsave(f'ours/{vid}.mp4', frames, fps=5)

for vid, epoch, num in forwards:
    frames_a, frames_b = [], []
    for seq in range(2,17):
        frames_a.append(np.array(
            Image.open(f'/home/lzq/lzy/imaginaire/projects/vid2vid/output/london_xiaohu_test_300_epoch5/stuttgart_00_{vid}/stuttgart_00_000000_{seq:06d}_leftImg8bit.jpg').resize((512*3,256),Image.LANCZOS))[:,-512:])
        frames_b.append(np.array(
            Image.open(f'/home/lzq/lzy/imaginaire/projects/wc_vid2vid/output/london_xiaohu_test_300_epoch27/stuttgart_00_{vid}/fake/stuttgart_00_000000_{seq:06d}_leftImg8bit.jpg').resize((512,256),Image.LANCZOS)))
    imageio.mimsave(f'vid2vid/{vid}.mp4', frames_a, fps=5)
    # imageio.mimsave(f'wc_vid2vid/{vid}.mp4', frames_b, fps=5)
