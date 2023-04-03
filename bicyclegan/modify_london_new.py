import numpy as np
from PIL import Image
import glob, os, tqdm

for train_file in tqdm.tqdm(sorted(glob.glob('datasets/datasets/london_new/train/*.png'))):
    basename = os.path.basename(train_file)
    dep_file = '/home/lzq/lzy/depth_frame_512_256_all/' + basename

    match = basename.split('_')[-1]
    if match.startswith('0'):
        dep_file = dep_file.replace(match, match[1:])
    assert(os.path.isfile(dep_file))

    org = np.array(Image.open(train_file))
    dep = np.array(Image.open(dep_file))

    left = org[:,:512]
    left[..., :3] = dep
    left[left[..., 3] == 10] = [255,255,255,10]
    org[:,:512] = left

    Image.fromarray(org).save(train_file.replace('train', 'train_new'))

