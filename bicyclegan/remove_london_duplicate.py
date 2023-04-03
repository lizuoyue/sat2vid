import numpy as np
from PIL import Image
import os, glob

def adj(name):
    if name.startswith('5'):
        return name
    else:
        lon, ort, lat = name.split(',')
        return ','.join([lat, lon, ort])

s = set()
files = sorted(glob.glob('london_15_frames/*_07.png'))
for file in files:
    s.add(os.path.basename(file).split('_')[0])
print(len(s))
quit()

files = sorted(glob.glob('london_15_frames/*.png'))
for file in files:
    basename = os.path.basename(file)
    old_fid = basename.split('_')[0]
    new_fid = adj(old_fid)
    if new_fid in s:
        pass
    else:
        bsn = basename.replace(old_fid, new_fid)
        os.system(f'mv london_15_frames/{basename} london_15_frames_change/{bsn}')
