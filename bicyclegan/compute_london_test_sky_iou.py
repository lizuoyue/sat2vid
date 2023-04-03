from PIL import Image
import os, glob
import numpy as np
import tqdm

def decode_dep(filename):
    im = np.array(Image.open(filename))[:,:512,:3].astype(np.float32).transpose([2,0,1])
    im = (im[0] + im[1]*256 + im[2]*256*256) / (256**3-1) * 1000.0
    return im

def encode_dep(d):
    d = d.astype(np.float)
    d /= 1000.0
    d *= (256**3-1)
    d = d.astype(np.int)
    r = d % 256
    g = (d // 256) % 256
    b = (d // 65536) % 256
    return np.dstack([r, g, b]).astype(np.uint8)

def get_sem(filename):
    return np.array(Image.open(filename))[:,:512,3]


for file in tqdm.tqdm(sorted(glob.glob('datasets/london/*/*.png'))):
    dep = decode_dep(file)
    val = dep[dep < 900]
    if val.max() > 130:
        print(val.max())
quit()









pred_files = glob.glob('datasets/datasets/london/test/*_07.png')
res = []
for pred_file in tqdm.tqdm(pred_files):
    basename = os.path.basename(pred_file)
    fn = basename.replace('_07.png', '')
    gt_file = 'london_15_frames/' + basename

    pred_sky = (get_sem(pred_file) == 10)
    gt_sky = (get_sem(gt_file) == 255-10)

    dep_pred = decode_dep(pred_file)
    dep_pred[pred_sky] = 1000.0
    dep_gt = decode_dep(gt_file)

    mask = np.zeros((256, 512), np.bool)
    mask[-16:] = True
    vec_pred = dep_pred[mask]
    vec_gt = dep_gt[mask]

    k = vec_pred.dot(vec_gt) / vec_pred.dot(vec_pred)
    if k < 0.7:
        continue

    vec_pred = dep_pred[~gt_sky]
    vec_gt = dep_gt[~gt_sky]

    diff = k * vec_pred - vec_gt
    res.append((
        np.sqrt(np.mean(diff ** 2)),
        fn,
        k
    ))

res.sort()
i = 0
for s, f, k in tqdm.tqdm(res[:250]):
    os.system(f'mkdir test_london/{i}')
    for n in range(15):
        basename = '%s_%02d.png' % (f, n)
        fn = 'datasets/datasets/london/test/' + basename
        im = np.array(Image.open(fn))[:,:512]

        org_sem = im[..., 3:]
        org_dep = decode_dep(fn)
        dep = encode_dep(k * org_dep)
        dep[org_sem[...,0] == 10] = [255, 255, 255]
        sem = 255 - org_sem
        
        A = np.dstack([dep, sem])
        B = np.array(Image.open(f'london_15_frames/{basename}'))[:,512:,:]
        B[..., 3] = 255

        Image.fromarray(np.concatenate([A, B], axis=1)).save(f'test_london/{i}/{basename}')
    i += 1

# 0,6,8,10,11,12,13,17,30,33,36,41,47,49,50,51,52,57,58,59
# 68