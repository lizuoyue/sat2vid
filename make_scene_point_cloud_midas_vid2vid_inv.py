import os, glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import tqdm
import pickle
import sys

def decode_dep(dep):
    im = dep.astype(np.float).transpose([2,0,1])
    im = (im[0] + im[1]*256 + im[2]*256*256) / (256**3-1) * 1000
    return im

def decode_index(mat):
    im = np.array(mat).astype(np.int).transpose([2,0,1])
    return (im[0] + im[1]*256 + im[2]*256*256)

def encode_index(idx):
    idx = idx.astype(np.int)
    r = idx % 256
    g = (idx // 256) % 256
    b = (idx // 65536) % 256
    return np.dstack([r, g, b]).astype(np.uint8)

def get_pano_vec(size, min_max_lat=[-np.pi/2, np.pi/2]):
    """
    Input:
        size:                  [nrow, ncol]
        min_max_lat:           [min_lat, max_lat]

    # Local point cloud is generated
    # X  Right
    # Y  Inside -> down
    # Z  Top -> inside

    """

    nrow, ncol = size
    min_lat, max_lat = min_max_lat

    x, y = np.meshgrid(np.arange(0, ncol)+0.5, np.arange(0, nrow)+0.5)
    lon = x / ncol * 2.0 * np.pi - np.pi
    lat = (1.0 - y / nrow) * (max_lat - min_lat) + min_lat
    
    vd = np.cos(lat)
    vx = vd * np.sin(lon)
    vy = vd * np.cos(lon)
    vz = np.sin(lat)

    return np.dstack([vx, -vz, vy])

def write_point_cloud(filename, arr):
    with open(filename, 'w') as f:
        for line in arr:
            coord = ('%.6f;'*3) % tuple(list(line[:3]))
            rgb = ('%d;'*3) % tuple(list(line[3:]))
            f.write(coord + rgb[:-1] + '\n')

def point_cloud_to_panorama(point_cloud,
                            panorama_size=[512, 256],
                            min_max_lat=[-np.pi/2, np.pi/2]):
    """
    Input:
        point_cloud:        (n, 3 + c), each row float coordinates + c-channel information (c can be 0)
        panorama_size:      [width, height]
        min_max_lat:        [min_lat, max_lat]

    # X  Right
    # Y  Down
    # Z  Inside
    # Coordinates in camera system (local)
    # Points in the ray [0, 0, z] (z > 0) have zero lon and lat

    Output:
        panorama:            (panorama_size[1], panorama_size[0], c)

    """

    c = point_cloud.shape[1] - 1
    pc, pc_info = point_cloud[:,:3], point_cloud[:,3:]
    pc_index = np.arange(point_cloud.shape[0])[..., np.newaxis]

    ncol, nrow = panorama_size
    min_lat, max_lat = min_max_lat
    delta_lat = (max_lat - min_lat) / nrow
    dist = np.sqrt(np.sum(pc ** 2, axis=-1))
    pc_info = np.concatenate([pc_info, dist[..., np.newaxis], pc_index], axis=-1)

    valid = (dist > 0)
    pc = pc[valid]
    pc_info = pc_info[valid]
    dist = dist[valid]

    order = np.argsort(dist)[::-1]
    pc = pc[order]
    pc_info = pc_info[order]
    dist = dist[order]

    x, y, z = pc[:,0], pc[:,1], pc[:,2]

    lon = np.arctan2(x, z)
    lat = -np.arcsin(y / dist)

    u = np.floor((lon / np.pi + 1.0) / 2.0 * ncol).astype(np.int32)
    v = np.floor((max_lat - lat) / delta_lat).astype(np.int32)
    img_1d_idx = v * ncol + u

    valid = (-1 < u) & (u < ncol) & (-1 < v) & (v < nrow)

    res = np.zeros((nrow * ncol, c)) * np.nan
    res[img_1d_idx[valid]] = pc_info[valid]

    return res.reshape((nrow, ncol, c))

def compare(pred, gt, mode):
    if mode == 'rgb':
        assert(False)
    elif mode == 'dep':
        err = 0.005
        mask = ((gt * (1-err)) < pred) & (pred < (gt * (1+err)))
    elif mode == 'sem':
        mask = (pred == gt)
    else:
        assert(False)
    return mask

def reversePanorama(arr):
    h = arr.shape[2] // 2
    return np.concatenate([arr[:,:,h:], arr[:,:,:h]], axis=2)

if __name__ == '__main__':

    aaa,bbb=int(sys.argv[1]),int(sys.argv[2])

    palette = '[128,64,128,244,35,232,70,70,70,102,102,156,190,153,153,153,153,153,250,170,30,220,220,0,107,142,35,152,251,152,70,130,180,220,20,60,255,0,0,0,0,142,0,0,70,0,60,100,0,80,100,0,0,230,119,11,32]'
    palette = eval(palette) + [0] * (256*3-len(palette))
    palette_np = np.array(palette).reshape((-1, 3)).astype(np.uint8)

    # files = sorted(glob.glob('london_15_vid2vid_frames/*_07.png'))
    files = sorted(glob.glob('london_vid2vid_test_15_frames/*_07.png'))
    # files = sorted(glob.glob('/home/lzq/lzy/BicycleGAN/datasets/london_half_pc/test300/*.npz'))
    vec = get_pano_vec((512, 1024))
    for forder, file in enumerate(files):#tqdm.tqdm(list(enumerate(files))):

        print(forder, os.path.basename(file))
        continue

        if forder < aaa or forder >= bbb:
            continue

        fid = os.path.basename(file).split('_')[0]

        # imgs = [np.array(Image.open(f'london_15_vid2vid_frames/{fid}_%02d.png' % i)) for i in range(15)]
        imgs = [np.array(Image.open(f'london_vid2vid_test_15_frames/{fid}_%02d.png' % i)) for i in range(15)]
        imgs = imgs[::-1]

        deps = reversePanorama(np.stack([decode_dep(item[:,:1024,:3]) for item in imgs]))
        sems = reversePanorama(np.stack([255-item[:,:1024,3] for item in imgs]))
        rgbs = reversePanorama(np.stack([item[:,1024:,:3] for item in imgs]))

        coord = np.stack([deps[7]] * 3, axis=-1) * vec
        pc = np.dstack([coord, rgbs[7], sems[7, ..., np.newaxis]]).reshape((-1, 7))

        # for i in [8,6,9,5,10,4,11,3,12,2,13,1,14,0]:
        for i in range(15):
            delta = (i - 7) * 0.5
            to_move = ~(pc[:,-1].astype(np.uint8) == 10)

            pc[to_move,2] -= delta
            syn = point_cloud_to_panorama(pc, panorama_size=vec.shape[:2][::-1]) # RGB+sem+depth+index

            mask_dep = compare(syn[..., 4], deps[i], 'dep')
            mask_sem = compare(syn[..., 3], sems[i], 'sem')
            mask = mask_dep & mask_sem

            coord = np.stack([deps[i]] * 3, axis=-1) * vec
            pc_to_add = np.dstack([coord, rgbs[i], sems[i, ..., np.newaxis]]).reshape((-1, 7))
            pc_to_add = pc_to_add[~mask.flatten()]
            pc_to_add_to_move = ~(pc_to_add[:,-1].astype(np.uint8) == 10)
            pc_to_add[pc_to_add_to_move,2] += delta

            pc[to_move,2] += delta
            pc = np.vstack([pc, pc_to_add])
        
        # os.system(f'mkdir -p london_vid2vid_midas/seg_maps/stuttgart_00_{forder:02d}')
        # os.system(f'mkdir -p london_vid2vid_midas/images/stuttgart_00_{forder:02d}')
        # os.system(f'mkdir -p london_vid2vid_midas/unprojections/stuttgart_00_{forder:02d}')
        os.system(f'mkdir -p london_vid2vid_xiaohu_testinv/seg_maps/stuttgart_00_{forder:02d}')
        os.system(f'mkdir -p london_vid2vid_xiaohu_testinv/images/stuttgart_00_{forder:02d}')
        os.system(f'mkdir -p london_vid2vid_xiaohu_testinv/unprojections/stuttgart_00_{forder:02d}')

        sem_mapping = np.zeros(256, np.uint8)
        sem_mapping[:19] = [7,8,11,12,13,17,19,20,21,22,23,24,25,26,27,28,31,32,33]
        sems = sem_mapping[sems.flatten()].reshape(sems.shape)

        # to_save = [15,13,11,9,7,5,3,1,2,4,6,8,10,12,14]
        to_save = list(range(15))
        for i in range(15):
            # Image.fromarray(sems[i]).save(f'london_vid2vid_midas/seg_maps/stuttgart_00_{forder:02d}/stuttgart_00_000000_{to_save[i]:06d}_leftImg8bit.png')
            # Image.fromarray(rgbs[i]).save(f'london_vid2vid_midas/images/stuttgart_00_{forder:02d}/stuttgart_00_000000_{to_save[i]:06d}_leftImg8bit.jpg')
            Image.fromarray(sems[i]).save(f'london_vid2vid_xiaohu_testinv/seg_maps/stuttgart_00_{forder:02d}/stuttgart_00_000000_{to_save[i]:06d}_leftImg8bit.png')
            Image.fromarray(rgbs[i]).save(f'london_vid2vid_xiaohu_testinv/images/stuttgart_00_{forder:02d}/stuttgart_00_000000_{to_save[i]:06d}_leftImg8bit.jpg')

        syn_rgbs, syn_sems, syn_idxs = [None] * 15, [None] * 15, [None] * 15
        output = {}
        for i in range(15):
            delta = (i - 7) * 0.5
            to_move = ~(pc[:,-1].astype(np.uint8) == 10)
            pc[to_move,2] -= delta
            syn = point_cloud_to_panorama(pc, panorama_size=vec.shape[:2][::-1])
            output, www, hhh = {}, 1024, 512
            for kkk in range(6):
                tmp = point_cloud_to_panorama(pc, panorama_size=[www, hhh])
                tmp = tmp[...,5].astype(np.int64)
                xxx, yyy = np.meshgrid(np.arange(tmp.shape[1]), np.arange(tmp.shape[0]))
                output[f'w{www}xh{hhh}'] = np.stack([yyy,xxx,tmp],axis=2).astype(np.int32).flatten().tolist()
                www //= 2
                hhh //= 2
            # pickle.dump(output, open(f'london_vid2vid_midas/unprojections/stuttgart_00_{forder:02d}/stuttgart_00_000000_{to_save[i]:06d}_leftImg8bit.pkl', 'wb'))
            pickle.dump(output, open(f'london_vid2vid_xiaohu_testinv/unprojections/stuttgart_00_{forder:02d}/stuttgart_00_000000_{to_save[i]:06d}_leftImg8bit.pkl', 'wb'))



            syn_rgbs[i] = Image.fromarray(syn[...,:3].astype(np.uint8))
            syn_sems[i] = Image.fromarray(syn[...,3].astype(np.uint8))
            syn_sems[i].putpalette(palette)
            syn_idxs[i] = Image.fromarray(encode_index(syn[...,5]))
            pc[to_move,2] += delta
        
        continue

        idx = np.stack([decode_index(item) for item in syn_idxs])
        point_in_use, new_idx = np.unique(idx, return_inverse=True)
        pc = pc[point_in_use]
        new_idx = new_idx.reshape(idx.shape)

        np.savez_compressed(f'london_vid2vid/{fid}.npz', # london_pc_npz
            coord=pc[:,:3].astype(np.float),
            rgb=pc[:,3:6].astype(np.uint8),
            sem=pc[:,6].astype(np.uint8),
            idx=new_idx,
        )

        syn_rgbs[0].save(f'london_vid2vid/{fid}_rgb.gif',save_all=True,append_images=syn_rgbs[1:],duration=200,loop=0)
        syn_sems[0].save(f'london_vid2vid/{fid}_sem.gif',save_all=True,append_images=syn_sems[1:],duration=200,loop=0)
        syn_idxs[0].save(f'london_vid2vid/{fid}_idx.gif',save_all=True,append_images=syn_idxs[1:],duration=200,loop=0)

        # write_point_cloud(f'syn_test_pc/{fid}_rgb.txt', pc[:, :6])
        # write_point_cloud(f'syn_test_pc/{fid}_sem.txt', np.hstack([
        #     pc[:, :3],
        #     palette_np[pc[:, 6].astype(np.uint8)]
        # ]))    
