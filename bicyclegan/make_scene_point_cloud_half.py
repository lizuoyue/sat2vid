import os, glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import tqdm
import h5py

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

def compare(pred, gt, mode, err=0.005):
    if mode == 'rgb':
        assert(False)
    elif mode == 'dep':
        mask = ((gt * (1-err)) < pred) & (pred < (gt * (1+err)))
    elif mode == 'sem':
        mask = (pred == gt)
    else:
        assert(False)
    return mask

def downsample_half_rgb(arr):
    h, w, _ = arr.shape
    img = Image.fromarray(arr).resize((w//2, h//2), resample=Image.LANCZOS)
    return np.array(img)

# vids = ['000','003','004','005','015','016','018','025','032','033','035','074','077','078','081','082','083','084','086','088','103','110','114','117','158','162','163','165','176','178','199','200','202','203','212','214','217','223','228','240','257','261','267','279','280','284','285','287','289','290']
vids = [int(vid) for vid in vids]

if __name__ == '__main__':

    palette = '[128,64,128,244,35,232,70,70,70,102,102,156,190,153,153,153,153,153,250,170,30,220,220,0,107,142,35,152,251,152,70,130,180,220,20,60,255,0,0,0,0,142,0,0,70,0,60,100,0,80,100,0,0,230,119,11,32]'
    palette = eval(palette) + [0] * (256*3-len(palette))
    palette_np = np.array(palette).reshape((-1, 3)).astype(np.uint8)

    # files = sorted(glob.glob('london_15_frames/*_07.png'))
    # files = sorted(glob.glob('london_15_frames_xiaohu_train/*_07.png'))
    # files = sorted(glob.glob('sel_test/*/*_07.png'))
    # files = sorted(glob.glob('london_test_71_frames/*_30.png'))
    # files = sorted(glob.glob('london_test_template/*_07.png'))
    # files = sorted(glob.glob('london_test300_15_frames/*_07.png'))
    files = sorted(glob.glob('london_test300_15_frames_rebuttal/*_07.png'))

    num_frame, cidx, step = 15, 7, 0.5
    # num_frame, cidx, step = 71, 30, 0.1
    vec = get_pano_vec((128, 256))
    # for file in tqdm.tqdm(files):
    for vid in vids:
        file = files[vid]
        rank = file.split('/')[1]
        fid = os.path.basename(file).split('_')[0]

        # imgs = [np.array(Image.open(f'london_15_frames/{fid}_%02d.png' % i)) for i in range(num_frame)]
        # imgs = [np.array(Image.open(f'london_15_frames_xiaohu_train/{fid}_%02d.png' % i)) for i in range(num_frame)]
        # imgs = [np.array(Image.open(f'sel_test/{rank}/{fid}_%02d.png' % i)) for i in range(num_frame)]
        # imgs = [np.array(Image.open(f'london_test_71_frames/{fid}_%02d.png' % i)) for i in range(num_frame)]
        # imgs = [np.array(Image.open(f'london_test300_15_frames/{fid}_%02d.png' % i)) for i in range(num_frame)]
        imgs = [np.array(Image.open(f'london_test300_15_frames_rebuttal/{fid}_%02d.png' % i)) for i in range(num_frame)]

        deps = np.stack([decode_dep(item[::2,:512:2,:3]) for item in imgs])
        sems = np.stack([255-item[::2,:512:2,3] for item in imgs])
        org_rgbs = np.stack([item[:,512:,:3] for item in imgs])
        rgbs = np.stack([downsample_half_rgb(item[:,512:,:3]) for item in imgs])
        cons = np.stack([item[::2,512::2,3] for item in imgs])

        coord = np.stack([deps[cidx]] * 3, axis=-1) * vec
        pc = np.dstack([coord, rgbs[cidx], sems[cidx, ..., np.newaxis], cons[cidx, ..., np.newaxis]]).reshape((-1, 8))

        # for i in [8,6,9,5,10,4,11,3,12,2,13,1,14,0]:
        # for i in [34,36,33,37,32,38,31,39,30,40,29,41,28,42,27,43,26,44,25,45,24,46,23,47,22,48,21,49,20,50,19,51,18,52,17,53,16,54,15,55,14,56,13,57,12,58,11,59,10,60,9,61,8,62,7,63,6,64,5,65,4,66,3,67,2,68,1,69,0,70]:
        # li1 = [40,30,45,25,50,20,55,15,60,10,65,5,70,0]
        # li2 = [34,36,33,37,32,38,31,39,29,41,28,42,27,43,26,44,24,46,23,47,22,48,21,49,19,51,18,52,17,53,16,54,14,56,13,57,12,58,11,59,9,61,8,62,7,63,6,64,4,66,3,67,2,68,1,69]
        # for li, err in zip([li1, li2], [0.005, 100]):
        # for li, err in zip([[6,8,5,9,4,10,3,11,2,12,1,13,0,14]], [0.005]):
        for li, err in zip([[8,6,9,5,10,4,11,3,12,2,13,1,14,0]], [0.005]):
            for i in li:
                delta = (i - cidx) * step
                # to_move = ~(pc[:,-2].astype(np.uint8) == 10)
                to_move = np.sqrt((pc[:, :3] ** 2).sum(axis=-1)) < 900

                shangxia = 64 * (1-np.sin(np.pi * i / 14))/30.0
                pc[to_move,1] += shangxia
                pc[to_move,2] -= delta

                syn = point_cloud_to_panorama(pc, panorama_size=vec.shape[:2][::-1]) # RGB+sem+con+depth+index

                mask_sem = compare(syn[..., 3], sems[i], 'sem')
                mask_dep = compare(syn[..., 5], deps[i], 'dep', err=err)
                mask = mask_dep & mask_sem

                coord = np.stack([deps[i]] * 3, axis=-1) * vec
                pc_to_add = np.dstack([coord, rgbs[i], sems[i, ..., np.newaxis], cons[i, ..., np.newaxis]]).reshape((-1, 8))
                pc_to_add = pc_to_add[~mask.flatten()]
                # pc_to_add_to_move = ~(pc_to_add[:,-2].astype(np.uint8) == 10)
                pc_to_add_to_move = np.sqrt((pc_to_add[:, :3] ** 2).sum(axis=-1)) < 900

                pc_to_add[pc_to_add_to_move,1] -= shangxia
                pc_to_add[pc_to_add_to_move,2] += delta

                pc[to_move,1] -= shangxia
                pc[to_move,2] += delta
                pc = np.vstack([pc, pc_to_add])
            
            print(pc.shape)
        
        syn_rgbs, syn_sems, syn_idxs = [None] * num_frame, [None] * num_frame, [None] * num_frame
        for i in range(num_frame):
            delta = (i - cidx) * step
            # to_move = ~(pc[:,-1].astype(np.uint8) == 10)
            to_move = np.sqrt((pc[:, :3] ** 2).sum(axis=-1)) < 900
            shangxia = 64 * (1-np.sin(np.pi * i / 14))/30.0
            pc[to_move,1] += shangxia
            pc[to_move,2] -= delta
            syn = point_cloud_to_panorama(pc, panorama_size=vec.shape[:2][::-1])
            syn_rgbs[i] = Image.fromarray(syn[...,:3].astype(np.uint8))
            # syn_rgbs[i] = Image.fromarray(np.vstack([hehe_rgbs[i], syn[...,:3].astype(np.uint8)]))
            syn_sems[i] = Image.fromarray(syn[...,3].astype(np.uint8))
            syn_sems[i].putpalette(palette)
            syn_idxs[i] = Image.fromarray(encode_index(syn[...,6]))
            pc[to_move,1] -= shangxia
            pc[to_move,2] += delta

        idx = np.stack([decode_index(item) for item in syn_idxs])
        point_in_use, new_idx = np.unique(idx, return_inverse=True)
        pc = pc[point_in_use]
        new_idx = new_idx.reshape(idx.shape)

        # print(pc[:,:3].min(axis=0))
        # print(pc[:,:3].max(axis=0))
        # input()
        # continue
        # print(pc.shape[0]/256/128, end='\r')

        print(pc.shape)

        # np.savez_compressed(f'london_15_frames_xiaohu_train_half_pc_npz/{fid}.npz',
        # np.savez_compressed(f'london_seltest_15_frames_half_pc_npz/{fid}.npz',
        # np.savez_compressed(f'london_test_71_frames_half_pc_npz/{fid}.npz',
        # np.savez_compressed(f'london_seltest_15_frames_half_pc_npz_inv/{fid}.npz',
        # np.savez_compressed(f'london_test300_15_frames_half_pc_npz_inv/{fid}.npz',
        np.savez_compressed(f'london_test300_15_frames_rebuttal_half_pc_npz/{fid}.npz',
            coord=pc[:,:3].astype(np.float),
            rgb=pc[:,3:6].astype(np.uint8),
            sem=pc[:,6].astype(np.uint8),
            con=pc[:,7].astype(np.uint8),
            idx=new_idx,
        )
        continue

        # h5_file = h5py.File(f'/run/user/1314886183/lzy/london_15_frames_xiaohu_train_half_pc_h5/{fid}.h5', 'w')
        # h5_file.create_dataset('coord', data=pc[:,:3].astype(np.float))
        # h5_file.create_dataset('rgb', data=pc[:,3:6].astype(np.uint8))
        # h5_file.create_dataset('sem', data=pc[:,6].astype(np.uint8))
        # h5_file.create_dataset('idx', data=new_idx)
        # h5_file.create_dataset('org_rgbs', data=org_rgbs)
        # h5_file.create_dataset('con', data=pc[:,7].astype(np.uint8))
        # h5_file.close()
        # input('pause')
        # continue

        syn_rgbs[0].save(f'syn_gif/{fid}_rgb.gif',save_all=True,append_images=syn_rgbs[1:],duration=200,loop=0)
        syn_sems[0].save(f'syn_gif/{fid}_sem.gif',save_all=True,append_images=syn_sems[1:],duration=200,loop=0)
        syn_idxs[0].save(f'syn_gif/{fid}_idx.gif',save_all=True,append_images=syn_idxs[1:],duration=200,loop=0)

        # write_point_cloud(f'syn_pc_inv/{fid}_rgb.txt', pc[:, :6])
        # write_point_cloud(f'syn_pc_inv/{fid}_sem.txt', np.hstack([
        #     pc[:, :3],
        #     palette_np[pc[:, 6].astype(np.uint8)]
        # ]))
        input('pause')
        continue
