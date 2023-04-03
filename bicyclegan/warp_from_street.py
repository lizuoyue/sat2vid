import os, glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import tqdm, cv2

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

if __name__ == '__main__':

    scale = 2

    palette = '[128,64,128,244,35,232,70,70,70,102,102,156,190,153,153,153,153,153,250,170,30,220,220,0,107,142,35,152,251,152,70,130,180,220,20,60,255,0,0,0,0,142,0,0,70,0,60,100,0,80,100,0,0,230,119,11,32]'
    palette = eval(palette) + [0] * (256*3-len(palette))
    palette_np = np.array(palette).reshape((-1, 3)).astype(np.uint8)

    files = sorted(glob.glob('london_15_frames/*_07.png'))
    vec = torch.from_numpy(get_pano_vec((256, 512))).cuda()
    for file in tqdm.tqdm(files):
        fid = os.path.basename(file).split('_')[0]
        imgs = [np.array(Image.open(f'london_15_frames/{fid}_%02d.png' % i)) for i in range(15)]
        deps = np.stack([decode_dep(item[:,:512,:3]) for item in imgs])
        sems = np.stack([255-item[:,:512,3] for item in imgs])
        rgbs = np.stack([item[:,512:,:3] for item in imgs])

        # for i in [8,6,9,5,10,4,11,3,12,2,13,1,14,0]:
        #     delta = (i - 7) * 0.5
        #     to_move = ~(pc[:,-1].astype(np.uint8) == 10)

        #     pc[to_move,2] -= delta
        #     syn = point_cloud_to_panorama(pc, panorama_size=vec.shape[:2][::-1]) # RGB+sem+depth+index

        #     mask_dep = compare(syn[..., 4], deps[i], 'dep')
        #     mask_sem = compare(syn[..., 3], sems[i], 'sem')
        #     mask = mask_dep & mask_sem

        #     coord = np.stack([deps[i]] * 3, axis=-1) * vec
        #     pc_to_add = np.dstack([coord, rgbs[i], sems[i, ..., np.newaxis]]).reshape((-1, 7))
        #     pc_to_add = pc_to_add[~mask.flatten()]
        #     pc_to_add_to_move = ~(pc_to_add[:,-1].astype(np.uint8) == 10)
        #     pc_to_add[pc_to_add_to_move,2] += delta

        #     pc[to_move,2] += delta
        #     pc = np.vstack([pc, pc_to_add])

        gif, gif_gt, fake_sates = [], [], []
        for seq, (rgb, dep, sem) in enumerate(zip(rgbs, deps, sems)):

            c_color = torch.from_numpy(rgb).float().view(-1, 3).cuda()
            c_seman = torch.from_numpy(sem).long().view(-1).cuda()
            c_dist = torch.from_numpy(dep).float().cuda()

            c_sky_color = c_color[c_seman==10, :]

            c_coord = vec * c_dist.unsqueeze(-1).expand(256,512,3)
            c_coord = c_coord.view(1, -1, 3)

            c_coord = c_coord[0, c_seman!=10, :] # N, 3
            c_color = c_color[c_seman!=10, :]

            order = torch.argsort(-c_coord[:,1])
            c_coord = c_coord[order]
            c_color = c_color[order]
            c_coord = torch.round(c_coord * scale)
            c_coord = c_coord + torch.Tensor([128,0,128]).cuda()

            c_coord = c_coord.int().cpu().numpy().astype(np.int)
            c_color = c_color.cpu().numpy().astype(np.uint8)

            w, h = 256, 256
            fake_sate = np.ones((w*h, 3), np.uint8) * 0
            mask = np.ones((w*h), np.uint8)
            xx = c_coord[:,0]
            yy = c_coord[:,2]
            valid = (0 <= xx) & (xx < w) & (0 <= yy) & (yy < h)
            idx = xx[valid] * w + yy[valid]
            fake_sate[idx] = c_color[valid]
            mask[idx] = 0

            fake_sate = fake_sate.reshape((h, w, 3))
            mask = mask.reshape((h, w))
            fake_sate = cv2.inpaint(fake_sate,mask,3,cv2.INPAINT_TELEA)

            fake_sates.append(Image.fromarray(fake_sate))

            fake_sate = fake_sate.reshape((1, h, w, 3)).transpose([0,3,1,2])
            fake_sate = torch.from_numpy(fake_sate).cuda().float()

            c_coord = vec * c_dist.unsqueeze(-1).expand(256,512,3) * scale
            c_coord = (c_coord + torch.Tensor([128,0,128]).cuda()) / 128 - 1
            c_coord = c_coord.unsqueeze(0)[..., [2,0]].float()

            ground = torch.nn.functional.grid_sample(fake_sate, c_coord, mode='bilinear', padding_mode='zeros', align_corners=None)
            ground = ground.cpu().numpy().astype(np.uint8)[0].transpose([1,2,0])
            ground[(c_seman==10).cpu().numpy().reshape((256,512))] = [255,255,255]

            gif.append(Image.fromarray(ground))
            gif_gt.append(Image.fromarray(rgb))

        gif[0].save('1.gif',loop=0,duration=250,append_images=gif[1:],save_all=True)
        gif_gt[0].save('2.gif',loop=0,duration=250,append_images=gif_gt[1:],save_all=True)
        fake_sates[0].save('3.gif',loop=0,duration=250,append_images=fake_sates[1:],save_all=True)
        input('press')

        # for i in range(15):
        #     gif[i].save(f'warp_from_street/{fid}_{i:02d}.png')
