import os, glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw
import cv2, tqdm
import matplotlib.pyplot as plt
import frnn

def encode_d(d):
    d = d.astype(np.float64)
    d = np.clip(d, 0, 1000)
    d /= 1000.0
    d *= (256**3-1)
    d = d.astype(np.int)
    r = d % 256
    g = (d // 256) % 256
    b = (d // 65536) % 256
    return np.dstack([r, g, b]).astype(np.uint8)

def decode_dep(dep_file):
    im = np.array(Image.open(dep_file).convert('RGB')).astype(np.float).transpose([2,0,1])
    im = (im[0] + im[1]*256 + im[2]*256*256) / (256**3-1) * 1000
    return im

def decode_inv_dep(dep_file):
    im = np.array(Image.open(dep_file).convert('RGB')).astype(np.float).transpose([2,0,1])
    im = (im[0] + im[1]*256 + im[2]*256*256) / (256**3-1) * 10000
    im = np.clip(im, 1e-3, 10000)
    return im

def pano_dis_to_point_cloud(dis,
                            min_max_lat=[-np.pi/2, np.pi/2]):
    """
    Input:
        dis:                    (height, width)
        min_max_lat:            [min_lat, max_lat]

    # Local point cloud is generated
    # X  Right->Right
    # Y  Inside->Down
    # Z  Top->Inside

    Output:
        point_cloud:            (n, 3), each row float coordinates

    """
    dis = np.array(dis).astype(np.float)
    nrow, ncol = dis.shape
    min_lat, max_lat = min_max_lat

    x, y = np.meshgrid(np.arange(0, ncol)+0.5, np.arange(0, nrow)+0.5)
    lon = x / ncol * 2.0 * np.pi - np.pi
    lat = (1.0 - y / nrow) * (max_lat - min_lat) + min_lat
    
    vd = np.cos(lat)
    vx = vd * np.sin(lon)
    vy = vd * np.cos(lon)
    vz = np.sin(lat)

    # v = np.dstack([vx, vy, vz]).reshape((-1, 3))
    v = np.dstack([vx, -vz, vy]).reshape((-1, 3))

    pc = (v.T * dis.reshape((-1))).T
    return pc

def get_panorama_vec(panorama_size=[512, 256],
                     min_max_lat=[-np.pi/2, np.pi/2]):
    """
    Input:
        voxel_occupy:        (batch_size, nz, ny, nx)
        center_loc:            (batch_size * n_sample, 3), last dim [-1 ~ 1, -1 ~ 1, -1 ~ 1]
        direction:            (batch_size * n_sample) # rad
        panorama_size:        [width, height]
        min_max_lat:        [min_lat, max_lat]

    Output:
        panorama:            (batch_size, n_sample, height, width)

    """

    ncol, nrow = panorama_size
    min_lat, max_lat = min_max_lat

    x = (torch.arange(0, ncol, 1, dtype=torch.float32) + 0.5).cuda() / ncol
    y = (torch.arange(0, nrow, 1, dtype=torch.float32) + 0.5).cuda() / nrow
    lon = x * 2.0 * np.pi - np.pi
    lat = (1.0 - y) * (max_lat - min_lat) + min_lat

    sin_lon = torch.sin(lon).view(1, ncol).expand(nrow, ncol)
    cos_lon = torch.cos(lon).view(1, ncol).expand(nrow, ncol)
    sin_lat = torch.sin(lat).view(nrow, 1).expand(nrow, ncol)
    cos_lat = torch.cos(lat).view(nrow, 1).expand(nrow, ncol)

    # Compute the unit vector of each pixel
    vx = cos_lat.mul(sin_lon)
    vy = cos_lat.mul(cos_lon)
    vz = sin_lat

    return torch.stack([vx, -vz, vy], dim=2)

def write_point_cloud(filename, arr):
    with open(filename, 'w') as f:
        for line in arr:
            coord = ('%.6f;'*3) % tuple(list(line[:3]))
            rgb = ('%d;'*3) % tuple(list(line[3:]))
            f.write(coord + rgb[:-1] + '\n')

def read_dep(dep1_file, dep2_file, decode_func):
    dep1 = decode_func(dep1_file)
    dep2 = decode_func(dep2_file)
    dep2 = np.concatenate([dep2[:, 512:], dep2[:, :512]], axis=1)

    weight2 = (np.cos(np.linspace(0, np.pi * 2, 1024)) + 1)/2
    weight1 = 1 - weight2
    weight1 = np.stack([weight1]*512)
    weight2 = np.stack([weight2]*512)
    return dep1 * weight1 + dep2 * weight2





if __name__ == '__main__':

    one_hot = torch.eye(20).float().cuda()

    img_files = sorted(glob.glob('rgb_1024_512_all/*.png'))
    sem_files = sorted(glob.glob('sem_1024_512_all/*.png'))
    dep_files = sorted(glob.glob('depth_frame_512_256_all/*_7.png'))
    assert(len(img_files) == len(sem_files) == len(dep_files))

    vec_256 = get_panorama_vec(panorama_size=[512, 256])
    vec_512 = get_panorama_vec(panorama_size=[1024, 512])

    lat = np.arcsin(-vec_512[:,:,1].cpu().numpy())
    undist_55 = 1.0 / np.cos(0.55 * lat)
    undist_95 = 1.0 / np.cos(0.95 * lat)
    undist_ratio = undist_95 * (lat >= 0) + undist_55 * (lat <= 0)

    for fid, (img_file, sem_file, dep_file) in tqdm.tqdm(list(enumerate(list(zip(img_files, sem_files, dep_files))))):

        if fid < 1373:
            continue

        basename = os.path.basename(img_file)
        palette = np.array(Image.open(sem_file).getpalette()).reshape((256, 3)).astype(np.uint8)

        img = np.array(Image.open(img_file))
        sem = np.array(Image.open(sem_file))
        dep = cv2.resize(decode_dep(dep_file), (1024, 512), interpolation=cv2.INTER_CUBIC)

        c_color = torch.from_numpy(np.array(img)).float().view(-1, 3).cuda()
        c_seman = torch.from_numpy(np.array(sem)).long().view(-1).cuda()
        c_dist = torch.from_numpy(dep).float().cuda()

        c_sky_color = c_color[c_seman==10, :]

        c_coord = vec_512 * c_dist.unsqueeze(-1).expand(512,1024,3)
        c_coord = c_coord.view(1, -1, 3)

        c_sky_coord = c_coord[0, c_seman==10, :] # N, 3
        # c_sky_ratio = -1000.0 / c_sky_coord[...,1]
        c_sky_ratio = 1000.0 / (c_sky_coord ** 2).sum(dim=-1)

        c_sky_coord = c_sky_coord * c_sky_ratio.unsqueeze(-1).expand(c_sky_coord.size(0), 3)
        c_sky_coord = c_sky_coord.unsqueeze(0)

        # first time there is no cached grid
        _, _, _, grid = frnn.frnn_grid_points(
            c_coord, c_coord, None, None, K=32, r=10000.0, grid=None, return_nn=False, return_sorted=True
        )
        
        if c_sky_coord.size(1) > 0:
            _, _, _, grid_sky = frnn.frnn_grid_points(
                c_sky_coord, c_sky_coord, None, None, K=32, r=10000.0, grid=None, return_nn=False, return_sorted=True
            )
        else:
            print(img_file)

        colors, semans, confs, coords = [], [], [], []
        encoded_depth = []
        for n, delta in enumerate(list(np.linspace(-3.5, 3.5, 15))):
            if n == 7:
                colors.append(c_color)
                semans.append(c_seman)
                confs.append(confs[-1])
                coords.append(c_coord[0])
                encoded_depth.append(encode_d(dep))
                continue

            f_dist = decode_dep('depth_frame_512_256_all/' + basename.replace('.png', f'_{n}.png'))
            f_dist = cv2.resize(f_dist, (1024, 512), interpolation=cv2.INTER_CUBIC)

            encoded_depth.append(encode_d(f_dist))

            f_dist = torch.from_numpy(f_dist).float().cuda()
            f_coord = (vec_512 * f_dist.unsqueeze(-1).expand(512,1024,3)).view(1, -1, 3)
            f_coord[..., 2] += delta

            dists, idxs, _, _ = frnn.frnn_grid_points(
                f_coord, c_coord, None, None, K=32, r=10000.0, grid=grid, return_nn=False, return_sorted=True
            )

            conf = (1 - torch.clamp(dists[0, :, 0], 0, 0.5) / 0.5) * 255
            weight = 1 / (dists[0].unsqueeze(-1) ** 2)
            weight /= torch.sum(weight, dim=1, keepdim=True)
            
            count = one_hot[c_seman[idxs.view(-1)]].view(512*1024,32,20)
            label = torch.sum(count * weight.expand(512*1024,32,20), dim=1).argmax(dim=-1)

            rgb = c_color[idxs.view(-1)].view(512*1024,32,3)
            rgb = torch.sum(rgb * weight.expand(512*1024,32,3), dim=1)

            if c_sky_coord.size(1) > 0:
                f_sky_coord = f_coord[0, label == 10, :] # N, 3
                sky_num = f_sky_coord.size(0)
                f_sky_coord[:, 2] -= delta
                # f_sky_ratio = -1000.0 / f_sky_coord[...,1]
                f_sky_ratio = 1000.0 / (f_sky_coord ** 2).sum(dim=-1)
                f_sky_coord = f_sky_coord * f_sky_ratio.unsqueeze(-1).expand(f_sky_coord.size(0), 3)
                f_sky_coord = f_sky_coord.unsqueeze(0)

                if f_sky_coord.size(1) > 0:

                    sky_dists, sky_idxs, _, _ = frnn.frnn_grid_points(
                        f_sky_coord, c_sky_coord, None, None, K=32, r=10000.0, grid=grid_sky, return_nn=False, return_sorted=True
                    )

                    sky_conf = (1 - torch.clamp(sky_dists[0, :, 0], 0, 0.5) / 0.5) * 255
                    sky_weight = 1 / (sky_dists[0].unsqueeze(-1) ** 2)
                    sky_weight /= torch.sum(sky_weight, dim=1, keepdim=True)

                    sky_rgb = c_sky_color[sky_idxs.view(-1)].view(sky_num,32,3)
                    sky_rgb = torch.sum(sky_rgb * sky_weight.expand(sky_num,32,3), dim=1)

                    rgb[label == 10] = sky_rgb
                    conf[label == 10] = sky_conf
                
                else:
                    print('F', img_file)


            colors.append(rgb)
            semans.append(label)
            coords.append(f_coord[0])
            confs.append(conf.int())

        colors = torch.clamp(torch.stack(colors), 0, 255).cpu().numpy().astype(np.uint8).reshape((15, 512, 1024, 3))
        semans = torch.stack(semans).cpu().numpy().astype(np.uint8).reshape((15, 512, 1024))
        coords = torch.stack(coords).cpu().numpy().reshape((-1, 3))
        confs = torch.stack(confs).cpu().numpy().astype(np.uint8).reshape((15, 512, 1024))

        colors_gif = [Image.fromarray(item) for item in colors]
        semans_gif = palette[semans.flatten()].reshape((15, 512, 1024, 3))

        # write_point_cloud(
        #     f'hehe/{basename}'.replace('.png', '_rgb.txt'),
        #     np.hstack([coords, colors.reshape((-1, 3))])
        # )
        # write_point_cloud(
        #     f'hehe/{basename}'.replace('.png', '_sem.txt'),
        #     np.hstack([coords, semans_gif.reshape((-1, 3))])
        # )

        semans_gif = [Image.fromarray(item) for item in semans_gif]

        colors_gif[0].save(f'london_15_vid2vid_frames_gif/{basename}'.replace('.png','_rgb.gif'), save_all=True, append_images=colors_gif[1:], loop=0, duration=250)
        semans_gif[0].save(f'london_15_vid2vid_frames_gif/{basename}'.replace('.png','_sem.gif'), save_all=True, append_images=semans_gif[1:], loop=0, duration=250)
        # input('press enter to continue')
        # continue

        for i in range(15):
            # A123 = np.array(Image.open('depth_frame_1024_512_all/' + basename.replace('.png', f'_{i}.png')))
            A123 = encoded_depth[i]
            A4 = semans[i].reshape((512,1024,1))
            A123[A4[...,0] == 10] = [255,255,255] 
            A4 = 255 - A4
            B123 = colors[i]
            B4 = confs[i, ..., np.newaxis] * 0 + 255
            to_save = np.hstack([np.dstack([A123, A4]), np.dstack([B123, B4])])
            Image.fromarray(to_save).save(('london_15_vid2vid_frames/' + basename).replace('.png', '_%02d.png' % i))
