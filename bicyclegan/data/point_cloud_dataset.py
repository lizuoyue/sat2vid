import os.path
import glob
import numpy as np
from data.base_dataset import BaseDataset, get_params, get_transform
from data.image_folder import make_dataset
from PIL import Image
import random
import torch
torch.set_default_dtype(torch.float32)

def get_transformation_matrix(waypoint):
    x, y, z, pitch, yaw, roll = waypoint
    cy = np.cos(np.deg2rad(yaw))
    sy = np.sin(np.deg2rad(yaw))
    cr = np.cos(np.deg2rad(roll))
    sr = np.sin(np.deg2rad(roll))
    cp = np.cos(np.deg2rad(pitch))
    sp = np.sin(np.deg2rad(pitch))
    return np.array([
        [cp * cy, cy * sp * sr - sy * cr, -cy * sp * cr - sy * sr,   x],
        [cp * sy, sy * sp * sr + cy * cr, -sy * sp * cr + cy * sr,   y],
        [     sp,               -cp * sr,                 cp * cr,   z],
        [    0.0,                    0.0,                     0.0, 1.0],
    ], np.float32)


class PointCloudDataset(BaseDataset):
    """A dataset class for point cloud dataset.

    It assumes that the directory '/path/to/data/train' contains image pairs in the form of {A,B}.
    During test time, you need to prepare a directory '/path/to/data/test'.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.info = {
            os.path.basename(npz).replace('.npz', ''): \
            dict(np.load(npz, allow_pickle=True)) \
        for npz in glob.glob(opt.dataroot + '/*.npz')}
        self.img_path = os.path.join(opt.dataroot, opt.phase)
        assert(self.opt.load_size >= self.opt.crop_size)   # crop_size should be smaller than the size of loaded image
        self.input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        self.output_nc = self.opt.input_nc if self.opt.direction == 'BtoA' else self.opt.output_nc

        assert(opt.max_len_path <= 15 and opt.max_len_path % 2 == 1)
        self.num_frames = opt.num_frames
        self.max_len_path = opt.max_len_path
        self.idx_list = list(range(opt.max_len_path))
        self.cidx = int((opt.max_len_path-1)/2)
        self.idx_list = self.idx_list[:self.cidx] + self.idx_list[self.cidx+1:]
        self.max_depth = 1000

        self.weathers = 'ClearNoon,CloudyNoon,ClearSunset'.split(',')

        self.palette = np.zeros((256,3), np.uint8)
        self.palette[:14] = np.array([
            (0, 0, 0),
            (70, 70, 70),
            (190, 153, 153),
            (250, 170, 160),
            (220, 20, 60),
            (153, 153, 153),
            (157, 234, 50),
            (128, 64, 128),
            (244, 35, 232),
            (107, 142, 35),
            (0, 0, 142),
            (102, 102, 156),
            (220, 220, 0),
            (70, 130, 180)
        ])

        self.class_new = np.array([6,1,1,6,6,6,2,2,3,4,6,1,6,5]).astype(np.int)
        self.palette_new = np.zeros((256,3), np.uint8)
        self.palette_new[:7] = np.array([
            (60, 60, 60), # reserved
            (80, 80, 80),
            (128, 64, 128),
            (244, 35, 232),
            (107, 142, 35),
            (70, 130, 180),
            (0, 0, 0),
        ])

        self.half_rsl = opt.half_rsl

        self.paths = []
        for weather in self.weathers:
            for sub in sorted(list(self.info.keys())):
                # if sub != 'Town10HD':
                #     continue
                for path in self.info[sub][f'{opt.phase}_paths']:
                    if type(path) == list:
                        if len(path) >= self.max_len_path:
                            self.paths.append((sub, weather, path[:self.max_len_path]))
                    else:
                        if path.size >= self.max_len_path:
                            self.paths.append((sub, weather, list(path[:self.max_len_path].flatten())))
        # for sub, weather, _ in self.paths:
        #     print(sub, weather)
        # input('Pause ...')
        return
    
    def decode_dep(self, im):
        im = im.transpose([2,0,1]).astype(np.float32)
        im = (im[0] + im[1]*256 + im[2]*256*256) / (256**3-1) * self.max_depth
        return torch.from_numpy(im)

    def decode_sem(self, im):
        im = self.palette[im.flatten()].reshape(im.shape[:2]+(3,))
        return torch.from_numpy(im)
    
    def decode_sem_idx(self, im):
        sem_idx = self.class_new[im.flatten()].reshape(im.shape[:2]+(1,))
        sem = self.palette_new[sem_idx].reshape(im.shape[:2]+(3,))
        return torch.from_numpy(sem), torch.from_numpy(sem_idx)

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor) - - an image in the input domain
            B (tensor) - - its corresponding image in the target domain
            A_paths (str) - - image paths
            B_paths (str) - - image paths (same as A_paths)
        """
        # read a image given a random integer index
        sub, weather, path = self.paths[index] # 'TownXX', [1234,1235,1236,...,XXXX]
        waypoints = self.info[sub]['waypoints']
        waypoints[:,2] += 2.0 # floating at 2.0 m
        li = [self.cidx]
        if self.num_frames > 1:
            li += sorted(random.sample(self.idx_list, self.num_frames-1))
        c_mat = torch.from_numpy(get_transformation_matrix(waypoints[path[self.cidx]]))
        dep, sem, rgb, mat, sem_idx = [], [], [], [], []
        for item in li:
            waypoint = waypoints[path[item]]
            mat.append(torch.from_numpy(get_transformation_matrix(waypoint)))
            img_AB = Image.open(self.img_path + '/%s_%s_%05d.png' % (sub, weather, path[item]))
            w, h = img_AB.size
            half_w = int(w/2)
            img_A = np.array(img_AB.crop((     0, 0, half_w, h)))
            img_B = np.array(img_AB.crop((half_w, 0,      w, h)), np.float32)
            if self.half_rsl:
                s = np.random.randint(2)
                img_A = img_A[s::2,s::2]
                img_B = img_B[s::2,s::2]
            dep.append(self.decode_dep(img_A[..., :3]).float())
            if self.half_rsl:
                res1, res2 = self.decode_sem_idx(img_A[..., 3:])
                sem.append(res1.float())
                sem_idx.append(res2)
            else:
                sem.append(self.decode_sem(img_A[..., 3:]).float())
                sem_idx.append(torch.from_numpy(img_A[..., 3:].astype(np.int)))
            rgb.append(torch.from_numpy(img_B[..., :3]))
        dep, sem, rgb, mat, sem_idx = torch.stack(dep), torch.stack(sem)/127.5-1, torch.stack(rgb)/127.5-1, torch.stack(mat), torch.stack(sem_idx)
        return {'idx': index, 'dep': dep, 'sem': sem, 'rgb': rgb, 'mat': mat, 'c_mat': c_mat, 'sem_idx': sem_idx}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.paths)
    
    
