import os.path
import glob
import numpy as np
from data.base_dataset import BaseDataset, get_params, get_transform
from data.image_folder import make_dataset
from PIL import Image
import random, h5py
# import torch
# torch.set_default_dtype(torch.float32)

to_simple_label = [2, 3, 1, 1, 1, 6, 6, 6, 4, 4, 5]
to_simple_label += [6] * (256-len(to_simple_label))
to_simple_label = np.array(to_simple_label).astype(np.uint8)

def coord2sem(coord):
    dis = np.sqrt((coord ** 2).sum(axis=-1))

    is_sky = dis > 145
    is_road = coord[:,2] < -2.9
    
    sem = np.ones(dis.shape, np.uint8)
    sem[is_sky] = 5
    sem[is_road] = 2
    
    return sem


class LondonPCDataset(BaseDataset):
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

        self.file_paths = []
        for phase in opt.phase.split(','):
            if phase.endswith('_h5'):
                self.file_paths += sorted(glob.glob(os.path.join(opt.dataroot, phase, '*.h5')))
            else:
                self.file_paths += sorted(glob.glob(os.path.join(opt.dataroot, phase, '*.npz')))

        self.input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        self.output_nc = self.opt.input_nc if self.opt.direction == 'BtoA' else self.opt.output_nc

        self.palette = np.zeros((256,3), np.uint8)
        self.palette[:6] = np.array([
            (70, 70, 70),   # 0 reserved for left building
            (70, 70, 70),   # 1 (right) building
            (128, 64, 128), # 2 road
            (244, 35, 232), # 3 sidewalk
            (107, 142, 35), # 4 vegetation
            (70, 130, 180), # 5 sky
                            # 6 other obj
        ])

        self.manual_semantics = opt.manual_semantics

        if opt.dataroot.endswith('half_pc'):
            self.max_pc_len = 300000
        else:
            self.max_pc_len = 700000
        
        self.final_upsample = opt.final_upsample

        return

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

        file_path = self.file_paths[index]

        if file_path.endswith('.h5'):
            fid = os.path.basename(file_path).replace('.h5', '')
            h5_file, d = h5py.File(file_path, 'r'), {}
            for key in h5_file.keys():
                d[key] = h5_file[key][()]
        else:
            fid = os.path.basename(file_path).replace('.npz', '')
            d = dict(np.load(file_path))
        assert(d['coord'].shape[0] == d['rgb'].shape[0] == d['sem'].shape[0] == (d['idx'].max()+1))

        # Coord
        dis = np.sqrt((d['coord'] ** 2).sum(axis=-1))
        ratio = np.clip(dis, 0, 150.0) / dis
        d['coord'] = (d['coord'].T * ratio).T
        d['coord'][:,1], d['coord'][:,2] = d['coord'][:,2].copy(), -d['coord'][:,1].copy()

        # Semantics
        if self.manual_semantics:
            d['sem_idx'] = coord2sem(d['coord'])
        else:
            d['sem_idx'] = to_simple_label[d['sem']].astype(np.uint8)

        d['sem'] = self.palette[d['sem_idx']].astype(np.uint8)

        # Padding
        d['pc_len'] = d['coord'].shape[0]
        padding_len = self.max_pc_len - d['pc_len']
        if padding_len < 0:
            print(fid)
            assert(False)

        d['coord'] = np.concatenate([d['coord'], np.zeros((padding_len, 3), d['coord'].dtype)])
        d['rgb'] = np.concatenate([d['rgb'], np.zeros((padding_len, 3), d['rgb'].dtype)]).astype(np.float32)/127.5-1
        d['sem'] = np.concatenate([d['sem'], np.zeros((padding_len, 3), d['sem'].dtype)]).astype(np.float32)/127.5-1
        d['sem_idx'] = np.concatenate([d['sem_idx'], np.zeros((padding_len), d['sem_idx'].dtype)])

        # Center RGB
        center_idx = d['idx'][d['idx'].shape[0] // 2]
        d['center_rgb'] = d['rgb'][center_idx.flatten()].reshape(center_idx.shape + (3,))
        d['data_idx'] = index

        d['con'] = 0

        if file_path.endswith('.h5'):
            d['org_rgbs'] = d['org_rgbs'].astype(np.float32)/127.5-1
            if self.final_upsample // 10 == 2:
                d['warp_sate'] = []
                pre = file_path.replace(f'{fid}.h5', '')
                for seq in range(15):
                    d['warp_sate'].append(np.array(
                        Image.open(f'{pre}/warp_from_sate/{fid}_{seq:02d}.png')
                    ).astype(np.float32)/127.5-1)
                d['warp_sate'] = np.stack(d['warp_sate'])
        else:
            d['org_rgbs'] = 0
            # d['rgb_img'] = []
            # for seq in range(15):
            #     temp = Image.open(f'/home/lzq/lzy/BicycleGAN/london_15_frames/{fid}_{seq:02d}.png')
            #     d['rgb_img'].append(np.array(temp)[:,512:,:3].astype(np.float32)/127.5-1)
            # d['rgb_img'] = np.stack(d['rgb_img'])

        return d

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.file_paths)



class Option(object):
    def __init__(self):
        self.phase = 'train_midas_h5'
        self.dataroot = 'datasets/london_half_pc'
        self.input_nc = 3
        self.output_nc = 3
        self.direction = 'AtoB'
        self.manual_semantics = False
        return

if __name__ == '__main__':
    opt = Option()
    dataset = LondonPCDataset(opt)
    for i in range(100):
        dataset.__getitem__(i)
