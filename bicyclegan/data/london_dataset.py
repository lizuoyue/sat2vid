import os.path
import glob
import numpy as np
from data.base_dataset import BaseDataset, get_params, get_transform
from data.image_folder import make_dataset
from PIL import Image
import random
import torch
torch.set_default_dtype(torch.float32)

to_simple_label = [2, 3, 1, 1, 1, 6, 6, 6, 4, 4, 5]
to_simple_label += [6] * (256-len(to_simple_label))
to_simple_label = np.array(to_simple_label).astype(np.uint8)

def get_transformation_matrix(idx=7, ratio=0.5):
    mat = np.eye(4)
    mat[0,3] = float(idx - 7) * ratio
    return torch.from_numpy(mat)

class LondonDataset(BaseDataset):
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

        self.img_paths = []
        for phase in opt.phase.split(','):
            self.img_paths += [
                item.replace('_00.png', '')
                for item in sorted(
                    glob.glob(os.path.join(opt.dataroot, phase, '*_00.png'))
                )
            ]
        self.num_frames = opt.num_frames
        self.max_len_path = opt.max_len_path
        half = opt.max_len_path//2
        self.idx_list = list(range(15))
        self.cidx = 7
        self.idx_list = self.idx_list[self.cidx-half:self.cidx] + self.idx_list[self.cidx+1:self.cidx+1+half]

        assert(self.opt.load_size >= self.opt.crop_size)   # crop_size should be smaller than the size of loaded image
        self.input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        self.output_nc = self.opt.input_nc if self.opt.direction == 'BtoA' else self.opt.output_nc

        self.palette = np.zeros((256,3), np.uint8)
        self.palette[:7] = np.array([
            (70, 70, 70),   # 0 reserved for left building
            (70, 70, 70),   # 1 (right) building
            (128, 64, 128), # 2 road
            (244, 35, 232), # 3 sidewalk
            (107, 142, 35), # 4 vegi
            (70, 130, 180), # 5 sky
            (0, 0, 0),      # 6 other obj
        ])

        self.half_rsl = opt.half_rsl
        return
    
    def decode_dep(self, im):
        im = im.transpose([2,0,1]).astype(np.float32)
        im = (im[0] + im[1]*256 + im[2]*256*256) / (256**3-1) * 1000.0
        return torch.from_numpy(im)

    def decode_sem(self, im):
        im = self.palette[im.flatten()].reshape(im.shape[:2]+(3,))
        return torch.from_numpy(im)

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

        img_path = self.img_paths[index]

        li = [self.cidx]
        if self.num_frames > 1:
            li += sorted(random.sample(self.idx_list, self.num_frames-1))

        dep, sem, rgb, mat, semidx, mask = [], [], [], [], [], []
        c_mat = get_transformation_matrix().float()
        for item in li:

            img_AB = Image.open(img_path + '_%02d.png' % item)
            w, h = img_AB.size
            half_w = int(w/2)
            img_A = np.array(img_AB.crop((     0, 0, half_w, h)))
            img_B = np.array(img_AB.crop((half_w, 0,      w, h)), np.float32)

            if self.half_rsl:
                s = np.random.randint(2)
                img_A = img_A[s::2,s::2]
                img_B = img_B[s::2,s::2]
            
            img_A[..., 3] = 255 - img_A[..., 3]
            img_A[..., 3] = to_simple_label[img_A[..., 3].flatten()].reshape(img_A[..., 3].shape)

            dis = self.decode_dep(img_A[..., :3]).float()
            dis[dis > 150.0] = 150.0

            sem_idx = torch.from_numpy(img_A[..., 3].astype(np.int))
            conf = torch.from_numpy(img_B[..., 3]).float() / 255.0

            # ignore_1 = (dis > 149.9) & (sem_idx != 5) # non-sky but distant
            # ignore_2 = (dis < 149.9) & (sem_idx == 5) # sky but close
            # ignore = ignore_1 | ignore_2
            # sem_idx[ignore] = 6 # other obj

            mask.append(conf*0+1)#~ignore * conf)

            dep.append(dis)
            semidx.append(sem_idx.unsqueeze(-1))
            
            sem.append(self.decode_sem(sem_idx.numpy()).float()/127.5-1)
            rgb.append(torch.from_numpy(img_B[..., :3])/127.5-1)
            mat.append(get_transformation_matrix(item).float())
        
        dep, sem, rgb, mat, semidx, mask = torch.stack(dep), torch.stack(sem), torch.stack(rgb), torch.stack(mat), torch.stack(semidx), torch.stack(mask)

        return {'idx': index, 'dep': dep, 'sem': sem, 'rgb': rgb, 'sem_idx': semidx, 'mat': mat, 'c_mat': c_mat, 'mask': mask}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.img_paths)

