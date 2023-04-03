import torch
import numpy as np
from .base_model import BaseModel
from . import networks
from PIL import Image
import lpips

def write_numpy_array(filename, arr):
    with open(filename, 'w') as f:
        for line in arr:
            f.write(('%.6f;'*3+'%d;'*3)[:-1] % tuple(list(line)) + '\n')

class InputDataProcessor(object):
    def __init__(self, opt, device):
        self.check_dep_sem = opt.check_dep_sem
        self.batch_size = opt.batch_size
        self.num_frames = opt.num_frames
        self.half_rsl = opt.half_rsl
        self.rotate_data = opt.rotate_data
        self.device = device
        self.vecs = self.get_panorama_vector().unsqueeze(0).unsqueeze(0).to(self.device)
        return
    
    def get_mat_random_rotate_XY(self):
        delta_z = torch.rand(1) * 4 - 2
        theta = torch.rand(1) * np.pi * 2
        cos_th = torch.cos(theta)
        sin_th = torch.sin(theta)
        mat = torch.eye(4)
        mat[0,0] = cos_th
        mat[1,1] = cos_th
        mat[0,1] = -sin_th
        mat[1,0] = sin_th
        mat[2,3] = delta_z
        return mat

    def get_panorama_vector(self,
            panorama_size=[512, 256],
            min_max_lat=[-np.pi/2, np.pi/2],
        ):

        ncol, nrow = panorama_size
        if self.half_rsl:
            ncol //= 2
            nrow //= 2
        self.ncol = ncol
        self.nrow = nrow
        min_lat, max_lat = min_max_lat

        x = (np.arange(0, ncol, 1, dtype=np.float32) + 0.5) / ncol
        y = (np.arange(0, nrow, 1, dtype=np.float32) + 0.5) / nrow
        lon = (x * 2.0 * np.pi - np.pi)
        lat = (1.0 - y) * (max_lat - min_lat) + min_lat

        sin_lon = np.tile(np.sin(lon).reshape((1, ncol)), (nrow, 1))
        cos_lon = np.tile(np.cos(lon).reshape((1, ncol)), (nrow, 1))
        sin_lat = np.tile(np.sin(lat).reshape((nrow, 1)), (1, ncol))
        cos_lat = np.tile(np.cos(lat).reshape((nrow, 1)), (1, ncol))

        vx, vy, vz = cos_lat * sin_lon, cos_lat * cos_lon, sin_lat
        # carla uses left-hand coordinate system
        return torch.from_numpy(np.stack([vy, vx, vz], axis=-1)).float()

    def homo(self, arr):
        return torch.cat([arr, torch.ones(*(arr.shape[:-1] + (1,))).to(arr.device)], dim=-1)
    
    def get_local_coord(self, dep):
        local_coord = dep.unsqueeze(-1).expand(*(dep.shape+(3,)))
        local_coord = local_coord * self.vecs.expand(*local_coord.shape)
        return self.homo(local_coord)

    def process_data(self, data):
        print('Process data', data['idx'])
        dep = data['dep'].to(self.device)
        sem = data['sem'].to(self.device)
        rgb = data['rgb'].to(self.device)
        sem_idx = data['sem_idx'].to(self.device)
        mat = data['mat'].to(self.device)
        c_mat = data['c_mat'].to(self.device)

        local_coord = self.get_local_coord(dep)

        coords = []
        for i in range(c_mat.size(0)):
            if self.rotate_data:
                rand_rot = self.get_mat_random_rotate_XY().to(self.device)
                inv_c_mat = torch.mm(rand_rot, torch.inverse(c_mat[i]))
            else:
                inv_c_mat = torch.inverse(c_mat[i])
            coord = []
            for j in range(self.num_frames):
                coord.append(
                    torch.mm(
                        torch.mm(inv_c_mat, mat[i, j]),
                        local_coord[i, j].view(-1, 4).t()
                    )[:3].t()
                )
            coord = torch.cat(coord, dim=0)
            coords.append(coord)

        coords = torch.stack(coords) # B, (F * H * W), 3

        local_coord = local_coord.reshape(coords.size(0), coords.size(1), 4)
        fix = dep.reshape(dep.size(0), -1) > 149.9
        for i in range(coords.size(0)):
            coords[i,fix[i]] = local_coord[i,fix[i],:3]
        #     write_numpy_array(f'tmp/{i}_rgb.txt', torch.cat([coords[i],(rgb[i].view(-1,3)+1)/2*255],dim=1).cpu().numpy())
        #     write_numpy_array(f'tmp/{i}_sem.txt', torch.cat([coords[i],(sem[i].view(-1,3)+1)/2*255],dim=1).cpu().numpy())
        # input('ready to download')

        d = {
            'pc_coord': coords,
            'im_sem':   torch.cat([
                            sem.permute([0,1,4,2,3]),
                            coords.view(self.batch_size,self.num_frames,self.nrow,self.ncol,3).permute([0,1,4,2,3]),
                            sem_idx.float().permute([0,1,4,2,3]),
                        ], dim=2),
            'im_rgb':   rgb.permute([0,1,4,2,3]),
            'mask':     data['mask'].to(self.device),
        }
        return d


class BiCycleGANModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        return parser

    def __init__(self, opt):
        if opt.isTrain:
            assert opt.batch_size % 2 == 0  # load two images at one time.

        BaseModel.__init__(self, opt)
        self.netG_name = opt.netG

        # >>> add >>>
        self.dataset_mode = opt.dataset_mode
        self.batch_size = opt.batch_size
        self.check_dep_sem = opt.check_dep_sem
        self.only_supervise_center_frame = opt.only_supervise_center_frame
        self.grid_sampling = opt.grid_sampling
        self.final_upsample = opt.final_upsample
        self.loss_fn_vgg = lpips.LPIPS(net='vgg').to(self.device)
        # <<< add <<<

        # specify the training losses you want to print out. The program will call base_model.get_current_losses
        self.loss_names = ['G_GAN', 'D', 'G_GAN2', 'D2', 'D3', 'G_L1', 'z_L1', 'kl']
        # specify the images you want to save/display. The program will call base_model.get_current_visuals
        self.visual_names = ['real_A_encoded', 'real_B_encoded', 'fake_B_random', 'fake_B_encoded']
        if self.dataset_mode.endswith('pc'):
            self.visual_names = [item+'_vis' for item in self.visual_names]
        if self.final_upsample:
            self.visual_names += ['fake_B_random_up', 'fake_B_encoded_up']
        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks
        use_D = opt.isTrain and opt.lambda_GAN > 0.0
        use_D2 = opt.isTrain and opt.lambda_GAN2 > 0.0 and not opt.use_same_D
        use_D3 = opt.isTrain and self.final_upsample // 10 == 3
        use_E = opt.isTrain or not opt.no_encode
        use_vae = True
        self.model_names = ['G']
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.nz, opt.ngf, middle_nc=opt.middle_nc, netG=opt.netG,
                                      norm=opt.norm, nl=opt.nl, use_dropout=opt.use_dropout, init_type=opt.init_type, init_gain=opt.init_gain,
                                      gpu_ids=self.gpu_ids, where_add=opt.where_add, upsample=opt.upsample, use_spade=opt.use_spade,
                                      grid_sampling=opt.grid_sampling, input_coord=not opt.not_input_coord,
                                      scn_ratio=opt.scn_ratio, rn_ratio=opt.rn_ratio, final_upsample=opt.final_upsample)
        D_output_nc = opt.input_nc + opt.output_nc if opt.conditional_D else opt.output_nc
        if use_D:
            self.model_names += ['D']
            self.netD = networks.define_D(D_output_nc, opt.ndf, netD=opt.netD, norm=opt.norm, nl=opt.nl,
                                          init_type=opt.init_type, init_gain=opt.init_gain, num_Ds=opt.num_Ds, gpu_ids=self.gpu_ids)
        if use_D2:
            self.model_names += ['D2']
            self.netD2 = networks.define_D(D_output_nc, opt.ndf, netD=opt.netD2, norm=opt.norm, nl=opt.nl,
                                           init_type=opt.init_type, init_gain=opt.init_gain, num_Ds=opt.num_Ds, gpu_ids=self.gpu_ids)
        else:
            self.netD2 = None
        
        if use_D3:
            self.model_names += ['D3']
            self.netD3 = networks.define_D(D_output_nc, opt.ndf, netD=opt.netD2, norm=opt.norm, nl=opt.nl,
                                           init_type=opt.init_type, init_gain=opt.init_gain, num_Ds=opt.num_Ds, gpu_ids=self.gpu_ids)
        else:
            self.netD3 = None

        if use_E:
            self.model_names += ['E']
            self.netE = networks.define_E(opt.output_nc, opt.nz, opt.nef, netE=opt.netE, norm=opt.norm, nl=opt.nl,
                                          init_type=opt.init_type, init_gain=opt.init_gain, gpu_ids=self.gpu_ids, vaeLike=use_vae)

        if opt.isTrain:
            self.criterionGAN = networks.GANLoss(gan_mode=opt.gan_mode).to(self.device)
            if self.check_dep_sem:
                self.criterionL1 = torch.nn.L1Loss(reduction='none')
            else:
                self.criterionL1 = torch.nn.L1Loss()
            self.criterionZ = torch.nn.L1Loss()
            # initialize optimizers
            self.optimizers = []
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            if use_E:
                self.optimizer_E = torch.optim.Adam(self.netE.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
                self.optimizers.append(self.optimizer_E)

            if use_D:
                self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
                self.optimizers.append(self.optimizer_D)
            if use_D2:
                self.optimizer_D2 = torch.optim.Adam(self.netD2.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
                self.optimizers.append(self.optimizer_D2)
            if use_D3:
                self.optimizer_D3 = torch.optim.Adam(self.netD3.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
                self.optimizers.append(self.optimizer_D3)

        # ===== Added by Zuoyue =====
        self.processor = InputDataProcessor(opt, self.device)
        # ===== Added by Zuoyue =====

        return

    def is_train(self):
        """check if the current batch is good for training."""
        return self.opt.isTrain and self.real_A.size(0) == self.opt.batch_size

    def set_input(self, input):
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']
    
    # ===== Added by Zuoyue =====
    def set_input_point_cloud(self, data):
        if self.dataset_mode.endswith('_pc'):
            print('Process data', data['data_idx'])
            for k in data:
                data[k] = data[k].to(self.device)
            #     print(k, data[k].shape)
            # print(data['pc_len'])
            # quit()
            self.real_coord = data['coord'].float()
            self.real_A = torch.cat([
                data['sem'].float(), data['coord'].float(), data['sem_idx'].unsqueeze(-1).float()
            ], dim=-1)
            self.real_B = data['rgb'].float()
            self.pc_len = data['pc_len'].long()
            self.img_idx = data['idx'].long()
            self.center_rgb = data['center_rgb'].permute([0,3,1,2]).unsqueeze(1).float()

            if self.final_upsample and self.opt.isTrain:
                self.final_im = data['org_rgbs'].permute([0,1,4,2,3]).float()

            self.warp_sate = torch.zeros(data['pc_len'].size(0),15,3,256,512).float().cuda()
            if self.final_upsample // 10 == 2:
                if 'warp_sate' in data:
                    self.warp_sate = data['warp_sate'].permute([0,1,4,2,3]).float()
                else:
                    pass
            
            # if self.final_upsample:
            #     self.center_rgb = self.final_im[:,7:8]

            # for i in range(self.real_coord.shape[0]):
            #     l = self.pc_len[i]
            #     write_numpy_array(f'tmp/{i}_rgb.txt', torch.cat([self.real_coord[i,:l],(self.real_B[i,:l,:3].view(-1,3)+1)/2*255],dim=1).cpu().numpy())
            #     write_numpy_array(f'tmp/{i}_sem.txt', torch.cat([self.real_coord[i,:l],(self.real_A[i,:l,:3].view(-1,3)+1)/2*255],dim=1).cpu().numpy())
            # input('ready to download')

            # for k in data:
            #     print(k, data[k].shape, data[k].min(), data[k].max())
            # quit()
            # coord torch.Size([2, 220000, 3]) tensor(-927.8769, dtype=torch.float64) tensor(999.9247, dtype=torch.float64)
            # rgb torch.Size([2, 220000, 3]) tensor(-0.9922) tensor(1.)
            # sem torch.Size([2, 220000, 3]) tensor(-1.) tensor(0.9137)
            # idx torch.Size([2, 15, 128, 256]) tensor(0) tensor(179228)
            # sem_idx torch.Size([2, 220000]) tensor(1, dtype=torch.uint8) tensor(6, dtype=torch.uint8)
            # pc_len torch.Size([2]) tensor(176906) tensor(179229)
            # center_rgb torch.Size([2, 128, 256, 3]) tensor(-0.9922) tensor(1.)
        else:
            d = self.processor.process_data(data) # N = F * H * W
            self.real_coord = d['pc_coord']       # B, N, 3
            self.real_A = d['im_sem']             # B, F, 3, H, W
            self.real_B = d['im_rgb']             # B, F, 3, H, W

            if self.check_dep_sem:
                self.mask = d['mask']             # B, F, H, W

            # for item in [self.real_coord, self.real_A, self.real_B]:
            #     print(item.shape, item.device)
            # input()
        
        return
    # ===== Added by Zuoyue =====

    def get_z_random(self, batch_size, nz, random_type='gauss'):
        if random_type == 'uni':
            z = torch.rand(batch_size, nz) * 2.0 - 1.0
        elif random_type == 'gauss':
            z = torch.randn(batch_size, nz)
        return z.detach().to(self.device)
    
    def get_z_random_local(self, tensor_size, random_type='gauss'):
        if random_type == 'uni':
            z = torch.rand(tensor_size) * 2.0 - 1.0
        elif random_type == 'gauss':
            torch.manual_seed(1994)
            z = torch.randn(tensor_size)
        # half = tensor_size[1] // 2
        # z[:,:half,:,:] = z[:,:half,0:1,0:1]
        return z.detach().to(self.device)

    def encode_original(self, input_image):
        mu, logvar = self.netE.forward(input_image)
        std = logvar.mul(0.5).exp_()
        eps = self.get_z_random(std.size(0), std.size(1))
        z = eps.mul(std).add_(mu)
        return z, mu, logvar

    # def encode(self, input_image):
    #     # input_image B//2, F, 3, H, W
    #     bb, ff, _, hh, ww = input_image.size()

    #     mu, logvar = self.netE.forward(input_image.view(bb*ff, 3, hh, ww))
    #     std = logvar.mul(0.5).exp_()

    #     # >>>>>>>
    #     mu = mu.view(bb, ff, self.opt.nz).mean(dim=1)
    #     std = std.view(bb, ff, self.opt.nz).mean(dim=1)
    #     logvar = std.log_().mul(2.0)
    #     # >>>>>>>

    #     eps = self.get_z_random(std.size(0), std.size(1))
    #     z = eps.mul(std).add_(mu)
    #     return z, mu, logvar

    def encode(self, input_image, local=False):
        # input_image B//2, F, 3, H, W
        if not local:
            mu, logvar = self.netE.forward(input_image[:,0,:3,:,1:-1])
            std = logvar.mul(0.5).exp_()
            eps = self.get_z_random(std.size(0), std.size(1))
            z = eps.mul(std).add_(mu)
            return z, mu, logvar
        else:
            mu, logvar = self.netE.forward(input_image[:,0,:3])
            std = logvar.mul(0.5).exp_()
            eps = self.get_z_random_local(std.size())
            z = eps.mul(std).add_(mu)
            return z, mu, logvar


    def test_original(self, z0=None, encode=False):
        with torch.no_grad():
            if encode:  # use encoded z
                z0, _ = self.netE(self.real_B)
            if z0 is None:
                z0 = self.get_z_random(self.real_A.size(0), self.opt.nz)
            self.fake_B = self.netG(self.real_A, z0)
            return self.real_A, self.fake_B, self.real_B

    def test(self, z0=None, encode=False, per_frame=False):
        with torch.no_grad():
            if encode:  # use encoded z
                z0, _, _ = self.encode(self.real_B, local=self.opt.local_encoder)
            if z0 is None:
                z0 = self.get_z_random(self.real_A.size(0), self.opt.nz)
            if per_frame:
                self.fake_B = []
                num = self.real_coord.size(1) // self.real_A.size(1)
                for i in range(self.real_A.size(1)):
                    self.fake_B.append(
                        self.netG(self.real_A[:,i:i+1], self.real_coord[:,i*num:i*num+num], z0)[0]
                    )
                self.fake_B = torch.cat(self.fake_B,dim=1)
            else:
                self.fake_B = self.netG(self.real_A, self.real_coord, z0)
            return self.real_A, self.fake_B, self.real_B
    

    def test_pc(self, z0=None, encode=False):
        with torch.no_grad():
            if encode:  # use encoded z
                z0, _, _ = self.encode(self.center_rgb, local=self.opt.local_encoder)
            if z0 is None:
                z0 = self.get_z_random(self.real_A.size(0), self.opt.nz)

            self.fake_B = self.netG(
                self.real_A, self.real_coord, z0,
                (self.pc_len, self.img_idx),
                self.warp_sate,
            )

            self.fake_B_up = None
            self.to_inpaint = None
            if type(self.fake_B) == dict:
                if 'to_inpaint' in self.fake_B:
                    self.to_inpaint = self.fake_B['to_inpaint']
                self.fake_B_up = self.fake_B['up']
                self.fake_B = self.fake_B['out']

            self.fake_B = self.fake_B.permute([0,2,1])
            # fake_B = self.fake_B

            # l = self.pc_len[0]
            # write_numpy_array('tmp/0_pred.txt', torch.cat([self.real_coord[0,:l],(self.fake_B[0,:l,:3].view(-1,3)+1)/2*255],dim=1).cpu().numpy())
            # qqq = torch.pca_lowrank(sparse_ft[0,:l], q=3)[0]
            # print(qqq.shape)
            # qm = torch.mean(qqq, dim=0, keepdim=True).expand_as(qqq)
            # qv = torch.std(qqq, dim=0, keepdim=True).expand_as(qqq)
            # qqq = torch.clamp((qqq - qm) / qv, -1, 1)
            # write_numpy_array('tmp/0_mid.txt', torch.cat([self.real_coord[0,:l],(qqq.view(-1,3)+1)/2*255],dim=1).cpu().numpy())
            # input('ready to download')

            real_A = self.pc2im(self.real_A[:,:,:3], self.pc_len, self.img_idx, no_cat=True)
            fake_B = self.pc2im(self.fake_B, self.pc_len, self.img_idx, no_cat=True)
            real_B = self.pc2im(self.real_B, self.pc_len, self.img_idx, no_cat=True)

            if self.fake_B_up is not None:
                if self.to_inpaint is not None:
                    return real_A, fake_B, real_B, self.fake_B_up.unsqueeze(0), self.to_inpaint
                else:
                    return real_A, fake_B, real_B, self.fake_B_up.unsqueeze(0)
            else:
                return real_A, fake_B, real_B

            # print(fake_B.shape, fake_B.min(), fake_B.max())
            # print(self.fake_B_up.shape, self.fake_B_up.min(), self.fake_B_up.max())
            # quit()

            # return real_A, fake_B, real_B


    def forward_original(self):
        # get real images
        half_size = self.opt.batch_size // 2
        # A1, B1 for encoded; A2, B2 for random
        self.real_A_encoded = self.real_A[0:half_size]
        self.real_B_encoded = self.real_B[0:half_size]
        self.real_A_random = self.real_A[half_size:]
        self.real_B_random = self.real_B[half_size:]
        # get encoded z
        self.z_encoded, self.mu, self.logvar = self.encode(self.real_B_encoded)
        # get random z
        self.z_random = self.get_z_random(self.real_A_encoded.size(0), self.opt.nz)
        # generate fake_B_encoded
        self.fake_B_encoded = self.netG(self.real_A_encoded, self.z_encoded)
        # generate fake_B_random
        self.fake_B_random = self.netG(self.real_A_encoded, self.z_random)
        if self.opt.conditional_D:   # tedious conditoinal data
            self.fake_data_encoded = torch.cat([self.real_A_encoded, self.fake_B_encoded], 1)
            self.real_data_encoded = torch.cat([self.real_A_encoded, self.real_B_encoded], 1)
            self.fake_data_random = torch.cat([self.real_A_encoded, self.fake_B_random], 1)
            self.real_data_random = torch.cat([self.real_A_random, self.real_B_random], 1)
        else:
            self.fake_data_encoded = self.fake_B_encoded
            self.fake_data_random = self.fake_B_random
            self.real_data_encoded = self.real_B_encoded
            self.real_data_random = self.real_B_random

        # compute z_predict
        if self.opt.lambda_z > 0.0:
            self.mu2, logvar2 = self.netE(self.fake_B_random)  # mu2 is a point estimate

    # ===== Added by Zuoyue =====
    def forward(self):
        # get real images
        first_half = self.opt.batch_size // 2
        rest_half = self.opt.batch_size - first_half
        # A1, B1 for encoded; A2, B2 for random                 # N = F * H * W
        self.real_A_encoded = self.real_A[0:first_half]         # B//2  , F, 3, H, W
        self.real_B_encoded = self.real_B[0:first_half]         # B//2  , F, 3, H, W
        self.real_coord_encoded = self.real_coord[0:first_half] # B//2  , N, 3
        if self.check_dep_sem:
            self.mask_encoded = self.mask[0:first_half]
            self.mask_random = self.mask[first_half:]

        self.real_A_random = self.real_A[first_half:]           # B-B//2, F, 3, H, W
        self.real_B_random = self.real_B[first_half:]           # B-B//2, F, 3, H, W
        self.real_coord_random = self.real_coord[first_half:]   # B-B//2, N, 3
        # get encoded z >>> shape (B//2, nz)      global
        # get encoded z >>> shape (B//2, 8, nz)   local
        self.z_encoded, self.mu, self.logvar = self.encode(self.real_B_encoded, local=self.opt.local_encoder)
        # get random z >>>> shape (B//2, nz)      global
        # get random z >>>> shape (B//2, 8, nz)   local
        if self.opt.local_encoder:
            self.z_random = self.get_z_random_local(self.mu.size())
        else:
            self.z_random = self.get_z_random(self.real_A_encoded.size(0), self.opt.nz)

        # generate fake_B_encoded
        self.fake_B_encoded = self.netG(self.real_A_encoded, self.real_coord_encoded, self.z_encoded)
        # generate fake_B_random
        self.fake_B_random = self.netG(self.real_A_encoded, self.real_coord_encoded, self.z_random)

        if self.opt.conditional_D:   # tedious conditoinal data
            assert(False, 'Not support')
            self.fake_data_encoded = torch.cat([self.real_A_encoded, self.fake_B_encoded], 1)
            self.real_data_encoded = torch.cat([self.real_A_encoded, self.real_B_encoded], 1)
            self.fake_data_random = torch.cat([self.real_A_encoded, self.fake_B_random], 1)
            self.real_data_random = torch.cat([self.real_A_random, self.real_B_random], 1)
        else:
            self.fake_data_encoded = self.fake_B_encoded
            self.fake_data_random = self.fake_B_random
            self.real_data_encoded = self.real_B_encoded
            self.real_data_random = self.real_B_random

        # compute z_predict
        if self.opt.lambda_z > 0.0:
            _, self.mu2, self.logvar2 = self.encode(self.fake_B_random, local=self.opt.local_encoder)  # mu2 is a point estimate

        self.fake_data_encoded = self.merge_first_two_dim(self.fake_data_encoded)
        self.fake_data_random = self.merge_first_two_dim(self.fake_data_random)
        self.real_data_encoded = self.merge_first_two_dim(self.real_data_encoded)
        self.real_data_random = self.merge_first_two_dim(self.real_data_random)
        self.real_A_encoded = self.merge_first_two_dim(self.real_A_encoded)
        self.real_B_encoded = self.merge_first_two_dim(self.real_B_encoded)
        self.fake_B_random = self.merge_first_two_dim(self.fake_B_random)
        self.fake_B_encoded = self.merge_first_two_dim(self.fake_B_encoded)
        if self.check_dep_sem:
            self.mask_encoded = self.merge_first_two_dim(self.mask_encoded)
            self.mask_random = self.merge_first_two_dim(self.mask_random)
        if self.netG_name == 'hybrid':
            self.middle_encoded = self.merge_first_two_dim(self.middle_encoded)
            self.middle_random = self.merge_first_two_dim(self.middle_random)
        return
    



    def forward_pc(self):
        # get real images
        first_half = self.opt.batch_size // 2
        rest_half = self.opt.batch_size - first_half
        # A1, B1 for encoded; A2, B2 for random                 # N = #points
        self.real_A_encoded = self.real_A[0:first_half]         # B//2, N, 7 (sem+coord+sem_idx)
        self.real_B_encoded = self.real_B[0:first_half]         # B//2, N, 3
        self.real_coord_encoded = self.real_coord[0:first_half] # B//2, N, 3
        self.pc_len_encoded = self.pc_len[0:first_half]         # B//2
        self.img_idx_encoded = self.img_idx[0:first_half]       # B//2, 15, H, W

        self.warp_sate_encoded = self.warp_sate[0:first_half]
        self.warp_sate_random = self.warp_sate[first_half:]

        # self.real_A_random = self.real_A[first_half:]         # B-B//2, N, 7
        self.real_B_random = self.real_B[first_half:]           # B-B//2, N, 3
        # self.real_coord_random = self.real_coord[first_half:] # B-B//2, N, 3
        self.pc_len_random = self.pc_len[first_half:]           # B-B//2
        self.img_idx_random = self.img_idx[first_half:]         # B-B//2, 15, H, W
        # get encoded z >>> shape (B//2, nz)      global
        # get encoded z >>> shape (B//2, 8, nz)   local
        self.z_encoded, self.mu, self.logvar = self.encode(self.center_rgb[0:first_half], local=self.opt.local_encoder)
        # get random z >>>> shape (B//2, nz)      global
        # get random z >>>> shape (B//2, 8, nz)   local

        if self.opt.local_encoder:
            self.z_random = self.get_z_random_local(self.mu.size())
        else:
            self.z_random = self.get_z_random(self.real_A_encoded.size(0), self.opt.nz)

        # generate fake_B_encoded
        self.fake_B_encoded = self.netG(
            self.real_A_encoded, self.real_coord_encoded, self.z_encoded,
            (self.pc_len_encoded, self.img_idx_encoded),
            self.warp_sate_encoded
        )

        if type(self.fake_B_encoded) == dict:
            self.fake_B_encoded_up = self.fake_B_encoded['up']
            self.fake_B_encoded = self.fake_B_encoded['out']

        if self.netG_name != 'randlanet2d':
            self.fake_B_encoded = self.fake_B_encoded.permute([0,2,1])


        # generate fake_B_random
        self.fake_B_random = self.netG(
            self.real_A_encoded, self.real_coord_encoded, self.z_random,
            (self.pc_len_encoded, self.img_idx_encoded),
            self.warp_sate_encoded
        )

        if type(self.fake_B_random) == dict:
            self.fake_B_random_up = self.fake_B_random['up']
            self.fake_B_random = self.fake_B_random['out']

        if self.netG_name != 'randlanet2d':
            self.fake_B_random = self.fake_B_random.permute([0,2,1])

        if self.opt.conditional_D:   # tedious conditoinal data
            assert(False, 'Not support')
        else:
            self.fake_data_encoded = self.fake_B_encoded # p1 fake RGB use encoded seed
            self.fake_data_random = self.fake_B_random # p1 fake RGB use random seed
            self.real_data_encoded = self.real_B_encoded # p1 real RGB
            self.real_data_random = self.real_B_random # p2 real RGB

        # compute z_predict
        if self.opt.lambda_z > 0.0:
            # print(self.fake_B_random.shape) B, N, 3
            # print(self.img_idx_encoded.shape) B, 15, H, W
            center_idx = self.img_idx_encoded.size(1) // 2
            hh, ww = self.img_idx_encoded.size()[2:]
            if self.netG_name != 'randlanet2d':
                to_encode = []
                for j in range(first_half):
                    pc_len = self.pc_len_encoded[j]
                    result = self.fake_B_random[j, :pc_len].permute([1, 0]) # 3, pc_len
                    sel = self.img_idx_encoded[j, center_idx].view(-1)
                    to_encode.append(result[:, sel].view(3, hh, ww))
                to_encode = torch.stack(to_encode).unsqueeze(1)
            else:
                to_encode = self.fake_B_random[0, 7:8].unsqueeze(1)
            _, self.mu2, self.logvar2 = self.encode(to_encode, local=self.opt.local_encoder)  # mu2 is a point estimate

        if self.netG_name != 'randlanet2d':
            # data goes to D
            self.fake_data_encoded = self.pc2im(self.fake_data_encoded, self.pc_len_encoded, self.img_idx_encoded) # p1 fake RGB use encoded seed
            self.fake_data_random = self.pc2im(self.fake_data_random, self.pc_len_encoded, self.img_idx_encoded) # p1 fake RGB use random seed
            self.real_data_encoded = self.pc2im(self.real_data_encoded, self.pc_len_encoded, self.img_idx_encoded) # p1 real RGB
            self.real_data_random = self.pc2im(self.real_data_random, self.pc_len_random, self.img_idx_random) # p2 real RGB

            if self.final_upsample:
                self.fake_data_encoded = self.fake_B_encoded_up
                self.fake_data_random = self.fake_B_random_up
                self.real_data_encoded = self.final_im[0]
                self.real_data_random = self.final_im[1]

            # data goes to loss
            self.real_B_encoded = self.merge_pc(self.real_B_encoded, self.pc_len_encoded) # p1 real RGB
            self.fake_B_encoded = self.merge_pc(self.fake_B_encoded, self.pc_len_encoded) # p1 fake RGB use encoded seed

            # data for vis
            self.real_A_encoded_vis = self.pc2im(self.real_A_encoded[:,:,:3], self.pc_len_encoded, self.img_idx_encoded) # p1 semantic input
            self.real_B_encoded_vis = self.real_data_encoded # p1 real RGB
            self.fake_B_random_vis =  self.fake_data_random # p1 fake RGB use random seed
            self.fake_B_encoded_vis = self.fake_data_encoded # p1 fake RGB use encoded seed
        else:

            # data goes to D
            self.fake_data_encoded = self.fake_data_encoded[0]
            self.fake_data_random = self.fake_data_random[0]
            self.real_data_encoded = self.pc2im(self.real_data_encoded, self.pc_len_encoded, self.img_idx_encoded) # p1 real RGB
            self.real_data_random = self.pc2im(self.real_data_random, self.pc_len_random, self.img_idx_random) # p2 real RGB

            # data goes to loss
            self.real_B_encoded = self.pc2im(self.real_B_encoded, self.pc_len_encoded, self.img_idx_encoded)
            self.fake_B_encoded = self.fake_B_encoded[0]
            
            # data for vis
            self.real_A_encoded_vis = self.pc2im(self.real_A_encoded[:,:,:3], self.pc_len_encoded, self.img_idx_encoded) # p1 semantic input
            self.real_B_encoded_vis = self.real_data_encoded # p1 real RGB
            self.fake_B_random_vis =  self.fake_data_random # p1 fake RGB use random seed
            self.fake_B_encoded_vis = self.fake_data_encoded

        return
    





    # ===== Added by Zuoyue =====

    def merge_pc(self, pc, pc_len):
        # pc: B, N, 3
        B = pc.size(0)
        res = []
        for i in range(B):
            res.append(pc[i, :pc_len[i]])
        return torch.cat(res)

    def pc2im(self, pc, pc_len, img_idx, no_cat=False):
        # pc: B, N, 3
        B = pc.size(0)
        assert(B == pc_len.size(0) == img_idx.size(0))
        _, F, H, W = img_idx.size()
        ims = []
        for i in range(B):
            im = pc[i, :pc_len[i]] [img_idx[i].view(-1)].reshape(F,H,W,3)
            ims.append(im.permute([0,3,1,2]))
        if no_cat:
            return torch.stack(ims)
        else:
            return torch.cat(ims)

    def merge_first_two_dim(self, tensor):
        shape = [tensor.size(i) for i in range(len(tensor.size()))]
        bb = shape[0]
        shape = shape[1:]
        shape[0] *= bb
        return tensor.reshape(*shape)


    def backward_D(self, netD, real, fake):
        # Fake, stop backprop to the generator by detaching fake_B
        pred_fake = netD(fake.detach()[:,:3])
        # real
        pred_real = netD(real[:,:3])
        loss_D_fake, _ = self.criterionGAN(pred_fake, False)
        loss_D_real, _ = self.criterionGAN(pred_real, True)
        # Combined loss
        loss_D = loss_D_fake + loss_D_real
        loss_D.backward()
        return loss_D, [loss_D_fake, loss_D_real]

    def backward_G_GAN(self, fake, netD=None, ll=0.0):
        if ll > 0.0:
            pred_fake = netD(fake[:,:3])
            loss_G_GAN, _ = self.criterionGAN(pred_fake, True)
        else:
            loss_G_GAN = 0
        return loss_G_GAN * ll

    def backward_EG(self):
        # 1, G(A) should fool D
        self.loss_G_GAN = self.backward_G_GAN(self.fake_data_encoded, self.netD, self.opt.lambda_GAN)
        # if self.netG_name == 'hybrid':
        #     self.loss_G_GAN_m = self.backward_G_GAN(self.middle_encoded, self.netD, self.opt.lambda_GAN * 0.7)
        if self.opt.use_same_D:
            self.loss_G_GAN2 = self.backward_G_GAN(self.fake_data_random, self.netD, self.opt.lambda_GAN2)
        #     if self.netG_name == 'hybrid':
        #         self.loss_G_GAN2_m = self.backward_G_GAN(self.middle_random, self.netD, self.opt.lambda_GAN2 * 0.7)
        else:
            self.loss_G_GAN2 = self.backward_G_GAN(self.fake_data_random, self.netD2, self.opt.lambda_GAN2)
            # if hasattr(self, 'fake_B_random_low'):
            #     self.loss_G_GAN3 = self.backward_G_GAN(self.fake_B_random_low, self.netD3, self.opt.lambda_GAN2)
            #     print('---> has D3 loss line 720')
        #     if self.netG_name == 'hybrid':
        #         self.loss_G_GAN2_m = self.backward_G_GAN(self.middle_random, self.netD2, self.opt.lambda_GAN2 * 0.7)
        # 2. KL loss
        if self.opt.lambda_kl > 0.0:
            self.loss_kl = torch.sum(1 + self.logvar - self.mu.pow(2) - self.logvar.exp()) * (-0.5 * self.opt.lambda_kl)
        else:
            self.loss_kl = 0
        # 3, reconstruction |fake_B-real_B|
        if self.opt.lambda_L1 > 0.0:
            if self.check_dep_sem:
                msk = torch.stack([self.mask_encoded]*3, dim=1)
                assert(self.only_supervise_center_frame)
                self.loss_G_L1 = torch.mean(self.criterionL1(self.fake_B_encoded, self.real_B_encoded) * msk) * self.opt.lambda_L1
            else:
                if self.only_supervise_center_frame:
                    self.loss_G_L1 = self.criterionL1(
                        self.fake_B_encoded[0::self.batch_size],
                        self.real_B_encoded[0::self.batch_size]
                    ) * self.opt.lambda_L1
                else:
                    if self.dataset_mode.endswith('_pc'):
                        self.loss_G_L1 = 0
                        # self.loss_G_L1 = self.criterionL1(self.fake_B_encoded, self.real_B_encoded) * self.opt.lambda_L1
                        if self.final_upsample:
                            # TODO: first_half
                            # self.loss_G_L1 += self.criterionL1(self.fake_B_encoded_up, self.final_im[0]) * self.opt.lambda_L1
                            # hehe = ((self.real_data_encoded.permute([0,2,3,1]).cpu().numpy()+1)*127.5).astype(np.uint8)
                            # hehe = [Image.fromarray(hehe[i]) for i in range(len(hehe))]
                            # hehe[0].save('hehe.gif',duration=250,loop=0,save_all=True,append_images=hehe[1:])
                            self.loss_G_L1 += self.criterionL1(self.fake_B_encoded_up, self.real_data_encoded) * self.opt.lambda_L1
                            self.loss_G_L1 += torch.mean(self.loss_fn_vgg(self.fake_B_encoded_up, self.real_data_encoded)) * self.opt.lambda_L1

                            # input('see gif')
                            # if hasattr(self, 'fake_B_encoded_low'):
                            #     self.loss_G_L1 += self.criterionL1(
                            #         self.fake_B_encoded_low,
                            #         torch.nn.functional.interpolate(
                            #             self.real_data_encoded,
                            #             scale_factor=0.5,
                            #             mode='bilinear',
                            #             align_corners=True,
                            #             recompute_scale_factor=True
                            #         )
                            #     ) * self.opt.lambda_L1
                            #     print('---> has low res loss')
                    else:
                        self.loss_G_L1 = self.criterionL1(self.fake_B_encoded, self.real_B_encoded) * self.opt.lambda_L1
        else:
            self.loss_G_L1 = 0.0

        self.loss_G = self.loss_G_GAN + self.loss_G_GAN2 + self.loss_G_L1 + self.loss_kl
        # if self.netG_name == 'hybrid':
        #     self.loss_G += self.loss_G_GAN_m + self.loss_G_GAN2_m
        self.loss_G.backward(retain_graph=True)
        return

    def update_D(self):
        self.set_requires_grad([self.netD, self.netD2, self.netD3], True)
        # update D1
        if self.opt.lambda_GAN > 0.0:
            self.optimizer_D.zero_grad()
            self.loss_D, self.losses_D = self.backward_D(self.netD, self.real_data_encoded, self.fake_data_encoded)
            if self.opt.use_same_D:
                self.loss_D2, self.losses_D2 = self.backward_D(self.netD, self.real_data_random, self.fake_data_random)
            self.optimizer_D.step()

        if self.opt.lambda_GAN2 > 0.0 and not self.opt.use_same_D:
            self.optimizer_D2.zero_grad()
            self.loss_D2, self.losses_D2 = self.backward_D(self.netD2, self.real_data_random, self.fake_data_random)
            self.optimizer_D2.step()

        self.loss_D3 = 0
        # if self.netD3 is not None:
        #     self.optimizer_D3.zero_grad()
        #     self.loss_D3, self.losses_D3 = self.backward_D(
        #         self.netD3,
        #         torch.nn.functional.interpolate(
        #             self.real_data_random,
        #             scale_factor=0.5,
        #             mode='bilinear',
        #             align_corners=True,
        #             recompute_scale_factor=True
        #         ),
        #         self.fake_B_random_low)
        #     print('---> has D3 loss line 802')
        #     self.optimizer_D3.step()

    def backward_G_alone(self):
        # 3, reconstruction |(E(G(A, z_random)))-z_random|
        if self.opt.lambda_z > 0.0:
            self.loss_z_L1 = self.criterionZ(self.mu2, self.z_random) * self.opt.lambda_z
            self.loss_z_L1.backward()
        else:
            self.loss_z_L1 = 0.0

    def update_G_and_E(self):
        # update G and E
        self.set_requires_grad([self.netD, self.netD2, self.netD3], False)
        self.optimizer_E.zero_grad()
        self.optimizer_G.zero_grad()
        self.backward_EG()

        # update G alone
        if self.opt.lambda_z > 0.0:
            self.set_requires_grad([self.netE], False)
            self.backward_G_alone()
            self.set_requires_grad([self.netE], True)

        self.optimizer_E.step()
        self.optimizer_G.step()

    def optimize_parameters(self):
        if self.dataset_mode.endswith('_pc'):
            self.forward_pc()
        else:
            self.forward()
        self.update_G_and_E()
        self.update_D()
