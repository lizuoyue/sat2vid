import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import functools
from torch.optim import lr_scheduler
from models.randlanet import RandLANet, Sine, SharedMLP
import sparseconvnet as scn
from PIL import Image
import numpy as np

###############################################################################
# Helper functions
###############################################################################


def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.
    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.
    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            if m.weight is not None:
                init.normal_(m.weight.data, 1.0, init_gain)
            if m.bias is not None:
                init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2
    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    return net


def get_scheduler(optimizer, opt):
    """Return a learning rate scheduler
    Parameters:
        optimizer          -- the optimizer of the network
        opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine
    For 'linear', we keep the same learning rate for the first <opt.niter> epochs
    and linearly decay the rate to zero over the next <opt.niter_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.niter, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def get_norm_layer(norm_type='instance'):
    """Return a normalization layer
    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none
    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_non_linearity(layer_type='relu'):
    if layer_type == 'relu':
        nl_layer = functools.partial(nn.ReLU, inplace=True)
    elif layer_type == 'lrelu':
        nl_layer = functools.partial(
            nn.LeakyReLU, negative_slope=0.2, inplace=True)
    elif layer_type == 'elu':
        nl_layer = functools.partial(nn.ELU, inplace=True)
    elif layer_type == 'sine':
        nl_layer = functools.partial(Sine, inplace=True)
    else:
        raise NotImplementedError(
            'nonlinearity activitation [%s] is not found' % layer_type)
    return nl_layer







class HybridNet3D2D(nn.Module):
    def __init__(self, d_in, d_out, d_noise, d_middle, num_downs, ngf=64,
                 norm_layer=None, nl_layer=None, use_dropout=False, upsample='basic'):
        super(HybridNet, self).__init__()
        self.d_middle = d_middle
        self.d_noise = d_noise
        self.net_3d = RandLANet(d_in+3, d_out, d_noise, d_middle, num_neighbors=8, decimation=4)
        self.net_2d = G_Unet_add_all(d_in+d_middle, d_out, d_noise, num_downs=num_downs, ngf=ngf, norm_layer=norm_layer, nl_layer=nl_layer,
                                     use_dropout=use_dropout, upsample=upsample)
        return

    def forward(self, im, coord, noise):
        import time
        B, F, _, H, W = im.size()
        out_3d, ft_3d = self.net_3d(im, coord, noise)
        # print(im.shape, out_3d.shape, ft_3d.shape, noise.shape)
        # B, F, 3, H, W
        # B, F, 3, H, W
        # B, F, 64, H, W
        # B, d_noise
        net_2d_in = torch.cat([im, ft_3d], dim=2).view(B*F, self.d_middle+3, H, W)
        noise = noise.unsqueeze(1).expand(B, F, self.d_noise).view(B*F, self.d_noise)
        out_2d = self.net_2d(net_2d_in, noise).view(B, F, 3, H, W)
        return out_2d, out_3d





class HybridNet(nn.Module):
    def __init__(self, d_in, d_out, d_noise, d_middle, num_downs, ngf=64,
                 norm_layer=None, nl_layer=None, use_dropout=False, upsample='basic'):
        super(HybridNet, self).__init__()
        self.d_middle = d_middle
        self.d_noise = d_noise
        self.net_3d = RandLANet(d_middle+d_in, d_out, d_noise, d_middle, num_neighbors=8, decimation=4)
        self.net_2d = G_Unet_add_all(d_in, d_middle, d_noise, num_downs=num_downs, ngf=ngf, norm_layer=norm_layer, nl_layer=nl_layer,
                                     use_dropout=use_dropout, upsample=upsample)
        return

    def forward(self, im, coord, noise):
        import time
        B, F, C, H, W = im.size()
        im_2d = im.reshape(B*F,C,H,W)
        noise_2d = noise.unsqueeze(1).expand(B, F, self.d_noise).view(B*F, self.d_noise)
        out_2d = self.net_2d(im_2d, noise_2d)
        d_middle = out_2d.size(1)
        out_2d = out_2d.view(B, F, d_middle, H, W)

        im_3d = torch.cat([im[:,:,:3], out_2d], dim=2)
        out_3d, _ = self.net_3d(im_3d, coord, noise)
        # print(im.shape, out_3d.shape, ft_3d.shape, noise.shape)
        # B, F, 3, H, W
        # B, F, 3, H, W
        # B, F, 64, H, W
        # B, d_noise

        return out_3d, None







class SparseConvNet(nn.Module):
    def __init__(self, d_in, d_out, d_noise, d_middle, num_downs, ngf=64,
                 norm_layer=None, nl_layer=None, use_dropout=False, upsample='basic'):
        super(SparseConvNet, self).__init__()

        self.dimension = 3
        self.reps = 2
        self.spatial = torch.ones(3).long() * 16384
        self.nPlanes = [ngf, 2*ngf, 3*ngf, 4*ngf, 5*ngf]
        self.sparse_net = scn.Sequential().add(
            scn.InputLayer(self.dimension, self.spatial, mode=2)).add(
            scn.SubmanifoldConvolution(self.dimension, d_in+d_noise, ngf, 3, False)).add(
            scn.ConcatUNet(self.dimension, self.reps, self.nPlanes, residual_blocks=True, downsample=[2,2], d_noise=d_noise)).add(
            scn.BatchNormReLU(ngf)).add(
            scn.OutputLayer(self.dimension))
        self.init_sky = torch.nn.Parameter(torch.randn(ngf), requires_grad=True)
        self.d_middle = ngf
        self.d_noise = d_noise
        self.net_3d = RandLANet(self.d_middle+d_in+3, d_out, d_noise, self.d_middle, num_neighbors=8, decimation=4)
        return

    def forward(self, im, coord, noise):

        import numpy as np

        # coord shape B//2, N, 3
        resolution = 10
        assert(coord.size(0) == 1)
        is_city = torch.sqrt((coord[0]**2).sum(dim=-1)) < 600
        is_sky = ~is_city
        loc = coord[0, is_city]
        # print(is_city.float().mean(), coord.shape, loc.shape)

        loc *= resolution
        bias = torch.floor(loc.min(dim=0)[0])
        loc -= bias
        loc = torch.floor(loc).long()
        loc = torch.cat([loc, torch.zeros(loc.size(0), 1).long().to(loc.device)], dim=1)
        # print(loc.min(dim=0)[0], loc.max(dim=0)[0])
        # print('numpy unique', np.unique(loc.cpu().numpy().astype(np.int32), axis=0).shape)

        B, F, C, H, W = im.size()
        N = F*H*W
        im_ft = im[:,:,:3].permute([0,1,3,4,2]).reshape(N, 3)
        assert(noise.size(0) == 1)
        noise_ft = noise.expand(N, self.d_noise)
        loc_ft = torch.cat([im_ft, noise_ft], dim=1)
        loc_ft = loc_ft[is_city]

        sparse_input = (loc, loc_ft, 1)
        self.sparse_net[2].set_noise(noise)
        sparse_output = self.sparse_net(sparse_input)
        assert(not torch.isnan(sparse_output).any())

        # for aaa, bbb in zip(loc, sparse_output):
        #     print(aaa, bbb[:6])
        #     input()
        d_middle = sparse_output.size(1)

        sparse_ft = torch.zeros(N, d_middle).cuda()
        sparse_ft[is_city] = sparse_output
        sparse_ft[is_sky] = self.init_sky
        sparse_ft = sparse_ft.unsqueeze(dim=0)
        sparse_ft = sparse_ft.reshape(B, F, H, W, d_middle).permute([0,1,4,2,3])

        im_3d = torch.cat([im[:,:,:3], sparse_ft], dim=2)

        out_3d, _ = self.net_3d(im_3d, coord, noise)

        # print(im.shape, out_3d.shape, noise.shape)
        # B, F, 3, H, W
        # B, F, 3, H, W
        # B, F, 64, H, W
        # B, d_noise

        return out_3d, None

    # def to(self, gpu_id):
    #     device = torch.device('cuda:%d' % gpu_id)
    #     self.net_3d.to(device)
    #     self.sparse_net = self.sparse_net.to(device)
    #     return








class SparseConvNetMultiNoise(nn.Module):
    def __init__(self, d_in, d_out, d_noise, d_middle, num_downs, ngf=64,
                 norm_layer=None, nl_layer=None, use_dropout=False, upsample='basic',
                 use_spade=False, grid_sampling=False, input_coord=True,
                 scn_ratio=32.0, rn_ratio=1.0, final_upsample=0):
        super(SparseConvNetMultiNoise, self).__init__()

        self.dimension = 3
        self.reps = 2
        self.resolution = scn_ratio
        self.rn_ratio = rn_ratio
        self.input_coord = input_coord
        self.spatial = torch.ones(3).long() * 16384
        self.bias = torch.Tensor([8192, 8192, 8192])
        self.nPlanes = [ngf, 2*ngf, 3*ngf, 4*ngf, 5*ngf]
        self.final_upsample = final_upsample

        self.sparse_net = scn.Sequential().add(
            scn.InputLayer(self.dimension, self.spatial, mode=2)).add(
            scn.SubmanifoldConvolution(self.dimension, d_in+3, self.nPlanes[0], 3, False)).add(
            scn.ConcatUNet(self.dimension, self.reps, self.nPlanes, residual_blocks=True, downsample=[2,2], d_noise=d_noise)).add(
            scn.BatchNormReLU(self.nPlanes[0])).add(
            scn.OutputLayer(self.dimension))

        if self.rn_ratio < 0:
            self.sparse_net = scn.Sequential().add(
                scn.InputLayer(self.dimension, self.spatial, mode=2)).add(
                scn.SubmanifoldConvolution(self.dimension, d_in+3, self.nPlanes[0], 3, False)).add(
                scn.ConcatUNet(self.dimension, self.reps, self.nPlanes, residual_blocks=True, downsample=[2,2], d_noise=d_noise)).add(
                scn.BatchNormReLU(self.nPlanes[0])).add(
                scn.SubmanifoldConvolution(self.dimension, self.nPlanes[0], 3, 1, False)).add(
                scn.Tanh()).add(
                scn.OutputLayer(self.dimension))

        self.init_sky = torch.nn.Parameter(torch.randn(self.nPlanes[0]), requires_grad=True)
        self.d_middle = ngf
        self.d_noise = d_noise
        self.net_3d = RandLANet(self.d_middle+d_in+3, d_out, d_noise, self.d_middle, num_neighbors=8, decimation=4,
            random_sel=1, local_noise=True, use_spade=use_spade, grid_sampling=grid_sampling, input_coord=input_coord, ratio=rn_ratio)

        if final_upsample // 10 == 1:
            self.final_up = nn.Sequential(*[
                nn.Conv2d(128, 64, (3, 3), padding=(1, 1)),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Upsample(scale_factor=2, mode='nearest'),
                # nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.Conv2d(64, 32, (3, 3), padding=(1, 1)),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.Conv2d(32, 3, (1, 1), padding=(0, 0)),
                nn.Tanh(),
            ])

        elif final_upsample // 10 == 2:
            self.final_up_1 = nn.Sequential(*[
                nn.Conv2d(128, 128, (3, 3), padding=(1, 1)),
                nn.BatchNorm2d(128),
                nn.ReLU(),
            ])

            self.final_mid = nn.Sequential(*[
                nn.Conv2d(128, 32, (1, 1), padding=(0, 0)),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.Conv2d(32, 3, (1, 1), padding=(0, 0)),
                nn.Tanh(),
            ])

            self.up_layer = nn.Upsample(scale_factor=2, mode='nearest')#, align_corners=True)

            self.final_up_2 = nn.Sequential(*[
                nn.Conv2d(128+3, 64, (3, 3), padding=(1, 1)),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Conv2d(64, 32, (1, 1), padding=(0, 0)),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.Conv2d(32, 3, (1, 1), padding=(0, 0)),
                nn.Tanh(),
            ])

        elif final_upsample // 10 == 3:
            self.final_up = nn.Sequential(*[
                nn.Conv2d(128, 64, (3, 3), padding=(1, 1)),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Upsample(scale_factor=2, mode='nearest'),
                # nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.Conv2d(64, 32, (3, 3), padding=(1, 1)),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.Conv2d(32, 3, (1, 1), padding=(0, 0)),
                nn.Tanh(),
            ])

            self.add_sate = nn.Conv2d(3, 64, (3, 3), padding=(1, 1))

            self.add_mid = nn.Sequential(*[
                nn.Conv2d(64, 32, (3, 3), padding=(1, 1)),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.Conv2d(32, 3, (1, 1), padding=(0, 0)),
                nn.Tanh(),
            ])
        
        elif final_upsample // 10 == 9 or final_upsample // 10 == 4:
            self.final_up = nn.Sequential(*[
                nn.Conv2d(128, 64, (3, 3), padding=(1, 1)),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Upsample(scale_factor=2, mode='nearest'),
                # nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.Conv2d(64, 32, (3, 3), padding=(1, 1)),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.Conv2d(32, 3, (1, 1), padding=(0, 0)),
                nn.Tanh(),
            ])

            self.add_noise = nn.Conv2d(d_noise, 64, (3, 3), padding=(1, 1))

        else:
            pass

        self.alt = 0
        self.switch = 5576
        return

    def forward(self, im, coord, noise, info=None, warp_from_sate=None):
        if len(im.size()) == 5:
            return self.forward_im(im, coord, noise)
        elif len(im.size()) == 3:
            if self.rn_ratio > 0.5:
                if self.resolution < 1:
                    return self.forward_rn(im, coord, noise, info)
                else:
                    print('use forward_pc()')
                    return self.forward_pc(im, coord, noise, info, warp_from_sate)
            elif self.rn_ratio < 0:
                return self.forward_scn(im, coord, noise, info)
            else:
                return self.forward_glb(im, coord, noise, info)
        else:
            assert(False)
        return None


    def forward_im(self, im, coord, noise):

        # print(im.shape) # B, F, 7, 128, 256 # 7=(sem+coord+sem_idx)
        # print(coord.shape) # B, N, 3
        # print(noise.shape) # B, 8, d_noise

        B, F, _, H, W = im.size()
        _, N, _ = coord.size()
        assert(N == F*H*W)
        assert(coord.size(0) == B)
        assert(noise.size(0) == B)

        batch_idx = torch.arange(B).float().reshape(B, 1, 1).expand(B, N, 1).to(coord.device)
        coord_4 = torch.cat([coord, batch_idx], dim=-1).reshape(B*N, 4)
        is_city = torch.sqrt((coord_4[:,:3]**2).sum(dim=-1)) < 600 # B*N
        is_sky = ~is_city # B*N

        loc = coord_4[is_city] # B*N, 4
        loc[:,:3] *= self.resolution
        loc[:,:3] += self.bias.to(loc.device)
        loc[:,:3] = torch.floor(loc[:,:3])
        loc = loc.long() # B*N, 4

        im_ft = im[:,:,:6].permute([0,1,3,4,2]).reshape(B*N, 6)
        if not self.input_coord:
            im_ft[:, 3:] *= 0

        self.sparse_net[2].set_noise(noise[:, -1]) # global noise
        sparse_input = (loc, im_ft[is_city], B)
        sparse_output = self.sparse_net(sparse_input) # V, d_middle
        # assert(not torch.isnan(sparse_output).any())

        d_middle = sparse_output.size(1)
        sparse_ft = torch.zeros(B*N, d_middle).to(sparse_output.device)
        sparse_ft[is_city] = sparse_output
        sparse_ft[is_sky] = self.init_sky
        sparse_ft = sparse_ft.unsqueeze(dim=0)
        sparse_ft = sparse_ft.reshape(B, F, H, W, d_middle).permute([0,1,4,2,3])

        sem_idx = im[:,:,6].reshape(B*N).long()
        sem_idx[(coord_4[:,1] < 0) & (sem_idx == 1)] = 0 # is_left_building
        sem_idx = sem_idx.reshape(B, N, 1).expand(B, N, self.d_noise)
        # sem_idx = sem_idx * 0 + 7 # all use global noise

        # shape of noise: B, 8, d_noise
        noise_ft = torch.gather(noise, 1, sem_idx) # B, N, d_noise

        im_3d = torch.cat([im[:,:,:3], sparse_ft], dim=2)
        out_3d, _ = self.net_3d.forward(im_3d, coord, noise_ft)

        # print(im.shape, out_3d.shape, noise.shape)
        # B, F, 3, H, W
        # B, F, 3, H, W
        # B, F, 64, H, W
        # B, d_noise

        return out_3d

    def forward_pc(self, im, coord, noise, info, warp_from_sate=None):

        pc_len, img_idx = info

        # print(im.shape) # B, N, 7
        # print(coord.shape) # B, N, 3
        # print(noise.shape) # B, 8, d_noise
        # print(pc_len) # B
        # print(img_idx.shape) # B, 15, H, W
        # print(img_idx.min(), img_idx.max())

        B, N, _ = im.size()

        batch_idx = torch.arange(B).float().reshape(B, 1, 1).expand(B, N, 1).to(coord.device)
        coord_4 = torch.cat([coord, batch_idx], dim=-1).reshape(B*N, 4)

        is_city = torch.sqrt((coord_4[:,:3]**2).sum(dim=-1)) < 600 # B*N
        is_sky = ~is_city # B*N

        loc = coord_4[is_city] # B*N, 4
        loc[:,:3] *= self.resolution
        loc[:,:3] += self.bias.to(loc.device)
        loc[:,:3] = torch.floor(loc[:,:3])
        loc = loc.long() # B*N, 4

        im_ft = im[:,:,:6].reshape(B*N, 6)
        if not self.input_coord:
            im_ft[:, 3:] *= 0

        self.sparse_net[2].set_noise(noise[:, -1]) # global noise
        sparse_input = (loc, im_ft[is_city], B)


        scn_no_grad = self.final_upsample % 10
        scn_no_grad = (scn_no_grad in [0,1,3] or (scn_no_grad == 7 and self.alt < self.switch))
        if scn_no_grad:
            print('sparse_net no grad')
            with torch.no_grad():
                sparse_output = self.sparse_net(sparse_input)
        else:
            print('sparse_net with grad')
            sparse_output = self.sparse_net(sparse_input) # V, d_middle
        # assert(not torch.isnan(sparse_output).any())

        d_middle = sparse_output.size(1)
        sparse_ft = torch.zeros(B*N, d_middle).to(sparse_output.device)
        sparse_ft[is_city] = sparse_output
        sparse_ft[is_sky] = self.init_sky
        sparse_ft = sparse_ft.reshape(B, N, d_middle)

        sem_idx = im[:,:,6].reshape(B*N).long()
        sem_idx[(coord_4[:,0] < 0) & (sem_idx == 1)] = 0 # is_left_building
        sem_idx = sem_idx.reshape(B, N, 1).expand(B, N, self.d_noise)
        # sem_idx = sem_idx * 0 + 7 # all use global noise

        # shape of noise: B, 8, d_noise
        noise_ft = torch.gather(noise, 1, sem_idx) # B, N, d_noise
        im_3d = torch.cat([im[:,:,:3], sparse_ft], dim=2) # B, N, 3+d_middle

        rand_no_grad = self.final_upsample % 10
        rand_no_grad = (rand_no_grad in [0,1,4] or (rand_no_grad == 7 and self.alt >= self.switch))
        out_3d, out_upsample = [], []
        for b in range(B):

            if rand_no_grad:
                print('randlanet no grad')
                with torch.no_grad():
                    out_net_3d = self.net_3d.forward(
                        im_3d[b:b+1, :pc_len[b]],
                        coord[b:b+1, :pc_len[b]],
                        noise_ft[b:b+1, :pc_len[b]]
                    )
            else:
                print('randlanet with grad')
                out_net_3d = self.net_3d.forward(
                    im_3d[b:b+1, :pc_len[b]],
                    coord[b:b+1, :pc_len[b]],
                    noise_ft[b:b+1, :pc_len[b]]
                )

            # print(out_net_3d['output'].shape) # 1, 3, len
            # print(out_net_3d['feature'].shape) # 1, 64, len

            out_3d_pc = out_net_3d['output']

            len_to_pad = N - out_3d_pc.size(2)
            out_3d_pc = torch.cat([
                out_3d_pc,
                torch.zeros(out_3d_pc.size(0), out_3d_pc.size(1), len_to_pad).to(out_3d_pc.device)
            ], dim=-1) # 1, 3, N
            out_3d.append(out_3d_pc)

            if self.final_upsample // 10 not in [4,9]:
                out_ft_pc = torch.cat([
                    torch.t(out_net_3d['feature'][0]), # len, 64
                    sparse_ft[b, :pc_len[b]], # len, 64
                ], dim=-1) # len, 64
                out_upsample.append(out_ft_pc[img_idx[b].view(-1)].view(15, img_idx.size(2), img_idx.size(3), -1))
            else:
                out_ft_pc = torch.cat([
                    torch.t(out_net_3d['feature'][0]), # len, 64
                    sparse_ft[b, :pc_len[b]], # len, 64
                    noise_ft[b, :pc_len[b]], # len, d_noise
                ], dim=-1) # len, 64
                out_upsample.append(out_ft_pc[img_idx[b].view(-1)].view(15, img_idx.size(2), img_idx.size(3), -1))

        out_3d = torch.cat(out_3d, dim=0)

        up_no_grad = self.final_upsample % 10
        up_no_grad = (up_no_grad in [0,4]) or (up_no_grad == 7 and self.alt >= self.switch)

        if self.final_upsample // 10 == 1:
            if up_no_grad:
                print('final upsample no grad')
                with torch.no_grad():
                    out_upsample = torch.cat(out_upsample, dim=0).permute([0,3,1,2])
                    out_upsample = self.final_up(out_upsample)
            else:
                print('final upsample with grad')
                out_upsample = torch.cat(out_upsample, dim=0).permute([0,3,1,2])
                before_tanh = self.final_up[:-1](out_upsample)
                out_upsample = self.final_up[-1](before_tanh)

                # check = (out_upsample.permute([0,2,3,1]) < -0.99607843137).cpu().numpy()
                # checkimg = (1 - check.astype(np.uint8)) * 255
                # checkimg = [Image.fromarray(item) for item in checkimg]
                # checkimg[0].save('hehehe.gif', append_images=checkimg[1:], duration=250, loop=0, save_all=True)

                # aaa = before_tanh.permute([0,2,3,1]).cpu().numpy()[check]
                # print(aaa.min(), aaa.max())

                # tanh_regular = 100 * (torch.clamp(torch.abs(before_tanh) - np.pi, min=0) ** 2).mean()

            self.alt = (self.alt + 1) % (self.switch * 2)
            return {'out': out_3d, 'up': out_upsample}#, 'tanh': tanh_regular}

        elif self.final_upsample // 10 == 2:
            if up_no_grad:
                print('final upsample no grad')
                with torch.no_grad():
                    out_upsample = torch.cat(out_upsample, dim=0).permute([0,3,1,2])
                    out_upsample = self.final_up_1(out_upsample)
                    # TODO:
                    # batch size here
                    out_upsample = torch.cat([
                        self.up_layer(out_upsample),
                        0*warp_from_sate[0]],
                    dim=1)
                    out_upsample = self.final_up_2(out_upsample)
            else:
                print('final upsample with grad')
                out_upsample = torch.cat(out_upsample, dim=0).permute([0,3,1,2])
                out_upsample = self.final_up_1(out_upsample)
                # TODO:
                # batch size here
                out_upsample = torch.cat([
                    self.up_layer(out_upsample),
                    0*warp_from_sate[0]],
                dim=1)
                out_upsample = self.final_up_2(out_upsample)

            self.alt = (self.alt + 1) % (self.switch * 2)
            return {'out': out_3d, 'up': out_upsample}

        elif self.final_upsample // 10 == 3:
            
            if up_no_grad:
                assert(False)
                print('final upsample no grad')
                with torch.no_grad():
                    out_upsample = torch.cat(out_upsample, dim=0).permute([0,3,1,2])
                    out_upsample = self.final_up[:3](out_upsample)
                    # TODO:
                    # batch size here
                    low_res = self.add_mid(out_upsample)
                    out_upsample = self.final_up[3:](out_upsample)
            else:
                print('final upsample with grad')
                out_upsample = torch.cat(out_upsample, dim=0).permute([0,3,1,2])
                out_upsample_1 = self.final_up[0](out_upsample)
                out_upsample_2 = self.add_sate(
                    F.interpolate(
                        warp_from_sate[0],
                        scale_factor=0.5,
                        mode='bilinear',
                        align_corners=True,
                        recompute_scale_factor=True,
                    )
                )
                out_upsample = self.final_up[1:3](out_upsample_1 + out_upsample_2)
                # TODO:
                # batch size here
                low_res = self.add_mid(out_upsample)
                out_upsample = self.final_up[3:](out_upsample)

            self.alt = (self.alt + 1) % (self.switch * 2)

            return {'out': low_res, 'up': out_upsample}



        elif self.final_upsample // 10 == 4:

            # uturn = [(0,0),(1,0),(2,0),(3,0),(4,0),(5,0),(6,0),(7,0),(8,0),(9,0),(10,0),(11,0),(12,0),(13,0),(14,0),(14,16),(14,32),(14,48),(14,64),(14,80),(14,96),(14,112),(14,128),(14,144),(14,160),(14,176),(14,192),(14,208),(14,224),(14,240),(14,256),(13,256),(12,256),(11,256),(10,256),(9,256),(8,256),(7,256),(6,256),(5,256),(4,256),(3,256),(2,256),(1,256),(0,256),(0,272),(0,288),(0,304),(0,320),(0,336),(0,352),(0,368),(0,384),(0,400),(0,416),(0,432),(0,448),(0,464),(0,480),(0,496)]
            # uturn = [(0,128),(1,128),(2,128),(3,128),(4,128),(5,128),(6,128),(7,128),(8,128),(9,128),(10,128),(11,128),(12,128),(13,128),(14,128)]
            # uturn = [(0,112),(1,104),(2,96),(3,88),(4,80),(5,72),(6,64),(7,56),(8,48),(9,40),(10,32),(11,24),(12,16),(13,8),(14,0)]
            uturn = [(0,0),(1,8),(2,16),(3,24),(4,32),(5,40),(6,48),(7,56),(8,64),(9,72),(10,80),(11,88),(12,96),(13,104),(14,112)]

            print('U-turn mode')
            print('final upsample with grad - add noise')

            with torch.no_grad():
                out_upsample_uturn = []
                for sub in out_upsample:
                    for iii, jjj in uturn:
                        out_upsample_uturn.append(
                            torch.cat([
                                sub[iii:iii+1,:,(jjj//2):],
                                sub[iii:iii+1,:,:(jjj//2)],
                            ], dim=2)
                        )
                out_upsample = torch.cat(out_upsample_uturn, dim=0).permute([0,3,1,2])

            part1 = self.final_up[0](out_upsample[:,:128])
            part2 = self.add_noise(out_upsample[:,128:])

            out_upsample = part1 + part2

            before_tanh = self.final_up[1:-1](out_upsample)
            out_upsample = self.final_up[-1](before_tanh)

            check_1 = (out_upsample.permute([0,2,3,1]) < -0.98431372549).cpu().numpy()
            check_2 = (out_upsample.permute([0,2,3,1]) > 0.98431372549).cpu().numpy()
            check = check_1 | check_2
            # checkimg = (1 - check.astype(np.uint8)) * 255
            # checkimg = [Image.fromarray(item) for item in checkimg]
            # checkimg[0].save('hehehe.gif', append_images=checkimg[1:], duration=250, loop=0, save_all=True)
            # input('check')
            # aaa = before_tanh.permute([0,2,3,1]).cpu().numpy()[check]
            # print(aaa.min(), aaa.max())

            # tanh_regular = 100 * (torch.clamp(torch.abs(before_tanh) - np.pi, min=0) ** 2).mean()

            self.alt = (self.alt + 1) % (self.switch * 2)
            return {'out': out_3d, 'up': out_upsample, 'to_inpaint': check}
            
        elif self.final_upsample // 10 == 9:

            print('final upsample with grad - add noise')
            out_upsample = torch.cat(out_upsample, dim=0).permute([0,3,1,2])

            part1 = self.final_up[0](out_upsample[:,:128])
            part2 = self.add_noise(out_upsample[:,128:])

            out_upsample = part1 + part2

            before_tanh = self.final_up[1:-1](out_upsample)
            out_upsample = self.final_up[-1](before_tanh)

            check_1 = (out_upsample.permute([0,2,3,1]) < -0.98431372549).cpu().numpy()
            check_2 = (out_upsample.permute([0,2,3,1]) > 0.98431372549).cpu().numpy()
            check = check_1 | check_2
            # checkimg = (1 - check.astype(np.uint8)) * 255
            # checkimg = [Image.fromarray(item) for item in checkimg]
            # checkimg[0].save('hehehe.gif', append_images=checkimg[1:], duration=250, loop=0, save_all=True)
            # input('check')
            # aaa = before_tanh.permute([0,2,3,1]).cpu().numpy()[check]
            # print(aaa.min(), aaa.max())

            # tanh_regular = 100 * (torch.clamp(torch.abs(before_tanh) - np.pi, min=0) ** 2).mean()

            self.alt = (self.alt + 1) % (self.switch * 2)
            return {'out': out_3d, 'up': out_upsample, 'to_inpaint': check}

        else:
            assert(False)
            pass

        return out_3d

    def forward_scn(self, im, coord, noise, info):

        pc_len, img_idx = info

        # print(im.shape) # B, N, 7
        # print(coord.shape) # B, N, 3
        # print(noise.shape) # B, 8, d_noise
        # print(pc_len) # B
        # print(img_idx.shape) # B, 15, H, W
        # print(img_idx.min(), img_idx.max())

        B, N, _ = im.size()

        batch_idx = torch.arange(B).float().reshape(B, 1, 1).expand(B, N, 1).to(coord.device)
        coord_4 = torch.cat([coord, batch_idx], dim=-1).reshape(B*N, 4)

        is_city = torch.sqrt((coord_4[:,:3]**2).sum(dim=-1)) < 600 # B*N
        is_sky = ~is_city # B*N

        loc = coord_4[is_city] # B*N, 4
        loc[:,:3] *= self.resolution
        loc[:,:3] += self.bias.to(loc.device)
        loc[:,:3] = torch.floor(loc[:,:3])
        loc = loc.long() # B*N, 4

        im_ft = im[:,:,:6].reshape(B*N, 6)
        if not self.input_coord:
            im_ft[:, 3:] *= 0

        self.sparse_net[2].set_noise(noise[:, -1]) # global noise
        sparse_input = (loc, im_ft[is_city], B)

        sparse_output = self.sparse_net(sparse_input) # V, d_middle
        # assert(not torch.isnan(sparse_output).any())

        # d_middle = sparse_output.size(1)
        # sparse_ft = torch.zeros(B*N, d_middle).to(sparse_output.device)
        # sparse_ft[is_city] = sparse_output
        # sparse_ft[is_sky] = self.init_sky[:3]
        sparse_ft = sparse_output.reshape(B, N, 3)

        return sparse_ft.permute([0, 2, 1])


    def forward_glb(self, im, coord, noise, info):

        pc_len, img_idx = info

        # print(im.shape) # B, N, 7
        # print(coord.shape) # B, N, 3
        # print(noise.shape) # B, 8, d_noise
        # print(pc_len) # B
        # print(img_idx.shape) # B, 15, H, W
        # print(img_idx.min(), img_idx.max())

        B, N, _ = im.size()

        batch_idx = torch.arange(B).float().reshape(B, 1, 1).expand(B, N, 1).to(coord.device)
        coord_4 = torch.cat([coord, batch_idx], dim=-1).reshape(B*N, 4)

        is_city = torch.sqrt((coord_4[:,:3]**2).sum(dim=-1)) < 600 # B*N
        is_sky = ~is_city # B*N

        loc = coord_4[is_city] # B*N, 4
        loc[:,:3] *= self.resolution
        loc[:,:3] += self.bias.to(loc.device)
        loc[:,:3] = torch.floor(loc[:,:3])
        loc = loc.long() # B*N, 4

        im_ft = im[:,:,:6].reshape(B*N, 6)
        if not self.input_coord:
            im_ft[:, 3:] *= 0

        self.sparse_net[2].set_noise(noise[:, -1]) # global noise
        sparse_input = (loc, im_ft[is_city], B)

        sparse_output = self.sparse_net(sparse_input) # V, d_middle
        # assert(not torch.isnan(sparse_output).any())

        d_middle = sparse_output.size(1)
        sparse_ft = torch.zeros(B*N, d_middle).to(sparse_output.device)
        sparse_ft[is_city] = sparse_output
        sparse_ft[is_sky] = self.init_sky
        sparse_ft = sparse_ft.reshape(B, N, d_middle)


        sem_idx = im[:,:,6].reshape(B*N).long()
        sem_idx[(coord_4[:,0] < 0) & (sem_idx == 1)] = 0 # is_left_building
        sem_idx = sem_idx.reshape(B, N, 1).expand(B, N, self.d_noise)
        sem_idx = sem_idx * 0 + 7 # all use global noise

        # shape of noise: B, 8, d_noise
        noise_ft = torch.gather(noise, 1, sem_idx) # B, N, d_noise
        im_3d = torch.cat([im[:,:,:3], sparse_ft], dim=2) # B, N, 3+d_middle

        out_3d = []
        for b in range(B):
            out_3d_pc, _ = self.net_3d.forward(
                im_3d[b:b+1, :pc_len[b]],
                coord[b:b+1, :pc_len[b]],
                noise_ft[b:b+1, :pc_len[b]]
            ) # 1, 3, len
            len_to_pad = N - out_3d_pc.size(2)
            out_3d_pc = torch.cat([out_3d_pc, out_3d_pc[:,:,:len_to_pad]], dim=-1) # 1, 3, N
            out_3d.append(out_3d_pc)
        out_3d = torch.cat(out_3d, dim=0)

        return out_3d

    def forward_rn(self, im, coord, noise, info):

        pc_len, img_idx = info

        # print(im.shape) # B, N, 7
        # print(coord.shape) # B, N, 3
        # print(noise.shape) # B, 8, d_noise
        # print(pc_len) # B
        # print(img_idx.shape) # B, 15, H, W
        # print(img_idx.min(), img_idx.max())

        B, N, _ = im.size()

        batch_idx = torch.arange(B).float().reshape(B, 1, 1).expand(B, N, 1).to(coord.device)
        coord_4 = torch.cat([coord, batch_idx], dim=-1).reshape(B*N, 4)

        im_ft = im[:,:,:6].reshape(B*N, 6)
        if not self.input_coord:
            im_ft[:, 3:] *= 0

        d_middle = self.init_sky.size(0)
        sparse_ft = self.init_sky.view(1, 1, d_middle).expand(B, N, d_middle)

        sem_idx = im[:,:,6].reshape(B*N).long()
        sem_idx[(coord_4[:,0] < 0) & (sem_idx == 1)] = 0 # is_left_building
        sem_idx = sem_idx.reshape(B, N, 1).expand(B, N, self.d_noise)
        sem_idx = sem_idx * 0 + 7 # all use global noise

        # shape of noise: B, 8, d_noise
        noise_ft = torch.gather(noise, 1, sem_idx) # B, N, d_noise
        im_3d = torch.cat([im[:,:,:3], sparse_ft], dim=2) # B, N, 3+d_middle

        out_3d = []
        for b in range(B):
            out_3d_pc, _ = self.net_3d.forward(
                im_3d[b:b+1, :pc_len[b]],
                coord[b:b+1, :pc_len[b]],
                noise_ft[b:b+1, :pc_len[b]]
            ) # 1, 3, len
            len_to_pad = N - out_3d_pc.size(2)
            out_3d_pc = torch.cat([out_3d_pc, out_3d_pc[:,:,:len_to_pad]], dim=-1) # 1, 3, N
            out_3d.append(out_3d_pc)
        out_3d = torch.cat(out_3d, dim=0)

        return out_3d





























class RandLANet2D(nn.Module):
    def __init__(self, d_in, d_out, d_noise, d_middle, num_downs, ngf=64,
                 norm_layer=None, nl_layer=None, use_dropout=False, upsample='basic',
                 input_coord=True):
        super(RandLANet2D, self).__init__()

        self.input_coord = input_coord
        self.net_3d = RandLANet(d_in+3, d_middle, d_noise, d_middle, num_neighbors=8, decimation=4,
            random_sel=1, local_noise=True, input_coord=input_coord)

        self.net_2d = G_Unet_add_all(d_middle+6, d_out, d_noise, 7, ngf, norm_layer=norm_layer, nl_layer=nl_layer,
            use_dropout=use_dropout, upsample=upsample)

        self.d_middle = d_middle
        self.d_noise = d_noise

        return


    def forward(self, im, coord, noise, info):

        pc_len, img_idx = info

        # print(im.shape) # B, N, 7
        # print(coord.shape) # B, N, 3
        # print(noise.shape) # B, 8, d_noise
        # print(pc_len) # B
        # print(img_idx.shape) # B, 15, H, W
        # print(img_idx.min(), img_idx.max())

        B, N, _ = im.size()

        batch_idx = torch.arange(B).float().reshape(B, 1, 1).expand(B, N, 1).to(coord.device)
        coord_4 = torch.cat([coord, batch_idx], dim=-1).reshape(B*N, 4)

        im_ft = im[:,:,:6].reshape(B*N, 6)
        if not self.input_coord:
            im_ft[:, 3:] *= 0

        sem_idx = im[:,:,6].reshape(B*N).long()
        sem_idx[(coord_4[:,0] < 0) & (sem_idx == 1)] = 0 # is_left_building
        sem_idx = sem_idx.reshape(B, N, 1).expand(B, N, self.d_noise)
        sem_idx = sem_idx * 0 + 7 # all use global noise

        # shape of noise: B, 8, d_noise
        noise_ft = torch.gather(noise, 1, sem_idx) # B, N, d_noise
        im_3d = im[:,:,:3] # B, N, 3

        out_3d = []
        for b in range(B):
            out_3d_pc, _ = self.net_3d.forward(
                im_3d[b:b+1, :pc_len[b]],
                coord[b:b+1, :pc_len[b]],
                noise_ft[b:b+1, :pc_len[b]]
            ) # 1, 3, len
            len_to_pad = N - out_3d_pc.size(2)
            out_3d_pc = torch.cat([out_3d_pc, out_3d_pc[:,:,:len_to_pad]], dim=-1) # 1, 3, N
            out_3d.append(out_3d_pc)
        out_3d = torch.cat(out_3d, dim=0).permute([0,2,1])
        out_3d = torch.cat([out_3d, im[:,:,:6]], dim=-1)
        # to shape 1, 220000, 32


        _, F, H, W = img_idx.size()
        ims = []
        for i in range(B):
            im = out_3d[i, :pc_len[i]] [img_idx[i].view(-1)].reshape(F,H,W,self.d_middle+6)
            ims.append(im.permute([0,3,1,2]))
        ims = torch.cat(ims)

        out_2d = self.net_2d(ims, noise_ft[0,0:F]).view(B, F, 3, H, W)

        return out_2d



































def define_G(input_nc, output_nc, nz, ngf, middle_nc=None, netG='unet_128', norm='batch', nl='relu',
             use_dropout=False, init_type='xavier', init_gain=0.02, gpu_ids=[], where_add='input', upsample='bilinear',use_spade=False,grid_sampling=False,input_coord=True,scn_ratio=32.0,rn_ratio=1.0,final_upsample=0):
    net = None
    norm_layer = get_norm_layer(norm_type=norm)
    nl_layer = get_non_linearity(layer_type=nl)

    if nz == 0:
        where_add = 'input'

    if netG == 'unet_128' and where_add == 'input':
        net = G_Unet_add_input(input_nc, output_nc, nz, 7, ngf, norm_layer=norm_layer, nl_layer=nl_layer,
                               use_dropout=use_dropout, upsample=upsample)
    elif netG == 'unet_256' and where_add == 'input':
        net = G_Unet_add_input(input_nc, output_nc, nz, 8, ngf, norm_layer=norm_layer, nl_layer=nl_layer,
                               use_dropout=use_dropout, upsample=upsample)
    elif netG == 'unet_128' and where_add == 'all':
        net = G_Unet_add_all(input_nc, output_nc, nz, 7, ngf, norm_layer=norm_layer, nl_layer=nl_layer,
                             use_dropout=use_dropout, upsample=upsample)
    elif netG == 'unet_256' and where_add == 'all':
        net = G_Unet_add_all(input_nc, output_nc, nz, 8, ngf, norm_layer=norm_layer, nl_layer=nl_layer,
                             use_dropout=use_dropout, upsample=upsample)
    elif netG == 'randlanet' and where_add == 'all':
        print('Using 3D conv for point cloud.')
        net = RandLANet(input_nc+3, output_nc, nz, middle_nc, num_neighbors=8, decimation=4)
    elif netG == 'hybrid' and where_add == 'all':
        print('Using 3D conv for point cloud and followed by 2D net.')
        net = HybridNet(input_nc, output_nc, nz, middle_nc, 8, ngf, norm_layer=norm_layer, nl_layer=nl_layer,
                        use_dropout=use_dropout, upsample=upsample)
    elif netG == 'newhybrid' and where_add == 'all':
        print('Using 2D conv followed by 3D net.')
        net = HybridNet(input_nc, output_nc, nz, middle_nc, 8, ngf, norm_layer=norm_layer, nl_layer=nl_layer,
                        use_dropout=use_dropout, upsample=upsample)
    elif netG == 'sparseconvnet' and where_add == 'all':
        print('Using sparse conv followed by randlanet.')
        net = SparseConvNet(input_nc, output_nc, nz, middle_nc, 8, ngf, norm_layer=norm_layer, nl_layer=nl_layer,
                        use_dropout=use_dropout, upsample=upsample)
    elif netG == 'sparseconvnetmultinoise' and where_add == 'all':
        print('Using sparse conv net followed by randlanet, using local noise.')
        net = SparseConvNetMultiNoise(input_nc, output_nc, nz, middle_nc, 8, ngf, norm_layer=norm_layer, nl_layer=nl_layer,
                        use_dropout=use_dropout, upsample=upsample, use_spade=use_spade, grid_sampling=grid_sampling, input_coord=input_coord,
                        scn_ratio=scn_ratio, rn_ratio=rn_ratio, final_upsample=final_upsample)
    elif netG == 'randlanet2d' and where_add == 'all':
        print('Using randlanet followed by 2D generator, using global noise.')
        net = RandLANet2D(input_nc, output_nc, nz, middle_nc, 7, ngf, norm_layer=norm_layer, nl_layer=nl_layer,
                        use_dropout=use_dropout, upsample=upsample, input_coord=input_coord)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % net)

    return init_net(net, init_type, init_gain, gpu_ids)


def define_D(input_nc, ndf, netD, norm='batch', nl='lrelu', init_type='xavier', init_gain=0.02, num_Ds=1, gpu_ids=[]):
    net = None
    norm_layer = get_norm_layer(norm_type=norm)
    nl = 'lrelu'  # use leaky relu for D
    nl_layer = get_non_linearity(layer_type=nl)

    if netD == 'basic_128':
        net = D_NLayers(input_nc, ndf, n_layers=2, norm_layer=norm_layer, nl_layer=nl_layer)
    elif netD == 'basic_256':
        net = D_NLayers(input_nc, ndf, n_layers=3, norm_layer=norm_layer, nl_layer=nl_layer)
    elif netD == 'basic_128_multi':
        net = D_NLayersMulti(input_nc=input_nc, ndf=ndf, n_layers=2, norm_layer=norm_layer, num_D=num_Ds)
    elif netD == 'basic_256_multi':
        net = D_NLayersMulti(input_nc=input_nc, ndf=ndf, n_layers=3, norm_layer=norm_layer, num_D=num_Ds)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' % net)
    return init_net(net, init_type, init_gain, gpu_ids)


def define_E(input_nc, output_nc, ndf, netE,
             norm='batch', nl='lrelu',
             init_type='xavier', init_gain=0.02, gpu_ids=[], vaeLike=False):
    net = None
    norm_layer = get_norm_layer(norm_type=norm)
    nl = 'lrelu'  # use leaky relu for E
    nl_layer = get_non_linearity(layer_type=nl)
    if netE == 'resnet_128':
        net = E_ResNet(input_nc, output_nc, ndf, n_blocks=4, norm_layer=norm_layer,
                       nl_layer=nl_layer, vaeLike=vaeLike)
    elif netE == 'resnet_256':
        net = E_ResNet(input_nc, output_nc, ndf, n_blocks=5, norm_layer=norm_layer,
                       nl_layer=nl_layer, vaeLike=vaeLike)
    elif netE == 'conv_128':
        net = E_NLayers(input_nc, output_nc, ndf, n_layers=4, norm_layer=norm_layer,
                        nl_layer=nl_layer, vaeLike=vaeLike)
    elif netE == 'conv_256':
        net = E_NLayers(input_nc, output_nc, ndf, n_layers=5, norm_layer=norm_layer,
                        nl_layer=nl_layer, vaeLike=vaeLike)
    elif netE == 'resnet_256_multi':
        net = E_ResNet_Multi(input_nc, output_nc, ndf, n_blocks=5, norm_layer=norm_layer,
                       nl_layer=nl_layer, vaeLike=vaeLike)
    elif netE == 'resnet_128_multi':
        net = E_ResNet_Multi(input_nc, output_nc, ndf, n_blocks=4, norm_layer=norm_layer,
                       nl_layer=nl_layer, vaeLike=vaeLike)
    else:
        raise NotImplementedError('Encoder model name [%s] is not recognized' % net)

    return init_net(net, init_type, init_gain, gpu_ids)


class D_NLayersMulti(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3,
                 norm_layer=nn.BatchNorm2d, num_D=1):
        super(D_NLayersMulti, self).__init__()
        # st()
        self.num_D = num_D
        if num_D == 1:
            layers = self.get_layers(input_nc, ndf, n_layers, norm_layer)
            self.model = nn.Sequential(*layers)
        else:
            layers = self.get_layers(input_nc, ndf, n_layers, norm_layer)
            self.add_module("model_0", nn.Sequential(*layers))
            self.down = nn.AvgPool2d(3, stride=2, padding=[
                                     1, 1], count_include_pad=False)
            for i in range(1, num_D):
                ndf_i = int(round(ndf / (2**i)))
                layers = self.get_layers(input_nc, ndf_i, n_layers, norm_layer)
                self.add_module("model_%d" % i, nn.Sequential(*layers))

    def get_layers(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw,
                              stride=2, padding=padw), nn.LeakyReLU(0.2, True)]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=1, padding=padw),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1,
                               kernel_size=kw, stride=1, padding=padw)]

        return sequence

    def forward(self, input):
        if self.num_D == 1:
            return self.model(input)
        result = []
        down = input
        for i in range(self.num_D):
            model = getattr(self, "model_%d" % i)
            result.append(model(down))
            if i != self.num_D - 1:
                down = self.down(down)
        return result


class D_NLayers(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        """Construct a PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(D_NLayers, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func != nn.BatchNorm2d
        else:
            use_bias = norm_layer != nn.BatchNorm2d

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)


##############################################################################
# Classes
##############################################################################
class RecLoss(nn.Module):
    def __init__(self, use_L2=True):
        super(RecLoss, self).__init__()
        self.use_L2 = use_L2

    def __call__(self, input, target, batch_mean=True):
        if self.use_L2:
            diff = (input - target) ** 2
        else:
            diff = torch.abs(input - target)
        if batch_mean:
            return torch.mean(diff)
        else:
            return torch.mean(torch.mean(torch.mean(diff, dim=1), dim=2), dim=3)


# Defines the GAN loss which uses either LSGAN or the regular GAN.
# When LSGAN is used, it is basically same as MSELoss,
# but it abstracts away the need to create the target label tensor
# that has the same size as the input
class GANLoss(nn.Module):
    """Define different GAN objectives.

    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.

        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image

        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.

        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, predictions, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.

        Parameters:
            prediction (tensor list) - - tpyically the prediction output from a discriminator; supports multi Ds.
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            the calculated loss.
        """
        all_losses = []
        for prediction in predictions:
            if self.gan_mode in ['lsgan', 'vanilla']:
                target_tensor = self.get_target_tensor(prediction, target_is_real)
                loss = self.loss(prediction, target_tensor)
            elif self.gan_mode == 'wgangp':
                if target_is_real:
                    loss = -prediction.mean()
                else:
                    loss = prediction.mean()
            all_losses.append(loss)
        total_loss = sum(all_losses)
        return total_loss, all_losses


def cal_gradient_penalty(netD, real_data, fake_data, device, type='mixed', constant=1.0, lambda_gp=10.0):
    """Calculate the gradient penalty loss, used in WGAN-GP paper https://arxiv.org/abs/1704.00028
    Arguments:
        netD (network)              -- discriminator network
        real_data (tensor array)    -- real images
        fake_data (tensor array)    -- generated images from the generator
        device (str)                -- GPU / CPU: from torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        type (str)                  -- if we mix real and fake data or not [real | fake | mixed].
        constant (float)            -- the constant used in formula ( | |gradient||_2 - constant)^2
        lambda_gp (float)           -- weight for this loss
    Returns the gradient penalty loss
    """
    if lambda_gp > 0.0:
        if type == 'real':   # either use real images, fake images, or a linear interpolation of two.
            interpolatesv = real_data
        elif type == 'fake':
            interpolatesv = fake_data
        elif type == 'mixed':
            alpha = torch.rand(real_data.shape[0], 1)
            alpha = alpha.expand(real_data.shape[0], real_data.nelement() // real_data.shape[0]).contiguous().view(*real_data.shape)
            alpha = alpha.to(device)
            interpolatesv = alpha * real_data + ((1 - alpha) * fake_data)
        else:
            raise NotImplementedError('{} not implemented'.format(type))
        interpolatesv.requires_grad_(True)
        disc_interpolates = netD(interpolatesv)
        gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolatesv,
                                        grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                                        create_graph=True, retain_graph=True, only_inputs=True)
        gradients = gradients[0].view(real_data.size(0), -1)  # flat the data
        gradient_penalty = (((gradients + 1e-16).norm(2, dim=1) - constant) ** 2).mean() * lambda_gp        # added eps
        return gradient_penalty, gradients
    else:
        return 0.0, None

# Defines the Unet generator.
# |num_downs|: number of downsamplings in UNet. For example,
# if |num_downs| == 7, image of size 128x128 will become of size 1x1
# at the bottleneck


class G_Unet_add_input(nn.Module):
    def __init__(self, input_nc, output_nc, nz, num_downs, ngf=64,
                 norm_layer=None, nl_layer=None, use_dropout=False,
                 upsample='basic'):
        super(G_Unet_add_input, self).__init__()
        self.nz = nz
        max_nchn = 8
        # construct unet structure
        unet_block = UnetBlock(ngf * max_nchn, ngf * max_nchn, ngf * max_nchn,
                               innermost=True, norm_layer=norm_layer, nl_layer=nl_layer, upsample=upsample)
        for i in range(num_downs - 5):
            unet_block = UnetBlock(ngf * max_nchn, ngf * max_nchn, ngf * max_nchn, unet_block,
                                   norm_layer=norm_layer, nl_layer=nl_layer, use_dropout=use_dropout, upsample=upsample)
        unet_block = UnetBlock(ngf * 4, ngf * 4, ngf * max_nchn, unet_block,
                               norm_layer=norm_layer, nl_layer=nl_layer, upsample=upsample)
        unet_block = UnetBlock(ngf * 2, ngf * 2, ngf * 4, unet_block,
                               norm_layer=norm_layer, nl_layer=nl_layer, upsample=upsample)
        unet_block = UnetBlock(ngf, ngf, ngf * 2, unet_block,
                               norm_layer=norm_layer, nl_layer=nl_layer, upsample=upsample)
        unet_block = UnetBlock(input_nc + nz, output_nc, ngf, unet_block,
                               outermost=True, norm_layer=norm_layer, nl_layer=nl_layer, upsample=upsample)

        self.model = unet_block

    def forward(self, x, z=None):
        if self.nz > 0:
            z_img = z.view(z.size(0), z.size(1), 1, 1).expand(
                z.size(0), z.size(1), x.size(2), x.size(3))
            x_with_z = torch.cat([x, z_img], 1)
        else:
            x_with_z = x  # no z

        return self.model(x_with_z)


def upsampleLayer(inplanes, outplanes, upsample='basic', padding_type='zero'):
    # padding_type = 'zero'
    if upsample == 'basic':
        upconv = [nn.ConvTranspose2d(
            inplanes, outplanes, kernel_size=4, stride=2, padding=1)]
    elif upsample == 'bilinear':
        upconv = [nn.Upsample(scale_factor=2, mode='bilinear'),
                  nn.ReflectionPad2d(1),
                  nn.Conv2d(inplanes, outplanes, kernel_size=3, stride=1, padding=0)]
    else:
        raise NotImplementedError(
            'upsample layer [%s] not implemented' % upsample)
    return upconv


# Defines the submodule with skip connection.
# X -------------------identity---------------------- X
#   |-- downsampling -- |submodule| -- upsampling --|
class UnetBlock(nn.Module):
    def __init__(self, input_nc, outer_nc, inner_nc,
                 submodule=None, outermost=False, innermost=False,
                 norm_layer=None, nl_layer=None, use_dropout=False, upsample='basic', padding_type='zero'):
        super(UnetBlock, self).__init__()
        self.outermost = outermost
        p = 0
        downconv = []
        if padding_type == 'reflect':
            downconv += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            downconv += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError(
                'padding [%s] is not implemented' % padding_type)
        downconv += [nn.Conv2d(input_nc, inner_nc,
                               kernel_size=4, stride=2, padding=p)]
        # downsample is different from upsample
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc) if norm_layer is not None else None
        uprelu = nl_layer()
        upnorm = norm_layer(outer_nc) if norm_layer is not None else None

        if outermost:
            upconv = upsampleLayer(
                inner_nc * 2, outer_nc, upsample=upsample, padding_type=padding_type)
            down = downconv
            up = [uprelu] + upconv + [nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = upsampleLayer(
                inner_nc, outer_nc, upsample=upsample, padding_type=padding_type)
            down = [downrelu] + downconv
            up = [uprelu] + upconv
            if upnorm is not None:
                up += [upnorm]
            model = down + up
        else:
            upconv = upsampleLayer(
                inner_nc * 2, outer_nc, upsample=upsample, padding_type=padding_type)
            down = [downrelu] + downconv
            if downnorm is not None:
                down += [downnorm]
            up = [uprelu] + upconv
            if upnorm is not None:
                up += [upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            return torch.cat([self.model(x), x], 1)


def conv3x3(in_planes, out_planes):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1,
                     padding=1, bias=True)


# two usage cases, depend on kw and padw
def upsampleConv(inplanes, outplanes, kw, padw):
    sequence = []
    sequence += [nn.Upsample(scale_factor=2, mode='nearest')]
    sequence += [nn.Conv2d(inplanes, outplanes, kernel_size=kw,
                           stride=1, padding=padw, bias=True)]
    return nn.Sequential(*sequence)


def meanpoolConv(inplanes, outplanes):
    sequence = []
    sequence += [nn.AvgPool2d(kernel_size=2, stride=2)]
    sequence += [nn.Conv2d(inplanes, outplanes,
                           kernel_size=1, stride=1, padding=0, bias=True)]
    return nn.Sequential(*sequence)


def convMeanpool(inplanes, outplanes):
    sequence = []
    sequence += [conv3x3(inplanes, outplanes)]
    sequence += [nn.AvgPool2d(kernel_size=2, stride=2)]
    return nn.Sequential(*sequence)


class BasicBlockUp(nn.Module):
    def __init__(self, inplanes, outplanes, norm_layer=None, nl_layer=None):
        super(BasicBlockUp, self).__init__()
        layers = []
        if norm_layer is not None:
            layers += [norm_layer(inplanes)]
        layers += [nl_layer()]
        layers += [upsampleConv(inplanes, outplanes, kw=3, padw=1)]
        if norm_layer is not None:
            layers += [norm_layer(outplanes)]
        layers += [conv3x3(outplanes, outplanes)]
        self.conv = nn.Sequential(*layers)
        self.shortcut = upsampleConv(inplanes, outplanes, kw=1, padw=0)

    def forward(self, x):
        out = self.conv(x) + self.shortcut(x)
        return out


class BasicBlock(nn.Module):
    def __init__(self, inplanes, outplanes, norm_layer=None, nl_layer=None):
        super(BasicBlock, self).__init__()
        layers = []
        if norm_layer is not None:
            layers += [norm_layer(inplanes)]
        layers += [nl_layer()]
        layers += [conv3x3(inplanes, inplanes)]
        if norm_layer is not None:
            layers += [norm_layer(inplanes)]
        layers += [nl_layer()]
        layers += [convMeanpool(inplanes, outplanes)]
        self.conv = nn.Sequential(*layers)
        self.shortcut = meanpoolConv(inplanes, outplanes)

    def forward(self, x):
        out = self.conv(x) + self.shortcut(x)
        return out


class E_ResNet(nn.Module):
    def __init__(self, input_nc=3, output_nc=1, ndf=64, n_blocks=4,
                 norm_layer=None, nl_layer=None, vaeLike=False):
        super(E_ResNet, self).__init__()
        self.vaeLike = vaeLike
        max_ndf = 4
        conv_layers = [
            nn.Conv2d(input_nc, ndf, kernel_size=4, stride=2, padding=1, bias=True)]
        for n in range(1, n_blocks):
            input_ndf = ndf * min(max_ndf, n)
            output_ndf = ndf * min(max_ndf, n + 1)
            conv_layers += [BasicBlock(input_ndf,
                                       output_ndf, norm_layer, nl_layer)]
        conv_layers += [nl_layer(), nn.AvgPool2d(8)]
        if vaeLike:
            self.fc = nn.Sequential(*[nn.Linear(output_ndf, output_nc)])
            self.fcVar = nn.Sequential(*[nn.Linear(output_ndf, output_nc)])
        else:
            self.fc = nn.Sequential(*[nn.Linear(output_ndf, output_nc)])
        self.conv = nn.Sequential(*conv_layers)

    def forward(self, x):
        x_conv = self.conv(x)
        conv_flat = x_conv.view(x.size(0), -1)
        output = self.fc(conv_flat)
        if self.vaeLike:
            outputVar = self.fcVar(conv_flat)
            return output, outputVar
        else:
            return output
        return output


# Defines the Unet generator.
# |num_downs|: number of downsamplings in UNet. For example,
# if |num_downs| == 7, image of size 128x128 will become of size 1x1
# at the bottleneck
class G_Unet_add_all(nn.Module):
    def __init__(self, input_nc, output_nc, nz, num_downs, ngf=64,
                 norm_layer=None, nl_layer=None, use_dropout=False, upsample='basic'):
        super(G_Unet_add_all, self).__init__()
        self.nz = nz
        # construct unet structure
        unet_block = UnetBlock_with_z(ngf * 8, ngf * 8, ngf * 8, nz, None, innermost=True,
                                      norm_layer=norm_layer, nl_layer=nl_layer, upsample=upsample)
        unet_block = UnetBlock_with_z(ngf * 8, ngf * 8, ngf * 8, nz, unet_block,
                                      norm_layer=norm_layer, nl_layer=nl_layer, use_dropout=use_dropout, upsample=upsample)
        for i in range(num_downs - 6):
            unet_block = UnetBlock_with_z(ngf * 8, ngf * 8, ngf * 8, nz, unet_block,
                                          norm_layer=norm_layer, nl_layer=nl_layer, use_dropout=use_dropout, upsample=upsample)
        unet_block = UnetBlock_with_z(ngf * 4, ngf * 4, ngf * 8, nz, unet_block,
                                      norm_layer=norm_layer, nl_layer=nl_layer, upsample=upsample)
        unet_block = UnetBlock_with_z(ngf * 2, ngf * 2, ngf * 4, nz, unet_block,
                                      norm_layer=norm_layer, nl_layer=nl_layer, upsample=upsample)
        unet_block = UnetBlock_with_z(
            ngf, ngf, ngf * 2, nz, unet_block, norm_layer=norm_layer, nl_layer=nl_layer, upsample=upsample)
        unet_block = UnetBlock_with_z(input_nc, output_nc, ngf, nz, unet_block,
                                      outermost=True, norm_layer=norm_layer, nl_layer=nl_layer, upsample=upsample)
        self.model = unet_block

    def forward(self, x, z):
        return self.model(x, z)


class UnetBlock_with_z(nn.Module):
    def __init__(self, input_nc, outer_nc, inner_nc, nz=0,
                 submodule=None, outermost=False, innermost=False,
                 norm_layer=None, nl_layer=None, use_dropout=False, upsample='basic', padding_type='zero'):
        super(UnetBlock_with_z, self).__init__()
        p = 0
        downconv = []
        if padding_type == 'reflect':
            downconv += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            downconv += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError(
                'padding [%s] is not implemented' % padding_type)

        self.outermost = outermost
        self.innermost = innermost
        self.nz = nz
        input_nc = input_nc + nz
        downconv += [nn.Conv2d(input_nc, inner_nc,
                               kernel_size=4, stride=2, padding=p)]
        # downsample is different from upsample
        downrelu = nn.LeakyReLU(0.2, True)
        uprelu = nl_layer()

        if outermost:
            upconv = upsampleLayer(
                inner_nc * 2, outer_nc, upsample=upsample, padding_type=padding_type)
            down = downconv
            up = [uprelu] + upconv + [nn.Tanh()]
        elif innermost:
            upconv = upsampleLayer(
                inner_nc, outer_nc, upsample=upsample, padding_type=padding_type)
            down = [downrelu] + downconv
            up = [uprelu] + upconv
            if norm_layer is not None:
                up += [norm_layer(outer_nc)]
        else:
            upconv = upsampleLayer(
                inner_nc * 2, outer_nc, upsample=upsample, padding_type=padding_type)
            down = [downrelu] + downconv
            if norm_layer is not None:
                down += [norm_layer(inner_nc)]
            up = [uprelu] + upconv

            if norm_layer is not None:
                up += [norm_layer(outer_nc)]

            if use_dropout:
                up += [nn.Dropout(0.5)]
        self.down = nn.Sequential(*down)
        self.submodule = submodule
        self.up = nn.Sequential(*up)

    def forward(self, x, z):
        # print(x.size())
        if self.nz > 0:
            z_img = z.view(z.size(0), z.size(1), 1, 1).expand(z.size(0), z.size(1), x.size(2), x.size(3))
            x_and_z = torch.cat([x, z_img], 1)
        else:
            x_and_z = x

        if self.outermost:
            x1 = self.down(x_and_z)
            x2 = self.submodule(x1, z)
            return self.up(x2)
        elif self.innermost:
            x1 = self.up(self.down(x_and_z))
            return torch.cat([x1, x], 1)
        else:
            x1 = self.down(x_and_z)
            x2 = self.submodule(x1, z)
            return torch.cat([self.up(x2), x], 1)


class E_NLayers(nn.Module):
    def __init__(self, input_nc, output_nc=1, ndf=64, n_layers=3,
                 norm_layer=None, nl_layer=None, vaeLike=False):
        super(E_NLayers, self).__init__()
        self.vaeLike = vaeLike

        kw, padw = 4, 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw,
                              stride=2, padding=padw), nl_layer()]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 4)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw)]
            if norm_layer is not None:
                sequence += [norm_layer(ndf * nf_mult)]
            sequence += [nl_layer()]
        sequence += [nn.AvgPool2d(8)]
        self.conv = nn.Sequential(*sequence)
        self.fc = nn.Sequential(*[nn.Linear(ndf * nf_mult, output_nc)])
        if vaeLike:
            self.fcVar = nn.Sequential(*[nn.Linear(ndf * nf_mult, output_nc)])

    def forward(self, x):
        x_conv = self.conv(x)
        conv_flat = x_conv.view(x.size(0), -1)
        output = self.fc(conv_flat)
        if self.vaeLike:
            outputVar = self.fcVar(conv_flat)
            return output, outputVar
        return output

class E_ResNet_Multi(nn.Module):
    def __init__(self, input_nc=3, output_nc=1, ndf=64, n_blocks=4,
                 norm_layer=None, nl_layer=None, vaeLike=False):
        super(E_ResNet_Multi, self).__init__()
        self.num_cls = 6
        self.output_nc = output_nc
        self.vaeLike = vaeLike
        max_ndf = 4
        conv_layers = [
            nn.Conv2d(input_nc, ndf, kernel_size=4, stride=2, padding=1, bias=True)
        ]
        for n in range(1, n_blocks):
            input_ndf = ndf * min(max_ndf, n)
            output_ndf = ndf * min(max_ndf, n + 1)
            conv_layers += [BasicBlock(input_ndf,
                                       output_ndf, norm_layer, nl_layer)]
        conv_layers += [nl_layer()]

        attention_layers = [
            nn.Conv2d(output_ndf, self.num_cls, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Softmax(dim=1)
        ]

        self.conv = nn.Sequential(*conv_layers)
        self.attention = nn.Sequential(*attention_layers)

        self.fc = nn.Conv2d(output_ndf, output_nc, kernel_size=1, stride=1, padding=0, bias=True)
        if vaeLike:
            self.fcVar = nn.Conv2d(output_ndf, output_nc, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):
        x_conv = self.conv(x) # B, 256, 8, 16
        half = x_conv.size(3) // 2
        x_att = torch.clamp(self.attention(x_conv), 0.0001, 0.9999) # B, n_cls, 8, 16
        x_att = torch.cat([x_att[:, :1], x_att], dim=1)
        x_att[:, 0, :, half:] *= 0 # left buildings
        x_att[:, 1, :, :half] *= 0 # right buildings
        x_ft = []
        for i in range(self.num_cls+1):
            w = x_att[:, i:i+1] # B, 1, 8, 16
            w_sum = torch.sum(torch.sum(w, dim=-1), dim=-1) # B, 1
            x_sum = torch.sum(torch.sum(w * x_conv, dim=-1), dim=-1) # B, 256
            x_ft.append((x_sum / w_sum).unsqueeze(-1).unsqueeze(-1)) # B, 256, 1, 1
        x_ft.append(F.avg_pool2d(x_conv, kernel_size=(x_conv.size(2), x_conv.size(3)))) # B, 256, 1, 1
        x_ft = torch.cat(x_ft, dim=-2) # B, 256, n_cls+2, 1

        output = self.fc(x_ft).squeeze(-1).transpose(-1, -2) # B, n_cls+2, d
        if self.vaeLike:
            outputVar = self.fcVar(x_ft).squeeze(-1).transpose(-1, -2) # B, n_cls+2, d
            return output, outputVar
        else:
            return output
