import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_points import knn
except (ModuleNotFoundError, ImportError):
    from torch_points_kernels import knn

def knn_func(coords1, coords2, nb, device, random_sel=1):
    assert(coords1.device == torch.device('cpu'))
    assert(coords2.device == torch.device('cpu'))
    out1, out2 = knn(coords1.contiguous(), coords2.contiguous(), nb*random_sel)
    if random_sel > 1:
        sel = torch.randperm(nb*random_sel)
        out1 = out1[:,:,sel]
        out2 = out2[:,:,sel]
    return out1.to(device), out2.to(device)

class Sine(nn.Module):
    def __init__(self, w0=32.0, inplace=False):
        super(Sine, self).__init__()
        self.w0 = w0
    def forward(self, x):
        return torch.sin(self.w0 * x)

class SharedMLP(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        transpose=False,
        padding_mode='zeros',
        bn=False,
        activation_fn=None,
        use_spade=False
    ):
        super(SharedMLP, self).__init__()

        conv_fn = nn.ConvTranspose2d if transpose else nn.Conv2d

        self.conv = conv_fn(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding_mode=padding_mode
        )
        # self.batch_norm = nn.BatchNorm2d(out_channels, eps=1e-6, momentum=0.99) if bn else None
        self.use_spade = use_spade
        if use_spade:
            self.batch_norm = nn.BatchNorm2d(out_channels, eps=1e-6, momentum=0.99, affine=False)
        else:
            self.batch_norm = nn.GroupNorm(16, out_channels, eps=1e-6) if bn else None
        self.activation_fn = activation_fn

    def forward(self, input, gamma=None, beta=None):
        r"""
            Forward pass of the network

            Parameters
            ----------
            input: torch.Tensor, shape (B, d_in, N, K)

            Returns
            -------
            torch.Tensor, shape (B, d_out, N, K)
        """

        x = self.conv(input)
        if self.batch_norm:
            x = self.batch_norm(x)
        if self.use_spade:
            x = x * gamma + beta
        if self.activation_fn:
            x = self.activation_fn(x)
        return x


class LocalSpatialEncoding(nn.Module):
    def __init__(self, d, num_neighbors):
        super(LocalSpatialEncoding, self).__init__()

        self.num_neighbors = num_neighbors
        self.mlp = SharedMLP(10, d, bn=True, activation_fn=nn.ReLU())

    def forward_bug(self, coords, features, knn_output):
        r"""
            Forward pass

            Parameters
            ----------
            coords: torch.Tensor, shape (B, N, 3)
                coordinates of the point cloud
            features: torch.Tensor, shape (B, d, N, 1)
                features of the point cloud
            neighbors: tuple

            Returns
            -------
            torch.Tensor, shape (B, 2*d, N, K)
        """
        # finding neighboring points
        idx, dist = knn_output

        B, N, K = idx.size()
        # idx(B, N, K), coords(B, N, 3)
        # neighbors[b, i, n, k] = coords[b, idx[b, n, k], i] = extended_coords[b, i, extended_idx[b, i, n, k], k]
        extended_idx = idx.unsqueeze(1).expand(B, 3, N, K)
        extended_coords = coords.transpose(-2,-1).unsqueeze(-1).expand(B, 3, N, K)
        neighbors = torch.gather(extended_coords, 2, extended_idx) # shape (B, 3, N, K)
        # if USE_CUDA:
        #     neighbors = neighbors.cuda()

        # relative point position encoding
        concat = torch.cat((
            extended_coords,
            neighbors,
            extended_coords - neighbors,
            dist.unsqueeze(-3)
        ), dim=-3)

        res = torch.cat((
            self.mlp(concat),
            features.expand(B, -1, N, K)
        ), dim=-3)

        return res
    
    def forward(self, coords, features, knn_output):
        r"""
            Forward pass
            Parameters
            ----------
            coords: torch.Tensor, shape (B, N, 3)
                coordinates of the point cloud
            features: torch.Tensor, shape (B, d, N, 1)
                features of the point cloud
            neighbors: tuple
            Returns
            -------
            torch.Tensor, shape (B, 2*d, N, K)
        """
        # finding neighboring points
        idx, dist = knn_output
        B, N, K = idx.size()
        # idx(B, N, K), coords(B, N, 3)
        # neighbors[b, i, n, k] = coords[b, idx[b, n, k], i] = expanded_coords[b, i, extended_idx[b, i, n, k], k]
        expanded_idx = idx.unsqueeze(1).expand(B, 3, N, K)
        expanded_coords = coords.transpose(-2,-1).unsqueeze(-1).expand(B, 3, N, K)
        neighbor_coords = torch.gather(expanded_coords, 2, expanded_idx) # shape (B, 3, N, K)

        expanded_idx = idx.unsqueeze(1).expand(B, features.size(1), N, K)
        expanded_features = features.expand(B, -1, N, K)
        neighbor_features = torch.gather(expanded_features, 2, expanded_idx)
        # if USE_CUDA:
        #     neighbors = neighbors.cuda()

        # relative point position encoding
        concat = torch.cat((
            expanded_coords,
            neighbor_coords,
            expanded_coords - neighbor_coords,
            dist.unsqueeze(-3)
        ), dim=-3).to(features.device)
        return torch.cat((
            self.mlp(concat),
            neighbor_features
        ), dim=-3)



class AttentivePooling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(AttentivePooling, self).__init__()

        # self.score_fn = nn.Sequential(
        #     nn.Linear(in_channels, in_channels, bias=False),
        #     nn.Softmax(dim=-2)
        # )

        self.mlp = SharedMLP(in_channels, out_channels, bn=True, activation_fn=nn.ReLU())

    def forward(self, x):
        r"""
            Forward pass

            Parameters
            ----------
            x: torch.Tensor, shape (B, d_in, N, K)

            Returns
            -------
            torch.Tensor, shape (B, d_out, N, 1)
        """
        # computing attention scores
        # scores = self.score_fn(x.permute(0,2,3,1)).permute(0,3,1,2)

        # sum over the neighbors
        # features = torch.sum(scores * x, dim=-1, keepdim=True) # shape (B, d_in, N, 1)

        # Zuoyue: use hard max instead
        # features, _ = torch.max(x, dim=-1, keepdim=True)

        # Zuoyue: use mean pooling instead
        features = torch.mean(x, dim=-1, keepdim=True)

        return self.mlp(features)



class LocalFeatureAggregation(nn.Module):
    def __init__(self, d_in, d_out, num_neighbors):
        super(LocalFeatureAggregation, self).__init__()

        self.num_neighbors = num_neighbors

        self.mlp1 = SharedMLP(d_in, d_out//2, activation_fn=nn.LeakyReLU(0.2))
        self.mlp2 = SharedMLP(d_out, 2*d_out)
        self.shortcut = SharedMLP(d_in, 2*d_out, bn=True)

        self.lse1 = LocalSpatialEncoding(d_out//2, num_neighbors)
        self.lse2 = LocalSpatialEncoding(d_out//2, num_neighbors)

        self.pool1 = AttentivePooling(d_out, d_out//2)
        self.pool2 = AttentivePooling(d_out, d_out)

        self.lrelu = nn.LeakyReLU()

    def forward(self, coords, features, random_sel=1):
        r"""
            Forward pass

            Parameters
            ----------
            coords: torch.Tensor, shape (B, N, 3)
                coordinates of the point cloud
            features: torch.Tensor, shape (B, d_in, N, 1)
                features of the point cloud

            Returns
            -------
            torch.Tensor, shape (B, 2*d_out, N, 1)
        """
        knn_output = knn_func(coords, coords, self.num_neighbors, features.device, random_sel=random_sel)
        # knn_output = knn(coords.cpu().contiguous(), coords.cpu().contiguous(), self.num_neighbors)
        coords = coords.to(features.device)

        x = self.mlp1(features)
        x = self.lse1(coords, x, knn_output)
        x = self.pool1(x)
        x = self.lse2(coords, x, knn_output)
        x = self.pool2(x)
        res = self.lrelu(self.mlp2(x) + self.shortcut(features))
        
        return res





class Spade(nn.Module):
    def __init__(self, d_in, d_out, num_neighbors):
        super(Spade, self).__init__()

        self.num_neighbors = num_neighbors

        self.mlp1 = SharedMLP(d_in, d_out//2, activation_fn=nn.LeakyReLU(0.2))

        self.mlp_gamma = SharedMLP(d_out, d_out)
        self.mlp_beta = SharedMLP(d_out, d_out)

        self.shortcut = SharedMLP(d_in, d_out, bn=True)

        self.lse1 = LocalSpatialEncoding(d_out//2, num_neighbors)
        self.lse2 = LocalSpatialEncoding(d_out//2, num_neighbors)

        self.pool1 = AttentivePooling(d_out, d_out//2)
        self.pool2 = AttentivePooling(d_out, d_out)



    def forward(self, coords, features, random_sel=1):
        r"""
            Forward pass

            Parameters
            ----------
            coords: torch.Tensor, shape (B, N, 3)
                coordinates of the point cloud
            features: torch.Tensor, shape (B, d_in, N, 1)
                features of the point cloud

            Returns
            -------
            torch.Tensor, shape (B, 2*d_out, N, 1)
        """
        knn_output = knn_func(coords, coords, self.num_neighbors, features.device, random_sel=random_sel)
        # knn_output = knn(coords.cpu().contiguous(), coords.cpu().contiguous(), self.num_neighbors)
        coords = coords.to(features.device)

        x = self.mlp1(features)
        x = self.lse1(coords, x, knn_output)
        x = self.pool1(x)
        x = self.lse2(coords, x, knn_output)
        x = self.pool2(x)
        shortcut = self.shortcut(features)
        gamma = self.mlp_gamma(x) + shortcut
        beta = self.mlp_beta(x) + shortcut
        
        return gamma, beta












class RandLANet(nn.Module):
    def __init__(self, d_in, d_out, d_noise, d_middle, num_neighbors=8, decimation=4, random_sel=1, local_noise=False, use_spade=False, grid_sampling=False, input_coord=True, ratio=1.0):
        super(RandLANet, self).__init__()
        self.num_neighbors = num_neighbors
        self.decimation = decimation
        self.d_noise = d_noise
        self.d_middle = d_middle
        self.random_sel = random_sel
        self.grid_sampling = grid_sampling
        self.input_coord = input_coord
        self.ratio = ratio

        self.fc_start = nn.Linear(d_in+d_noise, 64)
        self.bn_start = nn.Sequential(
            # nn.BatchNorm2d(32, eps=1e-6, momentum=0.99),
            nn.GroupNorm(16, 64, eps=1e-6),
            nn.LeakyReLU(0.2)
        )

        # encoding layers
        self.encoder = nn.ModuleList([
            LocalFeatureAggregation(64+d_noise, int(64/2), num_neighbors),
            LocalFeatureAggregation(64+d_noise, int(128/2), num_neighbors),
            LocalFeatureAggregation(128+d_noise, int(256/2), num_neighbors),
            LocalFeatureAggregation(256+d_noise, int(512/2), num_neighbors)
        ])

        self.mlp = SharedMLP(512, 512, activation_fn=nn.ReLU())

        # decoding layers
        decoder_kwargs = dict(
            transpose=True,
            bn=True,
            activation_fn=nn.ReLU(),
            use_spade=use_spade
        )
        self.decoder = nn.ModuleList([
            SharedMLP(1024, 256, **decoder_kwargs),
            SharedMLP(512, 128, **decoder_kwargs),
            SharedMLP(256, 64, **decoder_kwargs),
            SharedMLP(128, 64, **decoder_kwargs)
        ])
        if use_spade:
            self.spade = nn.ModuleList([
                Spade(6+d_noise, 64, num_neighbors),
                Spade(6+d_noise, 64, num_neighbors),
                Spade(6+d_noise, 128, num_neighbors),
                Spade(6+d_noise, 256, num_neighbors)
            ])

        # final semantic prediction
        self.fc_end = nn.Sequential(
            SharedMLP(64, 64, bn=True, activation_fn=nn.ReLU()),
            SharedMLP(64, 32, bn=True, activation_fn=nn.ReLU()),
            SharedMLP(32, d_out, activation_fn=nn.Tanh())
        )

        self.local_noise = local_noise
        self.use_spade = use_spade

        return
    
    def cat_noise(self, tensor, noise):
        size = tensor.size()
        if len(size) == 3:
            B, N, _ = size
            res = torch.cat([tensor, noise.view(B,1,self.d_noise).expand(B,N,self.d_noise)], dim=-1)
        else:
            B, _, N, _ = size
            res = torch.cat([tensor, noise.view(B,self.d_noise,1,1).expand(B,self.d_noise,N,1)], dim=1)
        return res
    
    def cat_local_noise(self, tensor, noise): # noise (B, N, d_noise)
        size = tensor.size()
        if len(size) == 3:
            res = torch.cat([tensor, noise], dim=-1)
        else:
            res = torch.cat([tensor, noise.transpose(1,2).unsqueeze(-1)], dim=1)
        return res
    
    def forward(self, im, coords, noise):
        r"""
            im shape: B, F, d_middle+3, H, W
            coords shape: B, N, 3 
            noise shape: B, N, d_noise
        """
        if len(im.size()) == 5:
            if self.grid_sampling:

                bb, ff, cc, hh, ww = im.size()
                im = im.permute([0,1,3,4,2]).reshape(bb,hh*ww*ff,cc)

                int_coords = (coords * self.ratio).int()
                unique_coords, inv_idx, counts = torch.unique(int_coords, dim=1, return_inverse=True, return_counts=True)
                im = torch.cat([im, noise], dim=-1)

                ft = torch.zeros(1, unique_coords.size(1), im.size(2)).to(im.device)
                ft.scatter_add_(
                    dim=1,
                    index=inv_idx.view(1, inv_idx.size(0), 1).expand(1, inv_idx.size(0), im.size(2)),
                    src=im
                )
                ft = ft / counts.view(1, counts.size(0), 1).expand(1, counts.size(0), ft.size(2))

                ft, noise = ft[:,:,:-noise.size(2)], ft[:,:,-noise.size(2):]
                if self.input_coord:
                    input_pc = torch.cat([ft, unique_coords.float()/self.ratio], dim=-1)
                else:
                    input_pc = torch.cat([ft, unique_coords.float() * 0], dim=-1)
                out = self.forward_base(input_pc, coords, noise)
                out = out[:, :, inv_idx]

                out = out.view(bb, 3, ff, hh, ww).permute([0,2,1,3,4])
            else:
                bb, ff, cc, hh, ww = im.size()
                if self.input_coord:
                    input_pc = torch.cat([im.permute([0,1,3,4,2]).reshape(bb,hh*ww*ff,cc), coords], dim=-1)
                else:
                    input_pc = torch.cat([im.permute([0,1,3,4,2]).reshape(bb,hh*ww*ff,cc), coords*0], dim=-1)
                out = self.forward_base(input_pc, coords, noise)
                out = out.view(bb, 3, ff, hh, ww).permute([0,2,1,3,4])
        
        else:
            if self.grid_sampling:
                int_coords = (coords * self.ratio).int()
                unique_coords, inv_idx, counts = torch.unique(int_coords, dim=1, return_inverse=True, return_counts=True)
                im = torch.cat([im, noise], dim=-1)

                ft = torch.zeros(1, unique_coords.size(1), im.size(2)).to(im.device)
                ft.scatter_add_(
                    dim=1,
                    index=inv_idx.view(1, inv_idx.size(0), 1).expand(1, inv_idx.size(0), im.size(2)),
                    src=im
                )
                ft = ft / counts.view(1, counts.size(0), 1).expand(1, counts.size(0), ft.size(2))

                ft, noise = ft[:,:,:-noise.size(2)], ft[:,:,-noise.size(2):]
                if self.input_coord:
                    input_pc = torch.cat([ft, unique_coords.float()/self.ratio], dim=-1)
                else:
                    input_pc = torch.cat([ft, unique_coords.float() * 0], dim=-1)
                out = self.forward_base(input_pc, coords, noise)
                out = out[:, :, inv_idx]

            else:
                bb, nn, cc = im.size()
                if self.input_coord:
                    input_pc = torch.cat([im, coords], dim=-1)
                else:
                    input_pc = torch.cat([im, coords * 0], dim=-1)
                out = self.forward_base(input_pc, coords, noise)

        # out = F.upsample(out.view(bb*ff, 3, hh, ww), scale_factor=2, mode='bicubic')
        # out = out.view(bb, ff, 3, hh*2, ww*2)

        return out


    def forward_base(self, input_pc, coords, noise):
        r"""
            Forward pass

            Parameters
            ----------
            input_pc: torch.Tensor, shape (B, N, d_in)
                input points

            Returns
            -------
            torch.Tensor, shape (B, d_out, N)
                segmentation scores for each point
        """

        r"""
            input_pc shape: B, F*H*W, d_middle+3
            coords shape: B, N, 3 
            noise shape: B, N, d_noise
        """

        B = input_pc.size(0)
        N = input_pc.size(1)
        d = self.decimation

        if self.local_noise:
            noise_pc = noise
        #     li = []
        #     for a, b in [(0,0),(0,2),(0,3),(0,1)]:
        #         li.append(noise[:, :, a, b])
        #     noise = torch.stack(li, dim=1)
        #     assert(len(noise.size()) == 3)
        #     pcx, pcy = coords[:,:,0:1], coords[:,:,1:2]
        #     noise_idx = torch.ge(pcx, 0).long() * 2 + torch.ge(pcy, 0).long()
        #     noise_idx = noise_idx.expand(B, N, self.d_noise).to(noise.device)
        #     noise_pc = torch.gather(noise, 1, noise_idx)

        if not self.local_noise:
            start = self.cat_noise(input_pc, noise)
        else:
            start = self.cat_local_noise(input_pc, noise_pc)
        
        if self.use_spade:
            # print(start.shape) # B, N, d
            # 3 (semantic) + 64 (middle) + 3 (coord) + 16 (noise)
            # input()
            input_spade = torch.cat([start[...,:3], start[...,self.d_middle+3:]], dim=-1)
            input_spade = input_spade.transpose(1, 2).unsqueeze(-1)


        x = self.fc_start(start).transpose(-2,-1).unsqueeze(-1)
        x = self.bn_start(x) # shape (B, d, N, 1)

        decimation_ratio = 1

        # <<<<<<<<<< ENCODER
        x_stack = []
        if self.use_spade:
            gamma_beta_stack = []

        permutation = torch.randperm(N)
        coords = coords[:,permutation]
        coords = coords.cpu().contiguous()
        x = x[:,:,permutation]
        if self.use_spade:
            input_spade = input_spade[:,:,permutation]

        for idx, lfa in enumerate(self.encoder):
            # at iteration i, x.shape = (B, N//(d**i), d_in)
            if not self.local_noise:
                x = lfa(coords[:,:N//decimation_ratio], self.cat_noise(x, noise), random_sel=self.random_sel)
            else:
                x = lfa(coords[:,:N//decimation_ratio].contiguous(), self.cat_local_noise(x, noise_pc[:,:N//decimation_ratio]), random_sel=self.random_sel)
            
            x_stack.append(x.clone())
            if self.use_spade:
                gamma_beta_stack = [
                    self.spade[idx](
                        coords[:,:N//decimation_ratio],
                        input_spade[:,:,:N//decimation_ratio]
                    )
                ] + gamma_beta_stack

            decimation_ratio *= d
            x = x[:,:,:N//decimation_ratio]

        # for ggg, bbb in gamma_beta_stack:
        #     print(ggg.shape, bbb.shape)
        # input()

        # # >>>>>>>>>> ENCODER

        x = self.mlp(x)

        # <<<<<<<<<< DECODER
        for idx, mlp in enumerate(self.decoder):
            # neighbors, _ = knn(
            #     coords[:,:N//decimation_ratio].cpu().contiguous(), # original set
            #     coords[:,:d*N//decimation_ratio].cpu().contiguous(), # upsampled set
            #     1
            # ) # shape (B, N, 1)
            neighbors, _ = knn_func(
                coords[:,:N//decimation_ratio],
                coords[:,:d*N//decimation_ratio],
                1, x.device
            )

            extended_neighbors = neighbors.unsqueeze(1).expand(-1, x.size(1), -1, 1)

            x_neighbors = torch.gather(x, -2, extended_neighbors)

            x = torch.cat((x_neighbors, x_stack.pop()), dim=1)

            if self.use_spade:
                gamma, beta = gamma_beta_stack[idx]
                x = mlp(x, gamma, beta)
            else:
                x = mlp(x)

            decimation_ratio //= d

        # >>>>>>>>>> DECODER
        # inverse permutation
        x = x[:,:,torch.argsort(permutation)]

        out = self.fc_end(x).squeeze(-1) # shape B, 3, N

        return {'output': out, 'feature': x.squeeze(-1)}


    def forward_randperm(self, im, coords, noise):
        r"""
            im shape: B, F, d_middle+3, H, W
            coords shape: B, N, 3 
            noise shape: B, N, d_noise
        """

        bb, ff, cc, hh, ww = im.size()
        input_pc = torch.cat([im.permute([0,1,3,4,2]).reshape(bb,hh*ww*ff,cc), coords], dim=-1)

        perm = torch.randperm(input_pc.size(1))
        inv_perm = torch.argsort(perm)

        perm_group = perm.reshape(15, -1)
        res = []
        for group in perm_group:
            res.append(self.forward_base(input_pc[:,group,:], coords[:,group,:], noise[:,group,:]))

        res = torch.cat(res, dim=-1)
        res = res[:,:,inv_perm]

        out = res.view(bb, 3, ff, hh, ww).permute([0,2,1,3,4])

        return out, None






if __name__ == '__main__':
    import time
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    d_in = 3
    im = torch.randn(1, 2, d_in, 256, 512).to(device)
    coord = 1000*torch.randn(1, 2**18, 3).to(device)
    noise = torch.randn(1, 16).to(device)
    model = RandLANet(d_in=d_in+3, d_out=3, d_noise=16, num_neighbors=8, decimation=4).to(device)
    # model.load_state_dict(torch.load('checkpoints/checkpoint_100.pth'))
    # model.eval()

    t0 = time.time()
    pred = model(im, coord, noise)
    t1 = time.time()
    # print(pred)
    print(t1-t0)
    input()
