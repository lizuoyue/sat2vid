# Copyright 2016-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torch.autograd import Function
from torch.nn import Module
from .utils import *
from .sparseConvNetTensor import SparseConvNetTensor


class JoinTable(torch.nn.Sequential):
    def __init__(self, *args):
        torch.nn.Sequential.__init__(self, *args)

    def forward(self, input):
        output = SparseConvNetTensor()
        output.metadata = input[0].metadata
        output.spatial_size = input[0].spatial_size
        output.features = torch.cat([i.features for i in input], 1) if input[0].features.numel() else input[0].features
        return output

    def input_spatial_size(self, out_size):
        return out_size


class AddTable(torch.nn.Sequential):
    def __init__(self, *args):
        torch.nn.Sequential.__init__(self, *args)

    def forward(self, input):
        output = SparseConvNetTensor()
        output.metadata = input[0].metadata
        output.spatial_size = input[0].spatial_size
        output.features = sum([i.features for i in input])
        return output

    def input_spatial_size(self, out_size):
        return out_size


class ConcatTable(torch.nn.Sequential):
    def __init__(self, *args):
        torch.nn.Sequential.__init__(self, *args)

    def forward(self, input):
        return [module(input) for module in self._modules.values()]

    def add(self, module):
        self._modules[str(len(self._modules))] = module
        return self

    def input_spatial_size(self, out_size):
        return self._modules['0'].input_spatial_size(out_size)



class NoiseTable(torch.nn.Sequential):
    def __init__(self, *args):
        torch.nn.Sequential.__init__(self, *args)
        self.noise = None
        self.std = 1e-3
        return

    def forward(self, input_tensor):
        assert(self.noise is not None)
        batch_idx = input_tensor.get_spatial_locations()[:, -1]

        self.noise_expand = self.noise[batch_idx]

        N, _ = input_tensor.features.size()

        self.noise_expand += torch.randn(N, self.noise.size(1)).to(input_tensor.features.device) * self.std

        output = SparseConvNetTensor()
        output.metadata = input_tensor.metadata
        output.spatial_size = input_tensor.spatial_size
        output.features = torch.cat([input_tensor.features, self.noise_expand], dim=1)

        return output

    def input_spatial_size(self, out_size):
        return out_size
    
    def set_noise(self, noise):
        self.noise = noise
        return


"""
class MultiNoiseTable(torch.nn.Sequential):
    def __init__(self, *args):
        torch.nn.Sequential.__init__(self, *args)
        self.noise = None
        self.std = 1e-3

    def forward(self, input_tensor):
        # return input_tensor
        # assert(self.noise is not None)
        N, _ = input_tensor.features.size()

        noise_idx = self.split_input_tensor(input_tensor)
        self.noise_expand = torch.randn(N, self.noise.size(1)).cuda() * self.std + self.noise[noise_idx]

        output = SparseConvNetTensor()
        output.metadata = input_tensor.metadata
        output.spatial_size = input_tensor.spatial_size
        output.features = torch.cat([input_tensor.features, self.noise_expand], dim=1)
        # assert(not torch.isnan(output.features).any())
        return output

    def input_spatial_size(self, out_size):
        return out_size
    
    def set_noise(self, noise):
        li = []
        # for a, b in [(1,0),(0,0),(1,3),(0,3),(1,1),(0,1),(1,2),(0,2)]:
        for a, b in [(0,0),(0,2),(0,3),(0,1)]:
            li.append(noise[:, :, a, b])
        self.noise = torch.cat(li)
        return
    
    def split_input_tensor(self, input_tensor):
        locs = input_tensor.get_spatial_locations()
        split = (input_tensor.spatial_size // 2).long().unsqueeze(0).expand(locs.size(0), 3)
        idx = torch.ge(locs[:,:3], split).long()
        # idx = idx[:,0]*4+idx[:,1]*2+idx[:,2]
        idx = idx[:,0]*2+idx[:,1]*1
        return idx



class MultiClsNoiseTable(torch.nn.Sequential):
    def __init__(self, *args):
        torch.nn.Sequential.__init__(self, *args)
        self.noise = None
        self.std = 1e-3
        return

    def forward(self, input_tensor):
        assert(self.noise is not None)
        N, _ = input_tensor.features.size()

        idx = self.get_input_tensor_order(input_tensor)
        noise_expand = torch.randn(N, self.noise.size(1)).cuda() * self.std + self.noise[idx]

        output = SparseConvNetTensor()
        output.metadata = input_tensor.metadata
        output.spatial_size = input_tensor.spatial_size
        output.features = torch.cat([input_tensor.features, noise_expand], dim=1)

        return output

    def input_spatial_size(self, out_size):
        return out_size
    
    def set_noise(self, noise):
        self.noise = noise
        return
    
    def get_input_tensor_order(self, input_tensor):
        locs = input_tensor.get_spatial_locations()
        sp = input_tensor.spatial_size
        idx = torch.argsort(locs[:,0] * sp[1] * sp[2] + locs[:,1] * sp[2] + locs[:,2])
        idx = torch.argsort(idx)
        return idx
"""
