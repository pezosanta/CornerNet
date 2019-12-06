import torch
import torch.nn as nn
from utils import make_layer, make_layer_revr, make_pool_layer, make_unpool_layer, make_merge_layer, residual

class Hourglass_module(nn.Module):
    def __init__(
        self, n, dims, modules, layer = residual,
        make_up_layer = make_layer, make_low_layer = make_layer,
        make_hg_layer = make_layer, make_hg_layer_revr = make_layer_revr,
        make_pool_layer = make_pool_layer, make_unpool_layer = make_unpool_layer,
        make_merge_layer = make_merge_layer, **kwargs
    ):
        super(Hourglass_module, self).__init__()

        self.n = n

        curr_mod = modules[0]
        next_mod = modules[1]

        curr_dim = dims[0]
        next_dim = dims[1]

        self.up1  = make_up_layer(
            3, curr_dim, curr_dim, curr_mod, 
            layer=layer, **kwargs
        )  
        self.max1 = make_pool_layer(curr_dim)
        self.low1 = make_hg_layer(
            3, curr_dim, next_dim, curr_mod,
            layer = layer, **kwargs
        )
        self.low2 = Hourglass_module(
            n - 1, dims[1:], modules[1:], layer=layer, 
            make_up_layer = make_up_layer, 
            make_low_layer = make_low_layer,
            make_hg_layer = make_hg_layer,
            make_hg_layer_revr = make_hg_layer_revr,
            make_pool_layer = make_pool_layer,
            make_unpool_layer = make_unpool_layer,
            make_merge_layer = make_merge_layer,
            **kwargs
        ) if self.n > 1 else \
        make_low_layer(
            3, next_dim, next_dim, next_mod,
            layer = layer, **kwargs
        )
        self.low3 = make_hg_layer_revr(
            3, next_dim, curr_dim, curr_mod,
            layer = layer, **kwargs
        )
        self.up2 = make_unpool_layer(curr_dim)

        self.merge = make_merge_layer(curr_dim)

    def forward(self, x):
        up1  = self.up1(x)
        max1 = self.max1(x)
        low1 = self.low1(max1)
        low2 = self.low2(low1)
        low3 = self.low3(low2)
        up2  = self.up2(low3)
        merge = self.merge(up1, up2)
        
        return merge