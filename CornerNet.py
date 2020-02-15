import torch
import torch.nn as nn
from Hourglass_module import kp_module
from Corner_pooling import CornerPool_module
from utils import convolution, residual, make_layer, make_kp_layer, make_layer_revr, make_merge_layer, make_pool_layer, make_unpool_layer,\
                    make_cnv_layer, make_inter_layer, _tranpose_and_gather_feat

n = 5
dims    = [256, 256, 384, 384, 384, 512]
modules = [2, 2, 2, 2, 2, 4]
out_dim = 10

class kp(nn.Module):
    def __init__(
        self, n, nstack, dims, modules, out_dim, pre = None, cnv_dim = 256,        
        make_cnv_layer = make_cnv_layer, make_heat_layer = make_kp_layer,
        make_tag_layer = make_kp_layer, make_regr_layer = make_kp_layer,
        make_up_layer = make_layer, make_low_layer = make_layer, 
        make_hg_layer = make_layer, make_hg_layer_revr = make_layer_revr,
        make_pool_layer = make_pool_layer, make_unpool_layer = make_unpool_layer,
        make_merge_layer = make_merge_layer, make_inter_layer = make_inter_layer, 
        basic_layer = residual
    ):
        super(kp, self).__init__()

        self.nstack = nstack
        curr_dim = dims[0]

        self.pre = nn.Sequential(
                    convolution(7, 3, 128, stride=2),
                    residual(3, 128, 256, stride=2)
                )

        self.kps = nn.ModuleList([
                    kp_module(
                        n, dims, modules, layer = basic_layer,
                        make_up_layer = make_up_layer,
                        make_low_layer = make_low_layer,
                        make_hg_layer = make_hg_layer,
                        make_hg_layer_revr = make_hg_layer_revr,
                        make_pool_layer = make_pool_layer,
                        make_unpool_layer = make_unpool_layer,
                        make_merge_layer = make_merge_layer
                    ) for _ in range(nstack)
                ])

        self.cnvs = nn.ModuleList([
            make_cnv_layer(curr_dim, cnv_dim) for _ in range(nstack)
        ])

        '''
        self.tl_cnvs = nn.ModuleList([
            make_tl_layer(cnv_dim) for _ in range(nstack)
        ])
        self.br_cnvs = nn.ModuleList([
            make_br_layer(cnv_dim) for _ in range(nstack)
        ])
        '''

        self.corner_pools = nn.ModuleList([
            CornerPool_module() for _ in range(nstack)
        ])

        ## keypoint heatmaps
        self.tl_heats = nn.ModuleList([
            make_heat_layer(cnv_dim, curr_dim, out_dim) for _ in range(nstack)
        ])
        self.br_heats = nn.ModuleList([
            make_heat_layer(cnv_dim, curr_dim, out_dim) for _ in range(nstack)
        ])

        ## tags
        self.tl_tags  = nn.ModuleList([
            make_tag_layer(cnv_dim, curr_dim, 1) for _ in range(nstack)
        ])
        self.br_tags  = nn.ModuleList([
            make_tag_layer(cnv_dim, curr_dim, 1) for _ in range(nstack)
        ])

        for tl_heat, br_heat in zip(self.tl_heats, self.br_heats):
            tl_heat[-1].bias.data.fill_(-2.19)
            br_heat[-1].bias.data.fill_(-2.19)

        self.inters = nn.ModuleList([
            make_inter_layer(curr_dim) for _ in range(nstack - 1)
        ])

        self.inters_ = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(curr_dim, curr_dim, (1, 1), bias=False),
                nn.BatchNorm2d(curr_dim)
            ) for _ in range(nstack - 1)
        ])
        self.cnvs_ = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(cnv_dim, curr_dim, (1, 1), bias=False),
                nn.BatchNorm2d(curr_dim)
            ) for _ in range(nstack - 1)
        ])

        self.tl_regrs = nn.ModuleList([
            make_regr_layer(cnv_dim, curr_dim, 2) for _ in range(nstack)
        ])
        self.br_regrs = nn.ModuleList([
            make_regr_layer(cnv_dim, curr_dim, 2) for _ in range(nstack)
        ])

        self.relu = nn.ReLU(inplace=True)

    def forward(self, *xs):
        image   = xs[0]
        tl_inds = xs[1]
        br_inds = xs[2]

        inter = self.pre(image)
        #print('inter_done:{} '.format(inter.shape))
        outs  = []

        layers = zip(
            self.kps, self.cnvs,
            self.corner_pools,
            self.tl_heats, self.br_heats,
            self.tl_tags, self.br_tags,
            self.tl_regrs, self.br_regrs
        )
        for ind, layer in enumerate(layers):
            kp_, cnv_                       = layer[0:2]
            corner_pools_                   = layer[2]
            tl_heat_, br_heat_              = layer[3:5]
            tl_tag_, br_tag_                = layer[5:7]
            tl_regr_, br_regr_              = layer[7:9]

            kp = kp_(inter)
            #print('Hourglass_module_done:{} '.format(Hourglass_module.shape))
            cnv = cnv_(kp)
            #print('cnv_done:{} '.format(cnv.shape))
            tl_cnv, br_cnv = corner_pools_(cnv)
            #print('corner_pools_done:{}, {} '.format(tl_cnv.shape, br_cnv.shape))
            
            tl_heat, br_heat = tl_heat_(tl_cnv), br_heat_(br_cnv)
            #print('heat_done:{}, {} '.format(tl_heat.shape, br_heat.shape))
            tl_tag,  br_tag  = tl_tag_(tl_cnv),  br_tag_(br_cnv)
            #print('tag_done:{}, {} '.format(tl_tag.shape, br_tag.shape))
            tl_regr, br_regr = tl_regr_(tl_cnv), br_regr_(br_cnv)
            #print('regr_done:{}, {} '.format(tl_regr.shape, br_regr.shape))

            tl_tag  = _tranpose_and_gather_feat(tl_tag, tl_inds)
            #print('tl_tag_transpose_done:{} '.format(tl_tag.shape))
            br_tag  = _tranpose_and_gather_feat(br_tag, br_inds)
            #print('br_tag_transpose_done:{} '.format(br_tag.shape))
            tl_regr = _tranpose_and_gather_feat(tl_regr, tl_inds)
            #print('tl_regr_transpose_done:{} '.format(tl_regr.shape))
            br_regr = _tranpose_and_gather_feat(br_regr, br_inds)
            #print('br_regr_transpose_done:{} '.format(br_regr.shape))

            outs += [tl_heat, br_heat, tl_tag, br_tag, tl_regr, br_regr]

            if ind < self.nstack - 1:
                inter = self.inters_[ind](inter) + self.cnvs_[ind](cnv)
                inter = self.relu(inter)
                inter = self.inters[ind](inter)
                #print('inter_change_done')
        return outs

#Hourglass = Hourglass_module(n = n, dims = dims, modules = modules)