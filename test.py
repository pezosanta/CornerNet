import glob
import cv2
import numpy as np
from skimage import io
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from Hourglass_module_orig import kp_module
from Corner_pooling_v2 import CornerPool_module
from utils import convolution, residual, make_layer, make_kp_layer, make_layer_revr, make_merge_layer, make_pool_layer, make_unpool_layer,\
                    make_cnv_layer, make_inter_layer, _tranpose_and_gather_feat, _nms, _topk, _decode

n = 5
dims    = [256, 256, 384, 384, 384, 512]
modules = [2, 2, 2, 2, 2, 4]
out_dim = 10

# kwargs: 
top_k = 100
ae_threshold = 0.5
nms_threshold = 0.5
nms_kernel = 3

class Dataset_test(Dataset):
    def __init__(self):
        self.device = torch.device('cuda:0')
        self.test_image_paths = glob.glob('/content/test/*.jpg')
        self.resize_size = (640, 360)
        self.npad = ((0, 24), (0, 0), (0, 0))

    def __getitem__(self, index):
        image = io.imread(self.test_image_paths[index])
        image = cv2.resize(image, self.resize_size) 
        image = np.pad(image, pad_width = self.npad, mode = 'constant', constant_values = 0)
        
        tensor_image = torch.from_numpy(image / 255.0).float().to(device = self.device).permute(2, 0, 1)

        return tensor_image
    
    def __len__(self): 
      return len(self.test_image_paths)

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

    def forward(self, *xs, ae_threshold, K, kernel):       
        image = xs[0]

        inter = self.pre(image)
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

            kp  = kp_(inter)
            cnv = cnv_(kp)

            if ind == self.nstack - 1:
                tl_cnv, br_cnv = corner_pools_(cnv)

                tl_heat, br_heat = tl_heat_(tl_cnv), br_heat_(br_cnv)
                tl_tag,  br_tag  = tl_tag_(tl_cnv),  br_tag_(br_cnv)
                tl_regr, br_regr = tl_regr_(tl_cnv), br_regr_(br_cnv)

                outs += [tl_heat, br_heat, tl_tag, br_tag, tl_regr, br_regr]

            if ind < self.nstack - 1:
                inter = self.inters_[ind](inter) + self.cnvs_[ind](cnv)
                inter = self.relu(inter)
                inter = self.inters[ind](inter)

        return _decode(*outs[-6:], ae_threshold = ae_threshold, K = K, kernel = kernel)

def test(batch_size = 1):
    
    test_dataset = Dataset_test()
    test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle = True)
    test_loader_iter = iter(test_loader)

    CHECKPOINT_PATH = '/content/drive/My Drive/CornerNet/ModelParams/train_valid_pretrained_cornernet-epoch{}-iter{}.pth'.format(3, 5067)
    checkpoint = torch.load(CHECKPOINT_PATH)
    best_average_val_loss = checkpoint['val_loss']

    model = kp(n = n, nstack = 2, dims = dims, modules = modules, out_dim = out_dim).cuda()

    model.load_state_dict(checkpoint['model_state_dict'])

    model.eval()

    images = next(test_loader_iter)
    print(images.shape)

    detections = model(images, ae_threshold = ae_threshold, K = top_k, kernel = nms_kernel)

    print('Detections_size: {}\nDetections: {}'.format(detections.shape, detections))



if __name__ == "__main__":
    test()