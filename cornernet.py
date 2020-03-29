import time
import torch
import torch.nn as nn
from mobilenetv3 import mobilenetv3_large, mobilenetv3_small
from corner_pooling import CornerPool_module
from utils import convolution, residual, make_kp_layer,make_inter_layer, _tranpose_and_gather_feat, _decode

class CornerNet(nn.Module):
    def __init__(
        self, nstack = 2, out_dim = 10,
        mobilenet_out_dim = 160, mobilenet_1_input_dim = 16, mobilenet_2_input_dim = 40, cnv_dim = 256,        
        make_heat_layer = make_kp_layer, make_tag_layer = make_kp_layer, make_regr_layer = make_kp_layer,
        make_inter_layer = make_inter_layer, residual = residual
    ):
        super(CornerNet, self).__init__()

        self.nstack = nstack

        # Backbone architecture layers
        self.pre = nn.Sequential(
                    convolution(7, 3, 8, stride = 2),
                    residual(3, 8, mobilenet_1_input_dim, stride = 2))
        
        self.mobilenetv3s = nn.ModuleList([
                    mobilenetv3_large() for _ in range(nstack)
        ])

        # Loading pretrained parameters only into the first Mobilenetv3        
        self.loading_params(self.mobilenetv3s[0])      
        
        self.mobilenetv3s_out_cnvs = nn.ModuleList([
            residual(3, mobilenet_out_dim, mobilenet_2_input_dim) for _ in range(nstack)
        ])

        self.mobilenetv3s_out_upsamples = nn.ModuleList([nn.UpsamplingBilinear2d(size = (23, 40)) for _ in range(nstack)]) 

        self.predmods_inp_res = nn.ModuleList([
            residual(3, mobilenet_2_input_dim, cnv_dim) for _ in range(nstack)
        ])

        self.predmods_inp_upsamples = nn.ModuleList([nn.UpsamplingBilinear2d(size = (90, 160)) for _ in range(nstack)]) 

        # Layers for connecting stages
        self.pre_cnv_ = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(mobilenet_1_input_dim, mobilenet_2_input_dim, (1, 1), stride = (4, 4), bias = False),
                nn.BatchNorm2d(mobilenet_2_input_dim)
            ) for _ in range(nstack - 1)
        ])

        self.inters_cnvs_ = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(mobilenet_2_input_dim, mobilenet_2_input_dim, (1, 1), bias = False),
                nn.BatchNorm2d(mobilenet_2_input_dim)
            ) for _ in range(nstack - 1)
        ]) 

        self.inters_res_ = nn.ModuleList([
            make_inter_layer(mobilenet_2_input_dim) for _ in range(nstack - 1)
        ])

        # Prediction module layers
        self.corner_pools = nn.ModuleList([
            CornerPool_module() for _ in range(nstack)
        ])

        self.tl_heats = nn.ModuleList([
            make_heat_layer(cnv_dim, cnv_dim, out_dim) for _ in range(nstack)
        ])
        self.br_heats = nn.ModuleList([
            make_heat_layer(cnv_dim, cnv_dim, out_dim) for _ in range(nstack)
        ])
        for tl_heat, br_heat in zip(self.tl_heats, self.br_heats):
            tl_heat[-1].bias.data.fill_(-2.19)
            br_heat[-1].bias.data.fill_(-2.19)

        self.tl_tags  = nn.ModuleList([
            make_tag_layer(cnv_dim, cnv_dim, 1) for _ in range(nstack)
        ])
        self.br_tags  = nn.ModuleList([
            make_tag_layer(cnv_dim, cnv_dim, 1) for _ in range(nstack)
        ])

        self.tl_regrs = nn.ModuleList([
            make_regr_layer(cnv_dim, cnv_dim, 2) for _ in range(nstack)
        ])
        self.br_regrs = nn.ModuleList([
            make_regr_layer(cnv_dim, cnv_dim, 2) for _ in range(nstack)
        ])

        self.relu = nn.ReLU(inplace = True)

    def loading_params(self, model):        
        state_dict = torch.load('pretrained/mobilenetv3-large-657e7b3d.pth')

        state_dict["classifier.0.weight"] = state_dict["classifier.1.weight"]
        del state_dict["classifier.1.weight"]

        state_dict["classifier.0.bias"] = state_dict["classifier.1.bias"]
        del state_dict["classifier.1.bias"]

        state_dict["classifier.3.weight"] = state_dict["classifier.5.weight"]
        del state_dict["classifier.5.weight"]

        state_dict["classifier.3.bias"] = state_dict["classifier.5.bias"]
        del state_dict["classifier.5.bias"] 
    
        model.load_state_dict(state_dict, strict = True)

    def forward(self, *xs, mode, ae_threshold = 0.5, top_k = 100, kernel = 3):
        if mode == 'Train' or mode == 'Val':
            image   = xs[0]
            tl_inds = xs[1]
            br_inds = xs[2]

            inter = self.pre(image)

            outs  = []

            layers = zip(
                self.mobilenetv3s,
                self.mobilenetv3s_out_cnvs,
                self.mobilenetv3s_out_upsamples,
                self.predmods_inp_res,
                self.predmods_inp_upsamples,            
                self.corner_pools,
                self.tl_heats, self.br_heats,
                self.tl_tags, self.br_tags,
                self.tl_regrs, self.br_regrs
            )
            for ind, layer in enumerate(layers):
                mobilenetv3_                    = layer[0]
                mobilenetv3_out_cnv_            = layer[1]
                mobilenetv3_out_upsample_       = layer[2]
                predmod_inp_res_                = layer[3]
                predmod_inp_upsample_           = layer[4]
                corner_pool_                    = layer[5]
                tl_heat_, br_heat_              = layer[6:8]
                tl_tag_, br_tag_                = layer[8:10]
                tl_regr_, br_regr_              = layer[10:12]
                
                mobilenetv3 = mobilenetv3_(inter, stage = ind)
                
                mobilenetv3_out_cnv = mobilenetv3_out_cnv_(mobilenetv3)
                
                mobilenetv3_out_upsample = mobilenetv3_out_upsample_(mobilenetv3_out_cnv)
                
                predmod_inp_res = predmod_inp_res_(mobilenetv3_out_upsample)
                
                predmod_inp_upsample = predmod_inp_upsample_(predmod_inp_res)
                
                
                tl_cnv, br_cnv = corner_pool_(predmod_inp_upsample)
                
                tl_heat, br_heat = tl_heat_(tl_cnv), br_heat_(br_cnv) 
                tl_tag,  br_tag  = tl_tag_(tl_cnv),  br_tag_(br_cnv) 
               
                tl_regr, br_regr = tl_regr_(tl_cnv), br_regr_(br_cnv)

                tl_tag  = _tranpose_and_gather_feat(tl_tag, tl_inds)
                br_tag  = _tranpose_and_gather_feat(br_tag, br_inds)
                tl_regr = _tranpose_and_gather_feat(tl_regr, tl_inds)
                br_regr = _tranpose_and_gather_feat(br_regr, br_inds)


                outs += [tl_heat, br_heat, tl_tag, br_tag, tl_regr, br_regr]

                if ind < (self.nstack - 1):
                    inter = self.pre_cnv_[0](inter) + self.inters_cnvs_[0](mobilenetv3_out_upsample)
                    inter = self.relu(inter)
                    inter = self.inters_res_[ind](inter)
    
            return outs
        
        elif mode == 'mAP':
            image = xs[0]
            
            inter = self.pre(image)
            
            outs  = []

            layers = zip(
                self.mobilenetv3s,
                self.mobilenetv3s_out_cnvs,
                self.mobilenetv3s_out_upsamples,
                self.predmods_inp_res,
                self.predmods_inp_upsamples,            
                self.corner_pools,
                self.tl_heats, self.br_heats,
                self.tl_tags, self.br_tags,
                self.tl_regrs, self.br_regrs
            )
            for ind, layer in enumerate(layers):
                mobilenetv3_                    = layer[0]
                mobilenetv3_out_cnv_            = layer[1]
                mobilenetv3_out_upsample_       = layer[2]
                predmod_inp_res_                = layer[3]
                predmod_inp_upsample_           = layer[4]
                corner_pool_                    = layer[5]
                tl_heat_, br_heat_              = layer[6:8]
                tl_tag_, br_tag_                = layer[8:10]
                tl_regr_, br_regr_              = layer[10:12]
                
                mobilenetv3 = mobilenetv3_(inter, stage = ind)
                
                mobilenetv3_out_cnv = mobilenetv3_out_cnv_(mobilenetv3)
                
                mobilenetv3_out_upsample = mobilenetv3_out_upsample_(mobilenetv3_out_cnv)
                
                predmod_inp_res = predmod_inp_res_(mobilenetv3_out_upsample)
                
                predmod_inp_upsample = predmod_inp_upsample_(predmod_inp_res)
                
                if ind == (self.nstack - 1):
                    tl_cnv, br_cnv = corner_pool_(predmod_inp_upsample)
                    
                    tl_heat, br_heat = tl_heat_(tl_cnv), br_heat_(br_cnv)
                    tl_tag,  br_tag  = tl_tag_(tl_cnv),  br_tag_(br_cnv)
                    tl_regr, br_regr = tl_regr_(tl_cnv), br_regr_(br_cnv)
                
                    outs += [tl_heat, br_heat, tl_tag, br_tag, tl_regr, br_regr]

                if ind < (self.nstack - 1):
                    inter = self.pre_cnv_[0](inter) + self.inters_cnvs_[0](mobilenetv3_out_upsample)
                    inter = self.relu(inter)
                    inter = self.inters_res_[ind](inter)

            return _decode(*outs[-6:], ae_threshold = ae_threshold, K = top_k, kernel = kernel)

#Hourglass = Hourglass_module(n = n, dims = dims, modules = modules)

if __name__ == "__main__":
    import numpy as np
    # Model Hyperparameters
    nstack = 2
    out_dim = 10

    # Hyperparams for creating bounding boxes
    top_k = 100
    ae_threshold = 0.5
    nms_threshold = 0.5
    nms_kernel = 3
    min_score = 0.3

    model = CornerNet(nstack = nstack, out_dim = out_dim).cuda()
    model.eval()
    for i in range(100):
        print(i)
        images      = torch.zeros(20, 3, 360, 640).cuda()
        tl_tags     = np.zeros((20, 160), dtype = np.int64)
        br_tags     = np.zeros((20, 160), dtype = np.int64)
        tl_tags     = torch.from_numpy(tl_tags).cuda()
        br_tags     = torch.from_numpy(br_tags).cuda()

        xs = [images, tl_tags, br_tags]

        detections = model(*xs, mode = 'Train', ae_threshold = ae_threshold, top_k = top_k, kernel = nms_kernel)
    
    '''
    print('TL_HEAT: {}'.format(detections[0].shape))
    print('BR_HEAT: {}'.format(detections[1].shape))
    print('TL_TAG: {}'.format(detections[2].shape))
    print('BR_TAG: {}'.format(detections[3].shape))
    print('TL_REGR: {}'.format(detections[4].shape))
    print('BR_REGR: {}'.format(detections[5].shape))
    '''
    
    
    '''
    model.eval()
    with torch.no_grad():
        for idx in range(300):
            images = torch.zeros(1, 3, 360, 640).cuda()
            start_time = time.time()
            detections = model(images, mode = 'mAP', ae_threshold = ae_threshold, top_k = top_k, kernel = nms_kernel)
            torch.cuda.synchronize()
            time_taken = time.time() - start_time
            print('Run-Time: {:.4f} s'.format(time_taken))
    '''

    
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    NUM = count_parameters(model)
    print(NUM)
    

    '''
    from torch.utils.tensorboard import SummaryWriter

    # Model Hyperparameters
    n = 5
    nstack = 2
    dims    = [256, 256, 384, 384, 384, 512]
    modules = [2, 2, 2, 2, 2, 4]
    out_dim = 10

    writer = SummaryWriter('../CornerNet/Hourglass/logs/mobilenetv3/') 

    model = CornerNet(n = n, nstack = nstack, dims = dims, modules = modules, out_dim = out_dim).cuda()

    images = torch.zeros(1, 3, 720, 1280).cuda()

    writer.add_graph(model, input_to_model=images, verbose=False)
    writer.close()
    '''
    '''
    print(torch.cuda.device_count())
    print(torch.cuda.current_device())
    x = torch.zeros(1, 3, 360, 640).to('cuda:1')
    print(torch.cuda.current_device())
    print('DONE')
    '''
   