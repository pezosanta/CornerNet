import glob
import cv2
import numpy as np
from skimage import io
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
#from Dataset import Dataset as Dataset_test
from Hourglass_module import kp_module
from Corner_pooling import CornerPool_module
from utils import convolution, residual, make_layer, make_kp_layer, make_layer_revr, make_merge_layer, make_pool_layer, make_unpool_layer,\
                    make_cnv_layer, make_inter_layer, _tranpose_and_gather_feat, _nms, _topk, _decode
from external.nms import soft_nms, soft_nms_merge


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
        name = self.test_image_paths[index].rsplit('/', 1)[1]
        print(name)

        image = io.imread(self.test_image_paths[index])
        image = cv2.resize(image, self.resize_size) 
        image = np.pad(image, pad_width = self.npad, mode = 'constant', constant_values = 0)
        
        tensor_image = torch.from_numpy(image / 255.0).float().to(device = self.device).permute(2, 0, 1)

        return tensor_image, name
    
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
    
   
    CHECKPOINT_PATH = '/content/drive/My Drive/CornerNet/ModelParams/NEW_train_valid_pretrained_cornernet-epoch{}.pth'.format(9)
    checkpoint = torch.load(CHECKPOINT_PATH)
    best_average_val_loss = checkpoint['val_loss']
    
    '''
    val_dataset = Dataset_haha(mode = 'Val')
    val_loader = DataLoader(val_dataset, batch_size = batch_size, shuffle = True)
    val_loader_iter = iter(val_loader)
    images, tl_tags, br_tags, tl_heatmaps, br_heatmaps, tag_masks, tl_regrs, br_regrs, names = next(val_loader_iter)
    print('TL_TAGS_SHAPE: {}\nTL_HEATMAPS_SHAPE: {}\nTL_REGRS_SHAPE: {}'.format(tl_tags.shape, tl_heatmaps.shape, tl_regrs.shape))
    outs = [tl_heatmaps, br_heatmaps, tl_tags, br_tags, tl_regrs, br_regrs]
    detections = _decode(*outs[-6:], ae_threshold = ae_threshold, K = 160, kernel = nms_kernel)
    '''

    
    model = kp(n = n, nstack = 2, dims = dims, modules = modules, out_dim = out_dim).cuda()

    model.load_state_dict(checkpoint['model_state_dict'])

    model.eval()

    images, names = next(test_loader_iter)
    print(images.shape)

    detections = model(images, ae_threshold = ae_threshold, K = top_k, kernel = nms_kernel)
    
    

    #print('Detections_size: {}\nDetections: {}'.format(detections.shape, detections))
    print('Detections_size: {}'.format(detections.shape))

    '''
    z = (torch.randn(2, 3)*1000).int()

    z_view_2 = z.view(2, 3, 1)
    z_view_2_expand = z_view_2.expand(2, 3, 3)

    z_view_1 = z.view(2, 1, 3)
    z_view_1_expand = z_view_1.expand(2, 3, 3)

    print('Z: {}\nZ_VIEW_2: {}\nZ_VIEW_2_EXPAND: {}\nZ_VIEW_1: {}\nZ_VIEW_1_EXPAND: {}'.format(z, z_view_2, z_view_2_expand, z_view_1, z_view_1_expand))


    #tl_xs = tl_xs.view(batch, K, 1).expand(batch, K, K)
    #br_ys = br_ys.view(batch, 1, K).expand(batch, K, K)
    '''
    detections = detections.data.cpu().numpy()
    #print('NUMPY DETECTIONS_SHAPE: {}\nNUMPY_DETECTIONS: {}'.format(detections.shape, detections))

    out_width = 160
    dets = detections.reshape(2, -1, 8)
    #print('DETS_FIRST_RESHAPE_SHAPE: {}\nDETS_FIRTS_RESHAPE: {}'.format(dets.shape, dets))
    #print('DETS_BEFORE_SUB_FROM_WIDTH_SHAPE: {}\nDETS_BEFORE_SUB_FROM_WIDTH: {}'.format(dets[1, :, [0, 2]].shape, dets[1, :, [0, 2]]))
    dets[1, :, [0, 2]] = out_width - dets[1, :, [2, 0]]
    #print('DETS_AFTER_SUB_FROM_WIDTH_SHAPE: {}\nDETS_AFTER_SUB_FROM_WIDTH: {}'.format(dets[1, :, [0, 2]].shape, dets[1, :, [0, 2]]))
    detections = dets.reshape(1, -1, 8)
    #print('DETS_SECOND_RESHAPE_SHAPE: {}\nDETS_SECOND_RESHAPE: {}'.format(detections.shape, detections))

    classes    = detections[..., -1]
    #print(classes.shape, classes)
    classes    = classes[0]
    #print(classes.shape, classes)
    detections = detections[0]
    #print(detections.shape, detections)

    # reject detections with negative scores
    keep_inds  = (detections[:, 4] > -1)
    #print(keep_inds.shape, keep_inds)
    detections = detections[keep_inds]
    #print(detections.shape, detections)
    classes    = classes[keep_inds]
    #print(classes.shape, classes)

    categories = 10
    max_per_image = 100

    merge_bbox = False
    nms_algorithm   = 2 #"exp_soft_nms"
    weight_exp      = 8


    top_bboxes = {}
    for j in range(1, categories):
            keep_inds = (classes == j)
            #print('KEEP_INDS: {}'.format(keep_inds))
            
            top_bboxes[j] = detections[keep_inds][:, 0:7].astype(np.float32)
            if merge_bbox:
                soft_nms_merge(top_bboxes[j], Nt=nms_threshold, method=nms_algorithm, weight_exp=weight_exp)
            else:
                soft_nms(top_bboxes[j], Nt=nms_threshold, method=nms_algorithm)
            top_bboxes[j] = top_bboxes[j][:, 0:5]
   
    scores = np.hstack([
        top_bboxes[j][:, -1] 
        for j in range(1, categories)
    ])
    if len(scores) > max_per_image:
        kth    = len(scores) - max_per_image
        thresh = np.partition(scores, kth)[kth]
        #print("THRESHOLD: {}".format(thresh))
        for j in range(1, categories):
            keep_inds = (top_bboxes[j][:, -1] >= thresh)
            top_bboxes[j] = top_bboxes[j][keep_inds]

    print('TOP_BBOXES_LEN: {}\nTOP_BBOXES: {}'.format(len(top_bboxes), top_bboxes))
   # print('BUS_BBOXES_LEN: {}'.format(len(top_bboxes[0])))
    print('TRAFFIC_LIGHT_BBOXES_LEN: {}'.format(len(top_bboxes[1])))
    print('TRAFFIC_SIGN_BBOXES_LEN: {}'.format(len(top_bboxes[2])))
    print('PERSON_BBOXES_LEN: {}'.format(len(top_bboxes[3])))
    print('BICYCLE_BBOXES_LEN: {}'.format(len(top_bboxes[4])))
    print('TRUCK_BBOXES_LEN: {}'.format(len(top_bboxes[5])))
    print('MOTORCYCLE_BBOXES_LEN: {}'.format(len(top_bboxes[6])))
    print('CAR_BBOXES_LEN: {}'.format(len(top_bboxes[7])))
    print('TRAIN_BBOXES_LEN: {}'.format(len(top_bboxes[8])))
    print('RIDER_BBOXES_LEN: {}'.format(len(top_bboxes[9])))


    for j in range(1, categories):
        keep_inds = (top_bboxes[j][:, -1] > 0.4)    # > X -et átirni, ha kisebb biztonsággal prediktált bboxokat is akarunk!
        top_bboxes[j] = top_bboxes[j][keep_inds]

    print('TOP_BBOXES_LEN: {}\nTOP_BBOXES: {}'.format(len(top_bboxes), top_bboxes))
   # print('BUS_BBOXES_LEN: {}'.format(len(top_bboxes[0])))
    print('TRAFFIC_LIGHT_BBOXES_LEN: {}'.format(len(top_bboxes[1])))
    print('TRAFFIC_SIGN_BBOXES_LEN: {}'.format(len(top_bboxes[2])))
    print('PERSON_BBOXES_LEN: {}'.format(len(top_bboxes[3])))
    print('BICYCLE_BBOXES_LEN: {}'.format(len(top_bboxes[4])))
    print('TRUCK_BBOXES_LEN: {}'.format(len(top_bboxes[5])))
    print('MOTORCYCLE_BBOXES_LEN: {}'.format(len(top_bboxes[6])))
    print('CAR_BBOXES_LEN: {}'.format(len(top_bboxes[7])))
    print('TRAIN_BBOXES_LEN: {}'.format(len(top_bboxes[8])))
    print('RIDER_BBOXES_LEN: {}'.format(len(top_bboxes[9])))

    debug = True
    current_image_paths = '/content/test/' + names[0]
    print(current_image_paths)

    categories_dict = { 0: 'bus',
                    1: 'traffic light',
                    2: 'traffic sign',
                    3: 'person',
                    4: 'bicycle',
                    5: 'truck',
                    6: 'motorcycle',
                    7: 'car',
                    8: 'train',
                    9: 'rider' }
    
    if debug:
        #image_file = db.image_file(db_ind)
        image      = cv2.imread(current_image_paths)
        image_name = current_image_paths.rsplit('/', 1)[1]
        print(image)
        print(image_name)

        bboxes = {}
        for j in range(1, categories):
            #keep_inds = (top_bboxes[j][:, -1])
            cat_name  = categories_dict[j]
            cat_size  = cv2.getTextSize(cat_name, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            color     = np.random.random((3, )) * 0.6 + 0.4
            color     = color * 255
            color     = color.astype(np.int32).tolist()
            for bbox in top_bboxes[j]:
                bbox  = ((bbox[0:4])*8).astype(np.int32) #*8 nem is kell?
                if bbox[1] - cat_size[1] - 2 < 0:
                    cv2.rectangle(image,
                        (bbox[0], bbox[1] + 2),
                        (bbox[0] + cat_size[0], bbox[1] + cat_size[1] + 2),
                        color, -1
                    )
                    cv2.putText(image, cat_name, 
                        (bbox[0], bbox[1] + cat_size[1] + 2), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), thickness=1
                    )
                else:
                    cv2.rectangle(image,
                        (bbox[0], bbox[1] - cat_size[1] - 2),
                        (bbox[0] + cat_size[0], bbox[1] - 2),
                        color, -1
                    )
                    cv2.putText(image, cat_name, 
                        (bbox[0], bbox[1] - 2), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), thickness=1
                    )
                cv2.rectangle(image,
                    (bbox[0], bbox[1]),
                    (bbox[2], bbox[3]),
                    color, 2
                )
        cv2.imwrite(('/content/' + image_name), image)
    

if __name__ == "__main__":
    test()