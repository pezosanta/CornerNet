import torch
import torch.nn as nn


class convolution(nn.Module):
    def __init__(self, k, inp_dim, out_dim, stride=1, with_bn=True):
        super(convolution, self).__init__()

        pad = (k - 1) // 2
        self.conv = nn.Conv2d(inp_dim, out_dim, (k, k), padding=(pad, pad), stride=(stride, stride), bias=not with_bn)
        self.bn   = nn.BatchNorm2d(out_dim) if with_bn else nn.Sequential()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        conv = self.conv(x)
        bn   = self.bn(conv)
        relu = self.relu(bn)
        return relu

class fully_connected(nn.Module):
    def __init__(self, inp_dim, out_dim, with_bn=True):
        super(fully_connected, self).__init__()
        self.with_bn = with_bn

        self.linear = nn.Linear(inp_dim, out_dim)
        if self.with_bn:
            self.bn = nn.BatchNorm1d(out_dim)
        self.relu   = nn.ReLU(inplace=True)

    def forward(self, x):
        linear = self.linear(x)
        bn     = self.bn(linear) if self.with_bn else linear
        relu   = self.relu(bn)
        return relu

class residual(nn.Module):
    def __init__(self, k, inp_dim, out_dim, stride=1, with_bn=True):
        super(residual, self).__init__()

        self.conv1 = nn.Conv2d(inp_dim, out_dim, (3, 3), padding=(1, 1), stride=(stride, stride), bias=False)
        self.bn1   = nn.BatchNorm2d(out_dim)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_dim, out_dim, (3, 3), padding=(1, 1), bias=False)
        self.bn2   = nn.BatchNorm2d(out_dim)
        
        self.skip  = nn.Sequential(
            nn.Conv2d(inp_dim, out_dim, (1, 1), stride=(stride, stride), bias=False),
            nn.BatchNorm2d(out_dim)
        ) if stride != 1 or inp_dim != out_dim else nn.Sequential()
        self.relu  = nn.ReLU(inplace=True)

    def forward(self, x):
        conv1 = self.conv1(x)
        bn1   = self.bn1(conv1)
        relu1 = self.relu1(bn1)

        conv2 = self.conv2(relu1)
        bn2   = self.bn2(conv2)

        skip  = self.skip(x)
        return self.relu(bn2 + skip)

class MergeUp(nn.Module):
    def forward(self, up1, up2):
        return up1 + up2

def make_layer(k, inp_dim, out_dim, modules, layer=convolution, **kwargs):
    layers = [layer(k, inp_dim, out_dim, **kwargs)]
    for _ in range(1, modules):
        layers.append(layer(k, out_dim, out_dim, **kwargs))
    return nn.Sequential(*layers)

def make_layer_revr(k, inp_dim, out_dim, modules, layer=convolution, **kwargs):
    layers = []
    for _ in range(modules - 1):
        layers.append(layer(k, inp_dim, inp_dim, **kwargs))
    layers.append(layer(k, inp_dim, out_dim, **kwargs))
    return nn.Sequential(*layers)

def make_pool_layer(dim):
    return nn.MaxPool2d(kernel_size=2, stride=2)

def make_unpool_layer(dim):
    return nn.Upsample(scale_factor=2)

def make_merge_layer(dim):
    return MergeUp()

def make_kp_layer(cnv_dim, curr_dim, out_dim):
    return nn.Sequential(
        convolution(3, cnv_dim, curr_dim, with_bn=False),
        nn.Conv2d(curr_dim, out_dim, (1, 1))
    )

def make_cnv_layer(inp_dim, out_dim):
    return convolution(3, inp_dim, out_dim)

def make_inter_layer(dim):
    return residual(3, dim, dim)

def _gather_feat(feat, ind, mask=None):
    dim  = feat.size(2)
    ind  = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    feat = feat.gather(1, ind)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat

def _tranpose_and_gather_feat(feat, ind):
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))
    feat = _gather_feat(feat, ind)
    return feat

def _nms(heat, kernel=1):
    pad = (kernel - 1) // 2

    #print('---------------')
    #print('INSIDE NMS')
    #print('HEAT_SHAPE: {}\nHEAT: {}'.format(heat.shape, heat))

    hmax = nn.functional.max_pool2d(heat, (kernel, kernel), stride=1, padding=pad)

    #print('HMAX_SHAPE: {}\nHMAX: {}'.format(hmax.shape, hmax))

    keep = (hmax == heat).float()

    #print('KEEP_SHAPE: {}\nKEEP: {}'.format(keep.shape, keep))
    #print('HEAT*KEEP_SHAPE: {}\nHEAT*KEEP: {}'.format((heat*keep).shape, (heat*keep)))
    #print('EXITING NMS')
    #print('---------------')
    #print('INSIDE DECODE')

    return heat * keep

def _topk(scores, K=20):

    #print('---------------')
    #print('INSIDE TOPK')

    batch, cat, height, width = scores.size()

    #print('SCORES_SHAPE: {}\nSCORES: {}'.format(scores.shape, scores))

    topk_scores, topk_inds = torch.topk(scores.view(batch, -1), K)

    #print('TOPK_SCORES_SHAPE: {}\nTOPK_SCORES: {}'.format(topk_scores.shape, topk_scores))
    #print('TOPK_INDS_SHAPE: {}\nTOPK_INDS: {}'.format(topk_inds.shape, topk_inds))

    topk_clses = (topk_inds / (height * width)).int()

    #print('TOPK_CLASSES_SHAPE: {}\nTOPK_CLASSES: {}'.format(topk_clses.shape, topk_clses))

    topk_inds = topk_inds % (height * width)

    #print('TOPK_INDS_NORMED_SHAPE: {}\nTOPK_INDS_NORMED: {}'.format(topk_inds.shape, topk_inds))

    topk_ys   = (topk_inds / width).int().float()

    #print('TOPK_YS_SHAPE: {}\nTOPK_YS: {}'.format(topk_ys.shape, topk_ys))

    topk_xs   = (topk_inds % width).int().float()

    #print('TOPK_XS_SHAPE: {}\nTOPK_XS: {}'.format(topk_xs.shape, topk_xs))
    #print('EXITING TOPK')
    #print('---------------')
    #print('INSIDE DECODE')


    return topk_scores, topk_inds, topk_clses, topk_ys, topk_xs

def _decode(
    tl_heat, br_heat, tl_tag, br_tag, tl_regr, br_regr, 
    K = 100, kernel = 1, ae_threshold = 1, num_dets = 1000
):
    batch, cat, height, width = tl_heat.size()

    #print('TL_HEAT_SHAPE: {}\nTL_HEAT: {}'.format(tl_heat.shape, tl_heat))

    tl_heat = torch.sigmoid(tl_heat)
    br_heat = torch.sigmoid(br_heat)
    #print('GT_TL_HEAT_SUM_IN_DECODE_AFTER_SIGMOID:\n{}'.format(tl_heat.sum()))
    #print('GT_BR_HEAT_SUM_IN_DECODE_AFTER_SIGMOID:\n{}'.format(br_heat.sum()))
    #print('GT_TL_HEAT_IN_DECODE_AFTER_SIGMOID:\n{}'.format(tl_heat))

    #print('TL_HEAT_SIGMOID_SHAPE: {}\nTL_HEAT_SIGMOID: {}'.format(tl_heat.shape, tl_heat))

    # perform nms on heatmaps
    tl_heat = _nms(tl_heat, kernel=kernel)
    br_heat = _nms(br_heat, kernel=kernel)
    #print('GT_TL_HEAT_SUM_IN_DECODE_AFTER_NMS:\n{}'.format(tl_heat.sum()))
    #print('GT_BR_HEAT_SUM_IN_DECODE_AFTER_NMS:\n{}'.format(br_heat.sum()))


    tl_scores, tl_inds, tl_clses, tl_ys, tl_xs = _topk(tl_heat, K=K)
    br_scores, br_inds, br_clses, br_ys, br_xs = _topk(br_heat, K=K)
    #print('TL_SCORES_SHAPE_AFTER_TOPK: {}'.format(tl_scores.shape))
    #print('BR_CLSES_AFTER_TOPK:\n{}'.format(br_clses))
    #print('TL_SCORES_AFTER_TOPK: {}\nTL_INDS_AFTER_TOPK: {}\nTL_CLSES_AFTER_TOPK: {}\nTL_YS_AFTER_TOPK: {}\nTL_XS_AFTER_TOPK: {}\nBR_YS_AFTER_TOPK: {}'.format(tl_scores, tl_inds, tl_clses, tl_ys, tl_xs, br_ys))
   
    # Idáig elviekben jó !

    tl_ys = tl_ys.view(batch, K, 1).expand(batch, K, K)
    tl_xs = tl_xs.view(batch, K, 1).expand(batch, K, K)
    br_ys = br_ys.view(batch, 1, K).expand(batch, K, K)
    br_xs = br_xs.view(batch, 1, K).expand(batch, K, K)
    #print('TL_YS_SHAPE: {}\nTL_YS: {}'.format(tl_ys.shape, tl_ys))
    #print('BR_YS_SHAPE: {}\nBR_YS: {}'.format(br_ys.shape, br_ys))

    
    if tl_regr is not None and br_regr is not None:
        #print('TL_REGR_SHAPE: {}'.format(tl_regr.shape))
        tl_regr = _tranpose_and_gather_feat(tl_regr, tl_inds)   # ?????????? szerintem a topk-s index amit jelenleg vissza ad az nem jó, mert azok egy classra vonatkoznak
        #print('TL_REGR_TRANSANDGATH_SHAPE: {}'.format(tl_regr.shape))
        tl_regr = tl_regr.view(batch, K, 1, 2)
        #print('TL_REGR_TRANSANDGATH_VIEW_SHAPE: {}'.format(tl_regr.shape))
        br_regr = _tranpose_and_gather_feat(br_regr, br_inds)
        #print('BR_REGR_TRANSANDGATH_SHAPE: {}\nBR_REGR_TRANSANDGATH: {}'.format(br_regr.shape, br_regr))
        br_regr = br_regr.view(batch, 1, K, 2)
        #print('BR_REGR_TRANSANDGATH__VIEW_SHAPE: {}\nBR_REGR_TRANSANDGATH_VIEW: {}'.format(br_regr.shape, br_regr))

        #print('TL_XS_SHAPE: {}'.format(tl_xs.shape))
        #print('TL_REGR_TRANSANDGATH_VIEW_SHAPE: {}'.format(tl_regr[..., 0].shape))

        tl_xs = tl_xs + tl_regr[..., 0]
        #print('TL_XS_+_TL_REGR_SHAPE: {}\nTL_XS_+_TL_REGR: {}'.format(tl_xs.shape, tl_xs))
        tl_ys = tl_ys + tl_regr[..., 1]

        #print('BR_YS_SHAPE: {}\nBR_YS: {}'.format(br_ys.shape, br_ys))
        #print('BR_REGR_TRANSANDGATH__VIEW_SHAPE: {}\nBR_REGR_TRANSANDGATH_VIEW: {}'.format(br_regr[..., 1].shape, br_regr[..., 1]))
        br_xs = br_xs + br_regr[..., 0]
        br_ys = br_ys + br_regr[..., 1]
        #print('BR_YS_+_BR_REGR_SHAPE: {}\nBR_YS_+_BR_REGR: {}'.format(br_ys.shape, br_ys))
        #print('TL_XS_SHAPE: {}'.format(tl_xs.shape))
    
    # all possible boxes based on top k corners (ignoring class)
    bboxes = torch.stack((tl_xs, tl_ys, br_xs, br_ys), dim=3)
    #print('BBOXES_SHAPE: {}'.format(bboxes.shape))
    #print('BBOXES:\n{}'.format(bboxes))

    tl_tag = _tranpose_and_gather_feat(tl_tag, tl_inds)
    #print('TL_TAG_TRANSANDGATH_SHAPE: {}'.format(tl_tag.shape))
    tl_tag = tl_tag.view(batch, K, 1)
    #print('TL_TAG_VIEW_SHAPE: {}'.format(tl_tag.shape))
    br_tag = _tranpose_and_gather_feat(br_tag, br_inds)
    br_tag = br_tag.view(batch, 1, K)
    #print('BR_TAG_VIEW_SHAPE: {}'.format(br_tag.shape))
    dists  = torch.abs(tl_tag - br_tag)
    #print('DIST_SHAPE: {}'.format(dists.shape))
    #print('DIST:\n{}'.format(dists))


    # Iidáig is jó elviekben!

    tl_scores = tl_scores.view(batch, K, 1).expand(batch, K, K)
    br_scores = br_scores.view(batch, 1, K).expand(batch, K, K)
    scores    = (tl_scores + br_scores) / 2
    #print('TL_SCORES_AFTER_VIEW\n{}\nBR_SCORES_AFTER_VIEW\n{}'.format(tl_scores, br_scores))
    #print('SCORES_SHAPE: {}\nSCORES:\n{}'.format(scores.shape, scores))

    # reject boxes based on classes
    tl_clses = tl_clses.view(batch, K, 1).expand(batch, K, K)
    br_clses = br_clses.view(batch, 1, K).expand(batch, K, K)
    cls_inds = (tl_clses != br_clses)
    #cls_ind_lol = (tl_clses == br_clses)
    #print('CLS_INDS:\n{}'.format(cls_inds))

    # reject boxes based on distances
    dist_inds = (dists > ae_threshold)

    # reject boxes based on widths and heights
    width_inds  = (br_xs < tl_xs)
    height_inds = (br_ys < tl_ys)

    scores[cls_inds]    = -1
    scores[dist_inds]   = -1
    scores[width_inds]  = -1
    scores[height_inds] = -1
    
    #print('TL_SCORES_SUM:\n{}'.format(tl_scores.sum()))
    #print('BR_SCORES_SUM:\n{}'.format(br_scores.sum()))
    #print('SCORES_AFTER_REJECTIONS:\n{}'.format(scores))
    #print('SCORES_AFTER_REJECTIONS_SUM:\n{}'.format(scores.clamp(min=0).sum()))

   

    scores = scores.view(batch, -1)
    scores, inds = torch.topk(scores, num_dets)
    #print('SCORES_AFTER_TOPK:\n{}'.format(scores))
    #print('SCORES_AFTER_TOPK_SUM:\n{}'.format(scores.clamp(min=0).sum()))
    scores = scores.unsqueeze(2)
    #print('SCORES_SHAPE: {}\nSCORES: {}'.format(scores.shape, scores))

    bboxes = bboxes.view(batch, -1, 4)
    bboxes = _gather_feat(bboxes, inds)
    #print('BBOXES_SHAPE: {}\nBBOXES: {}'.format(bboxes.shape, bboxes))

    clses  = tl_clses.contiguous().view(batch, -1, 1)
    clses  = _gather_feat(clses, inds).float()
    #print('CLSES_SHAPE: {}\nCLSES: {}'.format(clses.shape, clses))

    tl_scores = tl_scores.contiguous().view(batch, -1, 1)
    tl_scores = _gather_feat(tl_scores, inds).float()
    br_scores = br_scores.contiguous().view(batch, -1, 1)
    br_scores = _gather_feat(br_scores, inds).float()
    #print('TL_SCORES_SHAPE: {}\nTL_SCORES: {}'.format(tl_scores.shape, tl_scores))
    #print('BR_SCORES_SHAPE: {}\nBR_SCORES: {}'.format(br_scores.shape, br_scores))

    detections = torch.cat([bboxes, scores, tl_scores, br_scores, clses], dim=2)
    return detections
