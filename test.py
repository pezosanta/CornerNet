import torch
import cv2
import numpy as np
import json
from torchvision.utils import make_grid
from torch.utils.tensorboard import SummaryWriter
from collections import defaultdict
from NonMaxSuppression.nms import soft_nms, soft_nms_merge
from cornernet import kp as cornernet


train_annotation_path       = "../BDD100K/bdd100k_labels_images_train.json"
new_train_annotation_path   = "../BDD100K/new_bdd100k_labels_images_train.json"
val_annotation_path         = "../BDD100K/bdd100k_labels_images_val.json"              # First rename the original "bdd100k_labels_images_val.json" to 
                                                                                        # "bdd100k_labels_images_test.json"
map_detection_path          = "../Detections/hourglass/"
gt_detection_path           = "../../../logs/cornernet/BDD100K/detection_val.json"

train_val_image_root        = "../../../logs/cornernet/BDD100K/train/"

reverse_categories_dict = { 0: 'bus',
                            1: 'traffic light',
                            2: 'traffic sign',
                            3: 'person',
                            4: 'bicycle',
                            5: 'truck',
                            6: 'motorcycle',
                            7: 'car',
                            8: 'train',
                            9: 'rider' }

orig_size = [720, 1280]
output_size = [96, 160]

class MyJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)

def separate_train_annotations():
    with open(train_annotation_path) as f:
        annotation_file = json.load(f)

    with open(val_annotation_path, 'w') as f:
        json.dump(annotation_file[60000:], f, ensure_ascii = False, indent = 4)

    with open(new_train_annotation_path, 'w') as f:
        json.dump(annotation_file[0:60000], f, ensure_ascii = False, indent = 4)        # Then rename the new json file to "bdd100k_labels_images_train.json

def save_detections(detections_json, current_epoch):
    with open(map_detection_path + "cornernet_hourglass_map_detections_epoch_{}.json".format(current_epoch), 'w') as f:
        json.dump(detections_json, f, ensure_ascii = False, indent = 4, cls = MyJSONEncoder)

def generate_detections(detections, names, orig_images, detections_json, nms_threshold, min_score, current_iter, annot_image_saving_iterations, \
                            pred_detections_images, gt_detections_images):
    out_width           = 160
    categories          = 10
    max_per_image       = 100
    merge_bbox          = False
    nms_algorithm       = 2                                # "exp_soft_nms"
    weight_exp          = 8

    width_ratio         = orig_size[1] / output_size[1]                                         
    height_ratio        = orig_size[0] / output_size[0]

    detections = detections.data.cpu().numpy()

    dets                = detections.reshape(2, -1, 8)
    dets[1, :, [0, 2]]  = out_width - dets[1, :, [2, 0]]
    detections          = dets.reshape(1, -1, 8)
    classes             = detections[..., -1]
    classes             = classes[0]
    detections          = detections[0]
    keep_inds           = (detections[:, 4] > -1)
    detections          = detections[keep_inds]
    classes             = classes[keep_inds]

    top_bboxes = {}
    for j in range(0, categories):                                                          
            keep_inds = (classes == j)            
            top_bboxes[j] = detections[keep_inds][:, 0:7].astype(np.float32)

            if merge_bbox:
                soft_nms_merge(top_bboxes[j], Nt=nms_threshold, method=nms_algorithm, weight_exp=weight_exp)
            else:
                soft_nms(top_bboxes[j], Nt=nms_threshold, method=nms_algorithm)
            top_bboxes[j] = top_bboxes[j][:, 0:5]
   
    scores = np.hstack([
        top_bboxes[j][:, -1] 
        for j in range(1, categories)])
        
    if len(scores) > max_per_image:
        kth    = len(scores) - max_per_image
        thresh = np.partition(scores, kth)[kth]
        for j in range(1, categories):
            keep_inds = (top_bboxes[j][:, -1] >= thresh)
            top_bboxes[j] = top_bboxes[j][keep_inds]
    
    # Keep only (score > min_score) bboxes
    for j in range(0, categories):
        keep_inds = (top_bboxes[j][:, -1] > min_score)    
        top_bboxes[j] = top_bboxes[j][keep_inds]

    '''
    print('TOP_BBOXES_LEN: {}\nTOP_BBOXES: {}'.format(len(top_bboxes), top_bboxes))
    print('BUS_BBOXES_LEN: {}'.format(len(top_bboxes[0])))
    print('TRAFFIC_LIGHT_BBOXES_LEN: {}'.format(len(top_bboxes[1])))
    print('TRAFFIC_SIGN_BBOXES_LEN: {}'.format(len(top_bboxes[2])))
    print('PERSON_BBOXES_LEN: {}'.format(len(top_bboxes[3])))
    print('BICYCLE_BBOXES_LEN: {}'.format(len(top_bboxes[4])))
    print('TRUCK_BBOXES_LEN: {}'.format(len(top_bboxes[5])))
    print('MOTORCYCLE_BBOXES_LEN: {}'.format(len(top_bboxes[6])))
    print('CAR_BBOXES_LEN: {}'.format(len(top_bboxes[7])))
    print('TRAIN_BBOXES_LEN: {}'.format(len(top_bboxes[8])))
    print('RIDER_BBOXES_LEN: {}'.format(len(top_bboxes[9])))
    '''    

    # Save the detected bounding boxes into detections_json
    for j in range(0, categories):
        for bbox in top_bboxes[j]:
            bbox[0:4:2] = bbox[0:4:2] * width_ratio             # Transform bbox coordinates to the original image size (96x160 --> 720x1280)
            bbox[1:4:2] = bbox[1:4:2] * height_ratio
            detections_json.append(
                {
                    "name": names[0],
                    "timestamp": 10000,
                    "category": reverse_categories_dict[j],
                    "bbox": bbox[0:4],
                    "score": 1
                }
            )   

    # Create and save annotated image from predictions along with the corresponding gt annotated image
    if current_iter in (annot_image_saving_iterations[0], annot_image_saving_iterations[1], annot_image_saving_iterations[2], annot_image_saving_iterations[3]):
        pred_image = generate_annotated_image(detections = top_bboxes, mode = 'PRED', image = None, image_name = names[0])
        pred_detections_images.append(torch.from_numpy(pred_image / 255.0).float().unsqueeze(dim = 0).permute(0,3,1,2))

        gt_detections_images.append((orig_images/255.0).permute(0,3,1,2))

def generate_annotated_image(detections, mode, image = None, image_name = None):
    num_categories = 10
    
    if mode == 'GT':
        top_bboxes = []
        for i in range(num_categories):
            cat_bboxes = []
            for j, det in enumerate(detections):
                if (det[0] == i):
                    cat_bboxes.append(det[1:])
            
            cat_bboxes = np.array(cat_bboxes)
            top_bboxes.append(cat_bboxes)

        annot_image = image

    elif mode == 'PRED':
        current_image_paths = train_val_image_root + image_name
        annot_image         = cv2.imread(current_image_paths)
        annot_image         = cv2.cvtColor(annot_image, cv2.COLOR_BGR2RGB)
        top_bboxes          = detections

    for i in range(0, num_categories):
        #keep_inds = (top_bboxes[j][:, -1])
        cat_name  = reverse_categories_dict[i]
        cat_size  = cv2.getTextSize(cat_name, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        color     = np.random.random((3, )) * 0.6 + 0.4
        color     = color * 255
        color     = color.astype(np.int32).tolist()
        for bbox in top_bboxes[i]:
            bbox  = (bbox[0:4]).astype(np.int32)
            bbox[0:4:2] = np.clip(bbox[0:4:2], 1, (orig_size[1] - 1))                   # This way the bboxes looks nicer on the edges
            bbox[1:4:2] = np.clip(bbox[1:4:2], 1, (orig_size[0] - 1))
            if bbox[1] - cat_size[1] - 2 < 0:
                cv2.rectangle(annot_image,
                    (bbox[0], bbox[1] + 2),
                    (bbox[0] + cat_size[0], bbox[1] + cat_size[1] + 2),
                    color, -1
                )
                cv2.putText(annot_image, cat_name, 
                    (bbox[0], bbox[1] + cat_size[1] + 2), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), thickness = 1
                )
            else:
                cv2.rectangle(annot_image,
                    (bbox[0], bbox[1] - cat_size[1] - 2),
                    (bbox[0] + cat_size[0], bbox[1] - 2),
                    color, -1
                )
                cv2.putText(annot_image, cat_name, 
                    (bbox[0], bbox[1] - 2), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), thickness = 1
                )
            cv2.rectangle(annot_image,
                (bbox[0], bbox[1]),
                (bbox[2], bbox[3]),
                color, 2
            )
    return annot_image

def make_grid_image(gt_detections_images, pred_detections_images):
    concat_gt_image     = torch.cat((gt_detections_images[0], gt_detections_images[1], gt_detections_images[2], gt_detections_images[3]), 0)
    concat_pred_image   = torch.cat((pred_detections_images[0], pred_detections_images[1], pred_detections_images[2], pred_detections_images[3]), 0)
    
    concat_image        = torch.cat((concat_pred_image, concat_gt_image.cpu()), 0)
  
    grid_image = make_grid(tensor = concat_image, nrow = 4, padding = 10, pad_value = 55.0)

    return grid_image

def get_ap(recalls, precisions):
    # correct AP calculation
    # first append sentinel values at the end
    recalls = np.concatenate(([0.], recalls, [1.]))
    precisions = np.concatenate(([0.], precisions, [0.]))

    # compute the precision envelope
    for i in range(precisions.size - 1, 0, -1):
        precisions[i - 1] = np.maximum(precisions[i - 1], precisions[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(recalls[1:] != recalls[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((recalls[i + 1] - recalls[i]) * precisions[i + 1])
    return ap


def group_by_key(detections, key):
    groups = defaultdict(list)
    for d in detections:
        groups[d[key]].append(d)
    return groups


def cat_pc(gt, predictions, thresholds):
    """
    Implementation refers to https://github.com/rbgirshick/py-faster-rcnn
    """
    num_gts = len(gt)
    image_gts = group_by_key(gt, 'name')
    image_gt_boxes = {k: np.array([[float(z) for z in b['bbox']]
                                   for b in boxes])
                      for k, boxes in image_gts.items()}
    image_gt_checked = {k: np.zeros((len(boxes), len(thresholds)))
                        for k, boxes in image_gts.items()}
    predictions = sorted(predictions, key=lambda x: x['score'], reverse=True)

    # go down dets and mark TPs and FPs
    nd = len(predictions)
    tp = np.zeros((nd, len(thresholds)))
    fp = np.zeros((nd, len(thresholds)))
    for i, p in enumerate(predictions):
        box = p['bbox']
        ovmax = -np.inf
        jmax = -1
        try:
            gt_boxes = image_gt_boxes[p['name']]
            gt_checked = image_gt_checked[p['name']]
        except KeyError:
            gt_boxes = []
            gt_checked = None

        if len(gt_boxes) > 0:
            # compute overlaps
            # intersection
            ixmin = np.maximum(gt_boxes[:, 0], box[0])
            iymin = np.maximum(gt_boxes[:, 1], box[1])
            ixmax = np.minimum(gt_boxes[:, 2], box[2])
            iymax = np.minimum(gt_boxes[:, 3], box[3])
            iw = np.maximum(ixmax - ixmin + 1., 0.)
            ih = np.maximum(iymax - iymin + 1., 0.)
            inters = iw * ih

            # union
            uni = ((box[2] - box[0] + 1.) * (box[3] - box[1] + 1.) +
                   (gt_boxes[:, 2] - gt_boxes[:, 0] + 1.) *
                   (gt_boxes[:, 3] - gt_boxes[:, 1] + 1.) - inters)

            overlaps = inters / uni
            ovmax = np.max(overlaps)
            jmax = np.argmax(overlaps)

        for t, threshold in enumerate(thresholds):
            if ovmax > threshold:
                if gt_checked[jmax, t] == 0:
                    tp[i, t] = 1.
                    gt_checked[jmax, t] = 1
                else:
                    fp[i, t] = 1.
            else:
                fp[i, t] = 1.

    # compute precision recall
    fp = np.cumsum(fp, axis=0)
    tp = np.cumsum(tp, axis=0)
    recalls = tp / float(num_gts)
    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    precisions = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = np.zeros(len(thresholds))
    for t in range(len(thresholds)):
        ap[t] = get_ap(recalls[:, t], precisions[:, t])

    return recalls, precisions, ap


def evaluate_detections(current_epoch, mean_ap_thresholds):
    result_path = map_detection_path + "cornernet_hourglass_map_detections_epoch_{}.json".format(current_epoch)
    thresholds = mean_ap_thresholds

    gt = json.load(open(gt_detection_path, 'r'))
    pred = json.load(open(result_path, 'r'))
    cat_gt = group_by_key(gt, 'category')
    cat_pred = group_by_key(pred, 'category')
    cat_list = sorted(cat_gt.keys())
    
    aps = np.zeros((len(thresholds), len(cat_list)))
    for i, cat in enumerate(cat_list):
        if cat in cat_pred:
            r, p, ap = cat_pc(cat_gt[cat], cat_pred[cat], thresholds)
            aps[:, i] = ap
    aps *= 100
   
    m_ap_0 = np.mean(aps[0, :])
    m_ap_1 = np.mean(aps[1, :])
    
    mean_ap_0, mean_ap_1, breakdown_0, breakdown_1 = m_ap_0, m_ap_1, aps[0, :].flatten().tolist(), aps[1, :].flatten().tolist()

    #print('{:.2f}'.format(mean_ap),
    #      ', '.join(['{:.2f}'.format(n) for n in breakdown]))
    
    return mean_ap_0, mean_ap_1, breakdown_0, breakdown_1, cat_list

def create_graph():
    '''
    To use this function, first some modifications are required in (CornerNet.py / kp).
    1.) Modify the forward function's inputs as below:
            def forward(self, image, tl_tags, br_tags, mode = 'Train', ae_threshold = 0.5, top_k = 100, kernel = 3):
            if ((mode == 'Train') or (mode == 'Val')):
                image   = image
                tl_inds = tl_tags
                br_inds = br_tags
    2.) Modify the forward function's return statement as below:
            return outs[0], outs[1], outs[2], outs[3], outs[4], outs[5]
    '''
    
    writer = SummaryWriter('logs/cornernet_hourglass_graph/') 

    # Model Hyperparameters
    n = 5
    nstack = 2
    dims    = [256, 256, 384, 384, 384, 512]
    modules = [2, 2, 2, 2, 2, 4]
    out_dim = 10

    model = cornernet(n = n, nstack = nstack, dims = dims, modules = modules, out_dim = out_dim).cuda()

    # Creating default inputs 
    image       = torch.zeros(1, 3, 384, 640).float().cuda()
    tl_tags     = torch.from_numpy(np.zeros((1, 160), dtype = np.int64)).cuda()
    br_tags     = torch.from_numpy(np.zeros((1, 160), dtype = np.int64)).cuda()

    writer.add_graph(model, input_to_model=(image, tl_tags, br_tags), verbose = True)
    writer.close()
    
    # Some errors occur here because of inplace operations in Corner_pooling.py
    torch.onnx.export(model = model, args = (image, tl_tags, br_tags), f = "cornernet.onnx", verbose = False, opset_version = 11, export_params = True, 
        input_names = ['image', 'tl_tags', 'br_tags'], output_names = ['tl_heat', 'br_heat', 'tl_tag', 'br_tag', 'tl_regr', 'br_regr'])


if __name__ == "__main__":
    create_graph()