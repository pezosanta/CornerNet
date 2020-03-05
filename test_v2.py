import torch
import cv2
import numpy as np
import json
import os
from torch.utils.tensorboard import SummaryWriter
from pprint import pprint
from NonMaxSuppression.nms import soft_nms, soft_nms_merge


train_annotation_path = "../BDD100K/bdd100k_labels_images_train.json"
new_train_annotation_path = "../BDD100K/new_bdd100k_labels_images_train.json"
val_annotation_path = "../BDD100K/bdd100k_labels_images_val.json"                      # First rename the original "bdd100k_labels_images_val.json" to 
                                                                                        # "bdd100k_labels_images_test.json" 

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
    with open('../Detections/Hourglass/mAP_detections_epoch_{}.json'.format(current_epoch), 'w') as f:
        json.dump(detections_json, f, ensure_ascii = False, indent = 4, cls = MyJSONEncoder)

def calculate_mAP(detections, names, orig_images, transformed_images, detections_json, nms_threshold):
    detections = detections.data.cpu().numpy()

    out_width = 160
    dets = detections.reshape(2, -1, 8)
    dets[1, :, [0, 2]] = out_width - dets[1, :, [2, 0]]
    detections = dets.reshape(1, -1, 8)
    classes    = detections[..., -1]
    classes    = classes[0]
    detections = detections[0]
    keep_inds  = (detections[:, 4] > -1)
    detections = detections[keep_inds]
    classes    = classes[keep_inds]
    categories = 10
    max_per_image = 100
    merge_bbox = False
    nms_algorithm   = 2 #"exp_soft_nms"
    weight_exp      = 8

    top_bboxes = {}
    for j in range(0, categories):                                                          #??????????????
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
        for j in range(1, categories)])
    if len(scores) > max_per_image:
        kth    = len(scores) - max_per_image
        thresh = np.partition(scores, kth)[kth]
        #print("THRESHOLD: {}".format(thresh))
        for j in range(1, categories):
            keep_inds = (top_bboxes[j][:, -1] >= thresh)
            top_bboxes[j] = top_bboxes[j][keep_inds]
    
    for j in range(0, categories):
        keep_inds = (top_bboxes[j][:, -1] > 0.3)    # > X -et átirni, ha kisebb biztonsággal prediktált bboxokat is akarunk!
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
    '''
    if os.path.isfile('../Detections/Hourglass/detections_test2.json'):
        with open('../Detections/Hourglass/detections_test2.json') as f:
            detections_json = json.load(f)
    else:
        print('CREATING JSON FILE!')
        detections_json = []
    '''
    for j in range(0, categories):
        #print(top_bboxes[j])
        for bbox in top_bboxes[j]:
            bbox[0:4] = bbox[0:4] * 8
            detections_json.append(
                {
                    "name": names[0],
                    "timestamp": 10000,
                    "category": categories_dict[j],
                    "bbox": bbox[0:4],
                    "score": 1
                }
            )    
    ''' 
    with open('../Detections/Hourglass/detections_test2.json', 'w') as f:
        json.dump(detections_json, f, ensure_ascii = False, indent = 4, cls = MyJSONEncoder)
    '''


    '''
    debug = False
    current_image_paths = '../BDD100K/train/' + names[0]
    #print(current_image_paths)
    
    if debug:
        #image_file = db.image_file(db_ind)
        image      = cv2.imread(current_image_paths)
        image      = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_name = current_image_paths.rsplit('/', 1)[1]
        #print(image)
        print(image_name)

        bboxes = {}
        for j in range(0, categories):
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
        print(image.shape)
        #cv2.imwrite(('/content/' + image_name), image)
        writer = SummaryWriter('logs/imagetest/')
        writer.add_image('test_v2', image, 0, dataformats='HWC')
        writer.close() 
    '''
