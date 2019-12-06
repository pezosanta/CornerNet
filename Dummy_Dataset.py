import glob
from PIL import Image
from skimage import io
import matplotlib.pyplot as plt
import json
import os
from os.path import exists, splitext, isdir, isfile, join, split, dirname
import sys
from pprint import pprint
import numpy as np
import torch
import math
from CornerNet import CornerNet
import cv2


with open("C://Users//Tony Stark//Desktop//Önálló laboratórium//bdd100k//labels//bdd100k_labels_images_val.json") as f:
    data = json.load(f)

#pprint(data[0])
pprint(data[0]["name"])
pprint(data[0]["labels"][0]["category"])
pprint(data[0]["labels"][0]["box2d"])
pprint(len(data[0]["labels"]))
pprint(len(data))
pprint(data[0]["labels"][0]["box2d"]['x1'])

#for det_ind, dets in enumerate(data):   
 #   print(det_ind, dets, '\n')

categories = [  'bus',
                'traffic light',
                'traffic sign',
                'person',
                'bicycle',
                'truck',
                'motorcycle',
                'car',
                'train',
                'rider' ]

categories_dict = { 'bus': 0,
                    'traffic light': 1,
                    'traffic sign': 2,
                    'person': 3,
                    'bicycle': 4,
                    'truck': 5,
                    'motorcycle': 6,
                    'car': 7,
                    'train': 8,
                    'rider': 9 }

detections = {}

for det_ind, dets in enumerate(data):   
    detection = [] 
    detections.update({dets["name"]: []})

    for obj_ind, objects in enumerate(dets["labels"]):
        #print(objects)
        object_ = []
        
        #object_.append(dets["name"]) 
        if objects["category"] in categories:
            object_.append(categories_dict[objects["category"]])
            object_.append(objects["box2d"]["x1"])
            object_.append(objects["box2d"]["y1"])
            object_.append(objects["box2d"]["x2"])
            object_.append(objects["box2d"]["y2"])
        if len(object_) > 1:
            detections[dets["name"]].append(object_)
    
    #detections[dets["name"]].append(detection)
    

print("lol")
#print(len(detections))
#print(len(detections['b1c66a42-6f7d68ca.jpg']))
#print(detections[0], detections[1])
#print(detections['b1c66a42-6f7d68ca.jpg'])
#print(detections['b1c66a42-6f7d68ca.jpg'][0], detections['b1c66a42-6f7d68ca.jpg'][1])

#for ind, dets in enumerate(detections['b1c66a42-6f7d68ca.jpg']):
#    print(dets)


index = 1575
val_image_paths = glob.glob("C://Users//Tony Stark//Desktop//Önálló laboratórium//bdd100k//images//100k//val//*.jpg")
#name = (val_image_paths[index].rsplit('.', 1)[0]).rsplit('\\', 1)[1]
name = val_image_paths[index].rsplit('\\', 1)[1]

print(detections[name])
print(detections[name][0])
print(detections[name][0][1])

#label = json.load('C://Users//Tony Stark//Desktop//Önálló laboratórium//bdd100k//label//bdd100k_labels_images_val.json')


#images = glob.glob(r'C:\Users\Tony Stark\Desktop\Önálló laboratórium\bdd100k\images\100k\val\\*.jpg')

#print(len(images))

#image = io.imread(r'C:\Users\Tony Stark\Desktop\Önálló laboratórium\bdd100k\images\100k\val\\b1c66a42-6f7d68ca.jpg')
image = io.imread(val_image_paths[index])


#image = myObj.image.type(torch.float32)
test = plt.figure('test_image')
#image_numpy = image.permute(1,2,0).numpy()/255.0
#plt.imshow(image)
    
#plt.show()


'''
top_left_corner = (x1, y1)
bottom_right_corner = (x2, y2)
'''


'''
with open("C://Users//Tony Stark//Desktop//Önálló laboratórium//bdd100k//bdd2coco_labels//bdd100k_labels_images_det_coco_val.json") as f:
    data = json.load(f)


pprint(data["annotations"][0])
'''
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def _resize_image(image, detection, size):
    detection    = detection.copy()
    print(len(detection), detection)
    height, width = image.shape[0:2]
    new_height, new_width = size

    image = cv2.resize(image, (new_width, new_height))
    
    height_ratio = new_height / height
    width_ratio  = new_width  / width
    for i in range(len(detection)):
        #print(detection[i][1:5:2])
        detection[i][1] *= width_ratio
        detection[i][3] *= width_ratio
        detection[i][2] *= height_ratio
        detection[i][4] *= height_ratio
    return image, detection

def _clip_detections(image, detections):
    detections    = detections.copy()
    height, width = image.shape[0:2]

    detections[:, 0:4:2] = np.clip(detections[:, 0:4:2], 0, width - 1)
    detections[:, 1:4:2] = np.clip(detections[:, 1:4:2], 0, height - 1)
    keep_inds  = ((detections[:, 2] - detections[:, 0]) > 0) & \
                 ((detections[:, 3] - detections[:, 1]) > 0)
    detections = detections[keep_inds]
    return detections

def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m+1,-n:n+1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h

def draw_gaussian(heatmap, center, radius, k=1):
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)

    x, y = center

    height, width = heatmap.shape[0:2]
    
    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap  = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
    np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)

def gaussian_radius(det_size, min_overlap):
    height, width = det_size

    a1  = 1
    b1  = (height + width)
    c1  = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1  = (b1 - sq1) / (2 * a1)

    a2  = 4
    b2  = 2 * (height + width)
    c2  = (1 - min_overlap) * width * height
    sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2  = (b2 - sq2) / (2 * a2)

    a3  = 4 * min_overlap
    b3  = -2 * min_overlap * (height + width)
    c3  = (min_overlap - 1) * width * height
    sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
    r3  = (b3 + sq3) / (2 * a3)
    return min(r1, r2, r3)



def kp_detection():
    '''
    data_rng   = system_configs.data_rng
    batch_size = system_configs.batch_size

    categories   = db.configs["categories"]
    input_size   = db.configs["input_size"]
    output_size  = db.configs["output_sizes"][0]

    border        = db.configs["border"]
    lighting      = db.configs["lighting"]
    rand_crop     = db.configs["rand_crop"]
    rand_color    = db.configs["rand_color"]
    rand_scales   = db.configs["rand_scales"]
    gaussian_bump = db.configs["gaussian_bump"]
    gaussian_iou  = db.configs["gaussian_iou"]
    gaussian_rad  = db.configs["gaussian_radius"]
    '''
    gaussian_bump = True
    gaussian_iou = 0.7
    gaussian_rad = -1
    #input_size = [720, 1280]
    #input_size = [768, 1280]
    input_size = [384, 640]
    #output_size = [180, 320]
    output_size = [96, 160]
    num_categories = 10

    index = 1575
    image = io.imread(val_image_paths[index])
        
    name = val_image_paths[index].rsplit('\\', 1)[1]
    detection = detections[name]
    #print('detection_check: ' + str(len(detection)) + ' ' + str(detection))
    #print('detection_check_2: ' + str(detection[0][1:5:2]))

    image, detection = _resize_image(image = image, detection = detection, size = (360, 640))

    print('resized_detection: ' + str(detection))

    npad = ((0, 24), (0, 0), (0, 0))
    image = np.pad(image, pad_width = npad, mode='constant', constant_values=0)
    plt.imshow(image)
    plt.show()

    #max_tag_len = 128
    max_tag_len = 160

    # allocating memory
    images      = np.zeros((3, input_size[0], input_size[1]), dtype = np.float32)
    tl_heatmaps = np.zeros((num_categories, output_size[0], output_size[1]), dtype = np.float32)
    br_heatmaps = np.zeros((num_categories, output_size[0], output_size[1]), dtype = np.float32)
    tl_regrs    = np.zeros((max_tag_len, 2), dtype = np.float32)
    br_regrs    = np.zeros((max_tag_len, 2), dtype = np.float32)
    tl_tags     = np.zeros((max_tag_len), dtype = np.int64)
    br_tags     = np.zeros((max_tag_len), dtype = np.int64)
    tag_masks   = np.zeros((max_tag_len), dtype = np.uint8)
    tag_lens    = np.zeros((1, ), dtype = np.int32)

    '''
    db_size = db.db_inds.size
    for b_ind in range(batch_size):
        if not debug and k_ind == 0:
            db.shuffle_inds()

        db_ind = db.db_inds[k_ind]
        k_ind  = (k_ind + 1) % db_size

        # reading image
        image_file = db.image_file(db_ind)
        image      = cv2.imread(image_file)

        # reading detections
        detections = db.detections(db_ind)

        # cropping an image randomly
        if not debug and rand_crop:
            image, detections = random_crop(image, detections, rand_scales, input_size, border=border)
        else:
            image, detections = _full_image_crop(image, detections)

        image, detections = _resize_image(image, detections, input_size)
        detections = _clip_detections(image, detections)

        width_ratio  = output_size[1] / input_size[1]
        height_ratio = output_size[0] / input_size[0]

        # flipping an image randomly
        if not debug and np.random.uniform() > 0.5:
            image[:] = image[:, ::-1, :]
            width    = image.shape[1]
            detections[:, [0, 2]] = width - detections[:, [2, 0]] - 1

        if not debug:
            image = image.astype(np.float32) / 255.
            if rand_color:
                color_jittering_(data_rng, image)
                if lighting:
                    lighting_(data_rng, image, 0.1, db.eig_val, db.eig_vec)
            normalize_(image, db.mean, db.std)
        images[b_ind] = image.transpose((2, 0, 1))
        '''
    width_ratio  = output_size[1] / input_size[1]
    height_ratio = output_size[0] / input_size[0]

    print(width_ratio, height_ratio)
    
    for ind, det in enumerate(detection):
        print("haaaaaaaaaaaaa: " + str(det))
        category = det[0]

        xtl, ytl = det[1], det[2]
        xbr, ybr = det[3], det[4]

        print(xtl, ytl, xbr, ybr)

        fxtl = (xtl * width_ratio)
        fytl = (ytl * height_ratio)
        fxbr = (xbr * width_ratio)
        fybr = (ybr * height_ratio)

        xtl = int(fxtl)
        ytl = int(fytl)
        xbr = int(fxbr)
        ybr = int(fybr)

        if gaussian_bump:
            width  = det[3] - det[1]
            height = det[4] - det[2]

            width  = math.ceil(width * width_ratio)
            height = math.ceil(height * height_ratio)

            if gaussian_rad == -1:
                radius = gaussian_radius((height, width), gaussian_iou)
                radius = max(0, int(radius))
            else:
                radius = gaussian_rad

            draw_gaussian(tl_heatmaps[category], [xtl, ytl], radius)
            draw_gaussian(br_heatmaps[category], [xbr, ybr], radius)
        else:
            tl_heatmaps[category, ytl, xtl] = 1
            br_heatmaps[category, ybr, xbr] = 1

        tag_ind = tag_lens
        tl_regrs[tag_ind, :] = [fxtl - xtl, fytl - ytl]
        br_regrs[tag_ind, :] = [fxbr - xbr, fybr - ybr]
        tl_tags[tag_ind] = ytl * output_size[1] + xtl
        br_tags[tag_ind] = ybr * output_size[1] + xbr
        tag_lens += 1

    print("tl_heatmap_shape:{} , tl_heatmap:{} ".format(tl_heatmaps.shape, tl_heatmaps[1, 75:80, 174:180]))
    print("br_heatmap_shape:{} , br_heatmap:{} ".format(br_heatmaps.shape, br_heatmaps[1, 79:85, 176:182]))
    print("tl_tags_shape:{}, tl_tags:{} ".format(tl_tags.shape, tl_tags))
    print("br_tags_shape:{}, br_tags:{} ".format(br_tags.shape, br_tags))
    print("tl_regrs_shape:{}, tl_regrs:{} ".format(tl_regrs.shape, tl_regrs))
    print("br_regrs_shape:{}, br_regrs:{} ".format(br_regrs.shape, br_regrs))
    

    #for b_ind in range(batch_size):
    tag_len = tag_lens[0]
    print("tag_len:{} ".format(tag_len))
    tag_masks[:tag_len] = 1
    print("tag_masks_shape:{}, tag_masks:{} ".format(tag_masks.shape, tag_masks))

    images      = torch.from_numpy(image/255.0).float().permute(2, 0, 1)
    tl_heatmaps = torch.from_numpy(tl_heatmaps)
    br_heatmaps = torch.from_numpy(br_heatmaps)
    tl_regrs    = torch.from_numpy(tl_regrs)
    br_regrs    = torch.from_numpy(br_regrs)
    tl_tags     = torch.from_numpy(tl_tags)
    br_tags     = torch.from_numpy(br_tags)
    tag_masks   = torch.from_numpy(tag_masks)

    return {
        "xs": [images, tl_tags, br_tags],
        "ys": [tl_heatmaps, br_heatmaps, tag_masks, tl_regrs, br_regrs]
    }

if __name__ == '__main__':

    n = 5
    dims    = [256, 256, 384, 384, 384, 512]
    modules = [2, 2, 2, 2, 2, 4]
    out_dim = 10

    returns = kp_detection()

    xs = returns['xs']
    ys = returns['ys']


    print('xs[0]_before:{} '.format(xs[0].shape))
    xs[0] = xs[0].unsqueeze(0)
    print('xs[0]_after:{} '.format(xs[0].shape))
    xs[1] = xs[1].unsqueeze(0)
    xs[2] = xs[2].unsqueeze(0)
    ys[2] = ys[2].unsqueeze(0)
    
    print('ys_mask_shape:{}, ys_mask:{} '.format(ys[2].shape, ys[2]))
    push_mask_1 = ys[2].unsqueeze(1)
    push_mask_2 = ys[2].unsqueeze(2)
    push_mask = push_mask_1 + push_mask_2
    print('push_mask_1_shape:{}, push_mask_1:{} '.format(push_mask_1.shape, push_mask_1))
    print('push_mask_2_shape:{}, push_mask_2:{} '.format(push_mask_2.shape, push_mask_2))
    print('push_mask_shape:{}, push_mask:{} '.format(push_mask.shape, push_mask))
    
    model = CornerNet(n = n, nstack = 2, dims = dims, modules = modules, out_dim = out_dim).eval()
    
    num_params = count_parameters(model)
    print('num_params: ' + str(num_params))
    
    outs = model(*xs)

    print(outs[0].shape, outs[2].shape, outs[4].shape)
    print(outs[4])

    '''
    a = np.ones((720, 1280, 3))
    npad = ((0, 48), (0, 0), (0, 0))
    b = np.pad(a, pad_width=npad, mode='constant', constant_values=0)
    print(b.shape)
    '''

    '''
    test_dataset = VAE_dataset(batch_size = 1)
    train_loader = DataLoader(train_dataset, batch_size = 1, shuffle = True)

    test_loader_iter = iter(train_loader)
    first_tensor_img = next(test_loader_iter)
    '''

    