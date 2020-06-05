from PIL import Image
from skimage import io
import json
import numpy as np
import torch
import torchvision.transforms as Transforms
from torch.utils.data import Dataset
from test import generate_annotated_image
import math
import cv2

categories = [  'bus',
                'traffic light',
                'traffic sign',
                'person',
                'bike',
                'truck',
                'motor',
                'car',
                'train',
                'rider' ]

categories_dict = { 'bus': 0,
                    'traffic light': 1,
                    'traffic sign': 2,
                    'person': 3,
                    'bike': 4,
                    'truck': 5,
                    'motor': 6,
                    'car': 7,
                    'train': 8,
                    'rider': 9 }

# Docker
train_annotation_path   = "../../../logs/cornernet/BDD100K/bdd100k_labels_images_train.json"           # bdd100k_labels_images_train.json[0:60000]
val_annotation_path     = "../../../logs/cornernet/BDD100K/bdd100k_labels_images_val.json"             # bdd100k_labels_images_train.json[60000:]
test_annotation_path    = "../../../logs/cornernet/BDD100K/bdd100k_labels_images_test.json"            # bdd100k_labels_images_val.json
train_image_root        = "../../../logs/cornernet/BDD100K/train/"
val_image_root          = "../../../logs/cornernet/BDD100K/val/"

def get_annotations(mode):   
    if mode == 'Train':
        with open(train_annotation_path) as f:
            annotation_file = json.load(f)
    elif mode == 'Val':
        with open(val_annotation_path) as f:
            annotation_file = json.load(f)
    elif mode == 'Test':
        with open(test_annotation_path) as f:
            annotation_file = json.load(f)
    
    image_names = []
    detections = {}

    for image_ind, one_image_dets in enumerate(annotation_file):

        image_names.append(one_image_dets["name"])
        detections.update({one_image_dets["name"]: []})

        for obj_ind, one_object in enumerate(one_image_dets["labels"]):  
            
            one_object_ = []            

            if one_object["category"] in categories:
                one_object_.append(categories_dict[one_object["category"]])
                one_object_.append(one_object["box2d"]["x1"])
                one_object_.append(one_object["box2d"]["y1"])
                one_object_.append(one_object["box2d"]["x2"])
                one_object_.append(one_object["box2d"]["y2"])
            if len(one_object_) > 1:                               
                detections[one_image_dets["name"]].append(one_object_)
    
    return image_names, detections
    
def get_image(mode, name):
    if mode == 'Train':
        image_path = train_image_root + name
    elif mode == 'Val':
        image_path = train_image_root + name
    elif mode == 'Test':
        image_path = val_image_root + name
    
    image = io.imread(image_path)

    return image

def resize_image(image, size, detections = None, with_detections = False):
    height, width = image.shape[0:2]
    new_height = size[0]
    new_width  = size[1]
    
    resized_image = cv2.resize(image, (new_width, new_height), interpolation = cv2.INTER_LANCZOS4)    
    
    if with_detections == True:
        resized_detections = detections.copy()

        height_ratio = new_height / height
        width_ratio  = new_width  / width

        resized_detections[:, 1:5:2] = (resized_detections[:, 1:5:2] * width_ratio)
        resized_detections[:, 2:5:2] = (resized_detections[:, 2:5:2] * height_ratio)
       
        return resized_image, resized_detections
    else:
        return resized_image

def calculate_iou(startpoint, new_size, detections):
    x, y = startpoint
    new_height, new_width = new_size

    clipped_detections = detections.copy()

    clipped_detections[:, 1:5:2] = np.clip(clipped_detections[:, 1:5:2], x, x + new_width)
    clipped_detections[:, 2:5:2] = np.clip(clipped_detections[:, 2:5:2], y, y + new_height)

    intersection = np.multiply(np.subtract(clipped_detections[:, 3], clipped_detections[:, 1]), np.subtract(clipped_detections[:, 4], clipped_detections[:, 2]))    # Clipped objektumok területe

    union = np.multiply(np.subtract(detections[:, 3], detections[:, 1]), np.subtract(detections[:, 4], detections[:, 2]))                                           # Az objektumok eredeti területe

    ious = np.divide(intersection, union)

    return ious, clipped_detections   

def random_zoom(image, detections, size):
    scale  = np.random.uniform(0.25, 0.75)              

    height = int(image.shape[0] * scale)                  # A kivágott kép az eredeti méret (0.25 ... 0.75)-szerese
    width  = int(image.shape[1] * scale)

    x = np.random.randint(0, image.shape[1] - width)      # A kivágott kép kezdőpontjának (bal felső sarok) koordinátája
    y = np.random.randint(0, image.shape[0] - height)

    image = image[y:y+height, x:x+width, :]               # A kivágott kép előállítása

    clipped_detections = detections.copy()

    ious, clipped_detections = calculate_iou(startpoint = (x,y), new_size = (height, width), detections = clipped_detections)
    
    keep_inds  = (ious >= 0.1)                                           # Az (iou < 0.1)-tel rendelkező objektumok elhagyása    
    clipped_detections = clipped_detections[keep_inds]
    
    clipped_detections[:, 1:5:2] = (clipped_detections[:, 1:5:2] - x)    # Az új kezdőponthoz illesztjük az objektumokat
    clipped_detections[:, 2:5:2] = (clipped_detections[:, 2:5:2] - y)
    
    resized_image, resized_detections = resize_image(image = image, size = size, detections = clipped_detections, with_detections = True)

    return resized_image, resized_detections

def clip_detections(image, detections):
    detections    = np.array(detections.copy())
    height, width = image.shape[0:2]                                      # (720 x 1280)

    if detections.size != 0:
        detections[:, 1:5:2] = np.clip(detections[:, 1:5:2], 0, (width - 1))
        detections[:, 2:5:2] = np.clip(detections[:, 2:5:2], 0, (height - 1))
    
    return list(detections)

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

def draw_gaussian(heatmap, center, radius, k = 1):
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma = diameter / 6)

    x, y = center

    height, width = heatmap.shape[0:2]
    
    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap  = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
    np.maximum(masked_heatmap, masked_gaussian * k, out = masked_heatmap)

def gaussian2D(shape, sigma = 1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m+1,-n:n+1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0

    return h

class Dataset(Dataset):
    def __init__(self, mode):
        self.mode = mode
        self.device = torch.device('cuda:0')
        #self.device = torch.device('cpu')

        self.image_names, self.detections = get_annotations(mode = self.mode) 

        self.gaussian_bump = True
        self.gaussian_iou = 0.3
        self.gaussian_rad = -1    
        self.orig_size = [720, 1280]
        self.input_size = [384, 640]     # Original_input_size / 2 (with modified height, that is essential for the network)
        self.output_size = [96, 160]     # Original_input_size / 8 
        self.num_categories = 10
        self.max_tag_len = 160
        self.ColorJitter = Transforms.ColorJitter(brightness = 0.25, contrast = 0.25, saturation = 0.25, hue = 0.1) # 0.3, 0.4, 0.4, 0.2

    def __getitem__(self, index):
        name = self.image_names[index]

        detection = self.detections[name].copy()
        
        image = get_image(mode = self.mode, name = name)

        orig_image = image.copy()
        orig_detection = detection.copy()      
        
        if self.mode == 'Train':
            image, detection = self.random_transforms(image, detection)

        orig_image        = generate_annotated_image(image = orig_image, detections = orig_detection, mode = 'GT')
        orig_image        = torch.from_numpy(orig_image).float()
        #transformed_image = generate_annotated_image(image = image, detections = detection, mode = 'GT')        

        images, tl_tags, br_tags, tl_heatmaps, br_heatmaps, tag_masks, tl_regrs, br_regrs = self.create_groundtruth(image = image, detection = detection)

        return images, tl_tags, br_tags, tl_heatmaps, br_heatmaps, tag_masks, tl_regrs, br_regrs, name, orig_image#, transformed_image
    
    def random_transforms(self, image, detection):
        detection = np.array(detection)
        
        # Random zoom
        if (np.random.uniform() > 0.5):
            image, detection = random_zoom(image, detection, size = self.orig_size)
            #print('RANDOM ZOOM!')
        
        # Random vertical flip (random mirror)
        if (np.random.uniform() > 0.5):
            image[:] = image[:, ::-1, :]
            width = detection[:, 3] - detection[:, 1]
            detection[:, 1:5:2] = self.orig_size[1] - detection[:, 1:5:2]
            detection[:, 1] -= width[:]                                         # Top-Right corner --> Top-Left corner
            detection[:, 3] += width[:]                                         # Bottom-Left corner --> Bottom-Right corner
            #print('RANDOM MIRROR!')
        
        # Random Color Jitter
        if (np.random.uniform() > 0.5):
            image = Image.fromarray(image)            
            image = self.ColorJitter(image)
            image = np.array(image)
            #print('RANDOM COLOR JIT!')
        
        return image, list(detection)

    def create_groundtruth(self, image, detection):
        detection = clip_detections(image = image, detections = detection)
        image = resize_image(image = image, size = self.input_size)     # GT Detections are still original sized

        tl_heatmaps = np.zeros((self.num_categories, self.output_size[0], self.output_size[1]), dtype = np.float32)
        br_heatmaps = np.zeros((self.num_categories, self.output_size[0], self.output_size[1]), dtype = np.float32)
        tl_regrs    = np.zeros((self.max_tag_len, 2), dtype = np.float32)
        br_regrs    = np.zeros((self.max_tag_len, 2), dtype = np.float32)
        tl_tags     = np.zeros((self.max_tag_len), dtype = np.int64)
        br_tags     = np.zeros((self.max_tag_len), dtype = np.int64)
        tag_masks   = np.zeros((self.max_tag_len), dtype = np.uint8)
        tag_lens    = np.zeros((1, ), dtype = np.int32)

        width_ratio  = self.output_size[1] / self.orig_size[1] # 1/8
        height_ratio = self.output_size[0] / self.orig_size[0]
        
        for ind, det in enumerate(detection):
            category = int(det[0]) #- 1

            xtl, ytl = det[1], det[2]   # Original sized coordinates
            xbr, ybr = det[3], det[4]

            fxtl = (xtl * width_ratio)  # 1/8 sized coodinates (float numbers)
            fytl = (ytl * height_ratio)
            fxbr = (xbr * width_ratio)
            fybr = (ybr * height_ratio)

            xtl = int(fxtl)             # Floored 1/8 sized coordinates (int numbers)
            ytl = int(fytl) 
            xbr = int(fxbr) 
            ybr = int(fybr)

            if self.gaussian_bump:
                width  = det[3] - det[1]    # Original width
                height = det[4] - det[2]

                width  = math.ceil(width * width_ratio) # 1/8 width
                height = math.ceil(height * height_ratio)

                if self.gaussian_rad == -1:
                    radius = gaussian_radius(det_size = (height, width), min_overlap = self.gaussian_iou)
                    radius = max(0, int(radius))
                else:
                    radius = self.gaussian_rad

                draw_gaussian(tl_heatmaps[category], [xtl, ytl], radius)
                draw_gaussian(br_heatmaps[category], [xbr, ybr], radius)
            else:
                tl_heatmaps[category, ytl, xtl] = 1
                br_heatmaps[category, ybr, xbr] = 1

            tag_ind = tag_lens[0]

            tl_regrs[tag_ind, :] = [fxtl - xtl, fytl - ytl]
            br_regrs[tag_ind, :] = [fxbr - xbr, fybr - ybr]

            tl_tags[tag_ind] = ytl * self.output_size[1] + xtl
            br_tags[tag_ind] = ybr * self.output_size[1] + xbr

            tag_lens[0] += 1
        
        tag_len = tag_lens[0]

        tag_masks[:tag_len] = 1

        images      = torch.from_numpy(image / 255.0).float().to(device = self.device).permute(2, 0, 1)
        tl_heatmaps = torch.from_numpy(tl_heatmaps).to(device = self.device)
        br_heatmaps = torch.from_numpy(br_heatmaps).to(device = self.device)
        tl_regrs    = torch.from_numpy(tl_regrs).to(device = self.device)
        br_regrs    = torch.from_numpy(br_regrs).to(device = self.device)
        tl_tags     = torch.from_numpy(tl_tags).to(device = self.device)
        br_tags     = torch.from_numpy(br_tags).to(device = self.device)
        tag_masks   = torch.from_numpy(tag_masks).to(device = self.device)

        return images, tl_tags, br_tags, tl_heatmaps, br_heatmaps, tag_masks, tl_regrs, br_regrs

    def __len__(self): 
      return len(self.image_names)
