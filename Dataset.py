import glob
from PIL import Image
from skimage import io
import matplotlib.pyplot as plt
import json
import os
from os.path import exists, splitext, isdir, isfile, join, split, dirname
import sys
#from pprint import pprint
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import math
from CornerNet import kp
import cv2
from losses import AELoss

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

categories_dict = { 'bus': 1,
                    'traffic light': 2,
                    'traffic sign': 3,
                    'person': 4,
                    'bicycle': 5,
                    'truck': 6,
                    'motorcycle': 7,
                    'car': 8,
                    'train': 9,
                    'rider': 10 }
# Windows
#train_annotation_path = "C://Users//Tony Stark//Desktop//Önálló laboratórium//bdd100k//labels//bdd100k_labels_images_train.json"
#val_annotation_path = "C://Users//Tony Stark//Desktop//Önálló laboratórium//bdd100k//labels//bdd100k_labels_images_val.json"
#train_image_root = "C://Users//Tony Stark//Desktop//Önálló laboratórium//bdd100k//images//100k//train//"
#val_image_root = "C://Users//Tony Stark//Desktop//Önálló laboratórium//bdd100k//images//100k//val//"

# Google Colab
#train_annotation_path = "/content/bdd100k_labels_images_train.json"
#val_annotation_path = "/content/bdd100k_labels_images_val.json"
#train_image_root = "/content/train/"
#val_image_root = "/content/val/"

# Docker
train_annotation_path = "./BDD100K/bdd100k_labels_images_train.json"
val_annotation_path = "./BDD100K/bdd100k_labels_images_val.json"
train_image_root = "./BDD100K/train/"
val_image_root = "./BDD100K/val/"

def get_annotations(mode):
    
    if mode == 'Train':
        with open(train_annotation_path) as f:
            annotation_file = json.load(f)
    elif mode == 'Val':
        with open(val_annotation_path) as f:
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
        image_path = val_image_root + name
    
    image = io.imread(image_path)

    return image

def _resize_image(image, size):
    #detection    = detection.copy()
    #print(len(detection), detection)
    height, width = image.shape[0:2]
    new_height, new_width = size

    image = cv2.resize(image, (new_width, new_height))
    
    '''
    height_ratio = new_height / height
    width_ratio  = new_width  / width
    for i in range(len(detection)):
        #print(detection[i][1:5:2])
        detection[i][1] *= width_ratio
        detection[i][3] *= width_ratio
        detection[i][2] *= height_ratio
        detection[i][4] *= height_ratio
    '''
    return image#, detection

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
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)

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
        self.gaussian_iou = 0.7
        self.gaussian_rad = -1   
        #self.input_size = [384, 640]    # Original_input_size / 2 (+ height_padding)
        self.input_size = [768, 1280]
        self.output_size = [96, 160]    # Original_input_size / 8 (+ height_padding)
        self.num_categories = 10
        self.max_tag_len = 160

    def __getitem__(self, index):
        name = self.image_names[index]
        #print(name)
        detection = self.detections[name]
        #print('Number of objects on the image: {}'.format(len(detection)))
        #print('GT boxes in the image:\n{}'.format(detection))
        
        image = get_image(mode = self.mode, name = name)

        images, tl_tags, br_tags, tl_heatmaps, br_heatmaps, tag_masks, tl_regrs, br_regrs = self.transform(image = image, detection = detection)

        return images, tl_tags, br_tags, tl_heatmaps, br_heatmaps, tag_masks, tl_regrs, br_regrs#, name

    def transform(self, image, detection):
        npad = ((0, 48), (0, 0), (0, 0))
        image = np.pad(image, pad_width = npad, mode = 'constant', constant_values = 0)     # this does not effect the annotations   

        image = _resize_image(image = image, size = (384, 640))             

        tl_heatmaps = np.zeros((self.num_categories, self.output_size[0], self.output_size[1]), dtype = np.float32)
        br_heatmaps = np.zeros((self.num_categories, self.output_size[0], self.output_size[1]), dtype = np.float32)
        tl_regrs    = np.zeros((self.max_tag_len, 2), dtype = np.float32)
        br_regrs    = np.zeros((self.max_tag_len, 2), dtype = np.float32)
        tl_tags     = np.zeros((self.max_tag_len), dtype = np.int64)
        br_tags     = np.zeros((self.max_tag_len), dtype = np.int64)
        tag_masks   = np.zeros((self.max_tag_len), dtype = np.uint8)
        tag_lens    = np.zeros((1, ), dtype = np.int32)

        width_ratio  = self.output_size[1] / self.input_size[1] # 1/8
        height_ratio = self.output_size[0] / self.input_size[0]

        for ind, det in enumerate(detection):
            #print("haaaaaaaaaaaaa: " + str(det))
            category = int(det[0]) - 1

            xtl, ytl = det[1], det[2]   # Original sized coordinates
            xbr, ybr = det[3], det[4]

            #print(xtl, ytl, xbr, ybr)

            fxtl = (xtl * width_ratio)  # 1/8 sized coodinates (float numbers)
            fytl = (ytl * height_ratio)
            fxbr = (xbr * width_ratio)
            fybr = (ybr * height_ratio)

            xtl = int(fxtl)             # Floored 1/8 sized coordinates (int numbers)
            ytl = int(fytl) 
            xbr = int(fxbr) 
            ybr = int(fybr)

            #print(xtl, (fxtl-xtl), ytl, (fytl-ytl), xbr, (fxbr-xbr), ybr, (fybr-ybr)) 

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
            #print('TAG_IND: {}'.format(tag_ind))
            tl_regrs[tag_ind, :] = [fxtl - xtl, fytl - ytl]
            #print('TL_REGRS[tag_ing]: {}'.format(tl_regrs[tag_ind]))
            br_regrs[tag_ind, :] = [fxbr - xbr, fybr - ybr]
            tl_tags[tag_ind] = ytl * self.output_size[1] + xtl
            #print('TL_TAGS[tag_ing]: {}'.format(tl_tags[tag_ind]))
            br_tags[tag_ind] = ybr * self.output_size[1] + xbr
            tag_lens[0] += 1
        
        tag_len = tag_lens[0]
        #print("tag_len:{} ".format(tag_len))
        tag_masks[:tag_len] = 1
        #print("tag_masks_shape:{}, tag_masks:{} ".format(tag_masks.shape, tag_masks))

        images      = torch.from_numpy(image / 255.0).float().to(device = self.device).permute(2, 0, 1)
        tl_heatmaps = torch.from_numpy(tl_heatmaps).to(device = self.device)
        br_heatmaps = torch.from_numpy(br_heatmaps).to(device = self.device)
        tl_regrs    = torch.from_numpy(tl_regrs).to(device = self.device)
        br_regrs    = torch.from_numpy(br_regrs).to(device = self.device)
        tl_tags     = torch.from_numpy(tl_tags).to(device = self.device)
        br_tags     = torch.from_numpy(br_tags).to(device = self.device)
        tag_masks   = torch.from_numpy(tag_masks).to(device = self.device)

        #print('GT_TL_HEAT_IN_DATASET:\n{}'.format(tl_heatmaps.sum()))
        #print('GT_BR_HEAT_IN_DATASET:\n{}'.format(br_heatmaps.sum()))

        return images, tl_tags, br_tags, tl_heatmaps, br_heatmaps, tag_masks, tl_regrs, br_regrs

    def __len__(self): 
      return len(self.image_names)



if __name__ == "__main__":

    n = 5
    dims    = [256, 256, 384, 384, 384, 512]
    modules = [2, 2, 2, 2, 2, 4]
    out_dim = 10

    model = kp(n = n, nstack = 2, dims = dims, modules = modules, out_dim = out_dim).eval().cpu()

    '''
    model_dict = model.state_dict()
    print('Model_dict_len: ' + str(len(model_dict)))
    #print(model_dict)

    print('MODEL DICT DONE !!')
    
    #CHECKPOINT_PATH = 'C://Users//Tony Stark//Desktop//Önálló laboratórium//Code//CornerNet//pretrained_cornernet.pkl'
    CHECKPOINT_PATH = '/content/drive/My Drive/CornerNet/CornerNet_500000.pkl'
    pretrained_dict = torch.load(CHECKPOINT_PATH)
    print('PRETRAINED DICT DONE !!')

    prefix = 'module.'
    n_clip = len(prefix)
    adapted_dict = {k[n_clip:]: v for k, v in pretrained_dict.items()
                if (k.startswith(prefix + 'pre') or k.startswith(prefix + 'kps') or k.startswith(prefix + 'inter') or k.startswith(prefix + 'cnv'))}
    print('ADAPTED_dict_len: ' + str(len(adapted_dict)))
    '''

    '''
    for k, v in adapted_dict.items():
        print(k)
    '''

    # Checking adapted_dict and pretrained_dict equality
    #for k, v in zip(pretrained_dict.items(), adapted_dict.items()):
    #    if(torch.equal(k[1], v[1])):
    #        print('EQUAL!')

    '''
    for k, v in zip(model_dict.items(), adapted_dict.items()):
        if k in model_dict:
            pretrained_dict = {k: v}
            print(k)
        print(k[0], v[0])
        j = v[0].split('.', 1)[1]
        v[0] = j
        print(k[0], v[0])
    '''
    '''
    #pretrained_dict = {k: v for k, v in adapted_dict.items() if k in model_dict}
    #model_dict.update(adapted_dict)
    model.load_state_dict(adapted_dict, strict = False)

    model.cuda()

    model_dict_2 = model.state_dict()
    print('Model_dict_2_len: ' + str(len(model_dict_2)))

    count = 0
    for name, param in model_dict_2.items():
        if name not in adapted_dict:
            continue
        #print(model_dict_2[name])
        if(torch.equal(adapted_dict[name], param)):
            print(name)
            count += 1
    print('LOADED PARAMS COUNT: ' + str(count))
    '''

    '''
    for k, v in zip(model_dict_2.items(), adapted_dict.items()):
        if(torch.equal(k[1], v[1])):
            print('EQUAL!')
    '''

    CHECKPOINT_PATH = '/content/drive/My Drive/CornerNet/ModelParams/pretrained_cornernet.pth'
    pretrained_dict = torch.load(CHECKPOINT_PATH)
    print('PRETRAINED DICT DONE !!')

    model.load_state_dict(pretrained_dict['model_state_dict'])#, strict = False)
    model.cuda()

    index = 32234

    dataset = Dataset(mode = 'Val')

    dataloader = DataLoader(dataset = dataset, batch_size = 2 ,shuffle = True)

    loader_iter = iter(dataloader)

    images, tl_tags, br_tags, tl_heatmaps, br_heatmaps, tag_masks, tl_regrs, br_regrs = next(loader_iter)

    xs = [images, tl_tags, br_tags]
    ys = [tl_heatmaps, br_heatmaps, tag_masks, tl_regrs, br_regrs]

    print('------- DATASET OUTPUT SHAPES -------')
    print(images.shape)
    print(tl_tags.shape)
    print(br_tags.shape)
    print(tl_heatmaps.shape)
    print(br_heatmaps.shape)
    print(tag_masks.shape)
    print(tl_regrs.shape)
    print(br_regrs.shape)


    outs = model(*xs)
    
    tl_heat = outs[0]
    br_heat = outs[1]
    tl_tag = outs[2]
    br_tag = outs[3]
    tl_regr = outs[4]
    br_regr = outs[5]

    print('------- CORNERNET MODEL OUTPUT SHAPES -------')
    print(tl_heat.shape)
    print(br_heat.shape)
    print(tl_tag.shape)
    print(br_tag.shape)
    print(tl_regr.shape)
    print(br_regr.shape)

    loss_model = AELoss(pull_weight = 1e-1, push_weight = 1e-1)

    loss = loss_model(outs, ys)

    print('-----------------LOSS DONE--------------------')
    print('loss_shape:{}, loss:{} '.format(loss.shape, loss))
    '''
    PATH = '/content/drive/My Drive/CornerNet/ModelParams/pretrained_cornernet.pth'                        
    torch.save({
                #'epoch': current_epoch,
                #'iter': current_train_iter,
                'model_state_dict': model.state_dict()#,
                #'optimizer_state_dict': optimizer.state_dict(),
                #'train_loss': train_loss,
                #'val_loss': current_average_val_loss
                }, PATH)

    print('!! SAVE DONE !!')
    
    
    #tl_heat, br_heat, tl_tag, br_tag, tl_regr, br_regr
    #images, tl_tags, br_tags, tl_heatmaps, br_heatmaps, tag_masks, tl_regrs, br_regrs = dataset.__getitem__(index)
    '''




    '''
    image_names, detections = get_annotations('Train')
    index = 32234
    name = image_names[index]

    print(detections[name])

    image = get_image('Train', name)

    print(len(image_names))
    print(image_names[index])
    print(len(detections))

    #index = 1575
    #val_image_paths = glob.glob("C://Users//Tony Stark//Desktop//Önálló laboratórium//bdd100k//images//100k//val//*.jpg")
    train_image_paths = glob.glob("C://Users//Tony Stark//Desktop//Önálló laboratórium//bdd100k//images//100k//train//*.jpg")
    #name = (val_image_paths[index].rsplit('.', 1)[0]).rsplit('\\', 1)[1]
    #name = val_image_paths[index].rsplit('\\', 1)[1]
    
   
        
 
        #image = io.imread(image_path)
        
    test = plt.figure('test_image')
        #image_numpy = image.permute(1,2,0).numpy()/255.0
    plt.imshow(image)
    
    plt.show()
    '''

        #print(len(detections))
        #print(detections[name])
        #print(detections[name][0])
        #print(detections[name][0][2])
    '''

 tensor_obs = torch.tensor((obs / 255.0), device = self.device).permute(2,0,1).float()
      tensor_next_obs = torch.tensor((next_obs / 255.0), device = self.device).permute(2,0,1).float()
      tensor_rew=torch.tensor(rew, device = self.device)
      tensor_act=torch.tensor(act, device = self.device)
      tensor_term=torch.tensor(term, device = self.device)

      return tensor_obs, tensor_rew, tensor_act, tensor_term, tensor_next_obs
'''

'''
# Tanító adathalmaz letöltése
export trainimagesid=1MymWKQUFCENauQP8A6QRk1EglinAKaud
export trainimagesfilename=train.zip
wget --save-cookies trainimagescookies.txt 'https://docs.google.com/uc?export=download&id='$trainimagesid -O- \
     | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1/p' > trainimagesconfirm.txt
wget --load-cookies trainimagescookies.txt -O $trainimagesfilename \
     'https://docs.google.com/uc?export=download&id='$trainimagesid'&confirm='$(<trainimagesconfirm.txt)


# Validációs adathalmaz letöltése
export valimagesid=1zgutvylvwv4CFz7rzlPFsL5mrTFHuqFG
export valimagesfilename=val.zip
wget --save-cookies valimagescookies.txt 'https://docs.google.com/uc?export=download&id='$valimagesid -O- \
     | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1/p' > valimagesconfirm.txt
wget --load-cookies valimagescookies.txt -O $valimagesfilename \
     'https://docs.google.com/uc?export=download&id='$valimagesid'&confirm='$(<valimagesconfirm.txt)


# Tanító annotáció letöltése
export trainannotationid=1JLkStcXlhVzvB7Fns-c2Wy_94j8NH75R
export trainannotationfilename=bdd100k_labels_images_train.json
wget --save-cookies trainannotationcookies.txt 'https://docs.google.com/uc?export=download&id='$trainannotationid -O- \
     | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1/p' > trainannotationconfirm.txt
wget --load-cookies trainannotationcookies.txt -O $trainannotationfilename \
     'https://docs.google.com/uc?export=download&id='$trainannotationid'&confirm='$(<trainannotationconfirm.txt)


# Validációs annotáció letöltése
export valannotationid=1fj9Sg4v4TwSvD2nxs90uzNqZgVythxLS
export valannotationfilename=bdd100k_labels_images_val.json
wget --save-cookies valannotationcookies.txt 'https://docs.google.com/uc?export=download&id='$valannotationid -O- \
     | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1/p' > valannotationconfirm.txt
wget --load-cookies valannotationcookies.txt -O $valannotationfilename \
     'https://docs.google.com/uc?export=download&id='$valannotationid'&confirm='$(<valannotationconfirm.txt)


# Best ModelParams letöltése
export modelparamsid=13tYTwt-1PL8e-tCBN8QBC2vCNmEgbkmu
export modelparamsfilename=train_valid_pretrained_cornernet-epoch3-iter5067.pth
wget --save-cookies modelparamscookies.txt 'https://docs.google.com/uc?export=download&id='$modelparamsid -O- \
     | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1/p' > modelparamsconfirm.txt
wget --load-cookies modelparamscookies.txt -O $modelparamsfilename \
     'https://docs.google.com/uc?export=download&id='$modelparamsid'&confirm='$(<modelparamsconfirm.txt)
'''

'''
GT tenzorokat vizualizálni és megnézni, hogy jó e a vizualizáció 
freezelni az eredeti súlyokat és ugy tanulni ==> nézni epochrol epochra a vizualizaciot
8-adolás és kivonni belőle az egész részét
'''