import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import numpy as np
import time
import glob
import warnings
from Dataset import Dataset
from CornerNet import kp as cornernet
from losses import AELoss
from test import generate_detections, save_detections, evaluate_detections, make_grid_image

warnings.filterwarnings(action = 'once')

def train(batch_size = 14, epochs = 100):
    since = time.time()

    # Hyperparams of the model
    n                               = 5
    nstack                          = 2
    dims                            = [256, 256, 384, 384, 384, 512]
    modules                         = [2, 2, 2, 2, 2, 4]
    out_dim                         = 10

    # Hyperparams of the optimizer
    base_lr_rate                    = 0.00025
    weight_decay                    = 0.000016

    # Hyperparams for creating bounding boxes
    top_k                           = 100
    ae_threshold                    = 0.5
    nms_threshold                   = 0.5       # Non Max Suppression IoU threshold
    nms_kernel                      = 3
    min_score                       = 0.3       # Keep bbox-es (before mAP calculation and pred annot image generation) only with score > min_score

    # Hyperparams for calculating mAP
    mean_ap_epoch_interval          = 3
    mean_ap_thresholds              = [0.5, 0.75]                  

    # Starting params for the training
    starting_epoch                  = 0
    starting_iter                   = 0

    best_epoch_average_train_loss   = 10000.0   # Updated only if best_epoch_average_val_loss is updated as well
    best_epoch_average_val_loss     = 10000.0
    best_mAP_50                     = 0.0
    best_mAP_75                     = 0.0
    best_breakdown_50               = []
    best_breakdown_75               = []

    last_epoch_average_train_loss   = 10000.0       
    last_epoch_average_val_loss     = 10000.0    
    last_mAP_50                     = 0.0
    last_mAP_75                     = 0.0
    last_breakdown_50               = []
    last_breakdown_75               = []

    cat_list                        = []
    
    # Tensorboard SummaryWriters for training logs
    writer_text                     = SummaryWriter('../Tensorboard/cornernet_hourglass_training_text/')
    writer_avg_train_loss           = SummaryWriter('../Tensorboard/cornernet_hourglass_training_avg_train_loss_per_epoch/')
    writer_avg_valid_loss           = SummaryWriter('../Tensorboard/cornernet_hourglass_training_avg_valid_loss_per_epoch/')
    writer_map_50                   = SummaryWriter('../Tensorboard/cornernet_hourglass_training_mean_ap_50/')
    writer_map_75                   = SummaryWriter('../Tensorboard/cornernet_hourglass_training_mean_ap_75/')
    writer_cat_ap_50                = SummaryWriter('../Tensorboard/cornernet_hourglass_training_categories_ap_50/')
    writer_cat_ap_75                = SummaryWriter('../Tensorboard/cornernet_hourglass_training_categories_ap_75/')
    writer_image                    = SummaryWriter('../Tensorboard/cornernet_hourglass_training_image/')
    writer_hparams                  = SummaryWriter('../Tensorboard/cornernet_hourglass_training_hparams/')
    '''
    # Loading the best checkpoint
    checkpoint_path                 = '../ModelParams/Hourglass/cornernet_hourglass_pretrained-epoch{}.pth'.format(11)
    checkpoint                      = torch.load(checkpoint_path)

    starting_epoch                  = checkpoint['epoch'] + 1
    starting_iter                   = 0#checkpoint['iter'] + 1
    base_lr_rate                    = checkpoint['lr']
    weight_decay                    = checkpoint['weight_decay']
    best_epoch_average_train_loss   = checkpoint['best_train_loss']
    best_epoch_average_val_loss     = checkpoint['best_val_loss']
    best_mAP_50                     = checkpoint['best_map_50']
    best_mAP_75                     = checkpoint['best_map_75']
    best_breakdown_50               = checkpoint['best_breakdown_50']
    best_breakdown_75               = checkpoint['best_breakdown_75']
    last_epoch_average_train_loss   = checkpoint['last_train_loss']
    last_epoch_average_val_loss     = checkpoint['last_val_loss']
    last_mAP_50                     = checkpoint['last_map_50']
    last_mAP_75                     = checkpoint['last_map_75']
    last_breakdown_50               = checkpoint['last_breakdown_50']
    last_breakdown_75               = checkpoint['last_breakdown_75']
    '''
    # Train dataset and train dataloader
    train_dataset                   = Dataset(mode = 'Train')
    train_loader                    = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
    
    # Val dataset and val dataloader
    val_dataset                     = Dataset(mode = 'Val')
    val_loader                      = DataLoader(val_dataset, batch_size = batch_size, shuffle = True)

    # mAP dataset and mAP dataloader
    mAP_dataset                     = Dataset(mode = 'Val')    
    mAP_loader                      = DataLoader(mAP_dataset, batch_size = 1, shuffle = False)
   
    model                           = cornernet(n = n, nstack = nstack, dims = dims, modules = modules, out_dim = out_dim).cuda()

    
    # Loading the original pretrained (on MSCOCO) weights in the first epoch
    checkpoint_path                 = '../ModelParams/Hourglass/CornerNet_500000.pkl'
    pretrained_dict                 = torch.load(checkpoint_path)
    prefix                          = 'module.'
    n_clip                          = len(prefix)
    adapted_dict                    = {k[n_clip:]: v for k, v in pretrained_dict.items()
                                        if (k.startswith(prefix + 'pre') or k.startswith(prefix + 'kps') or k.startswith(prefix + 'inter') or k.startswith(prefix + 'cnv'))}
    
    model.load_state_dict(adapted_dict, strict = False)

    model_dict                      = model.state_dict()
    count                           = 0
    for name, param in adapted_dict.items():
        if name not in model_dict:
            continue
        if(torch.equal(model_dict[name], param)):
            #print(name)
            count += 1
    
    print('NUMBER OF LAYERS THAT ARE LOADED FROM THE ORIGINAL PRETRAINED MODEL: {}'.format(count))
    

    criterion                       = AELoss(pull_weight = 1e-1, push_weight = 1e-1)
    optimizer                       = optim.Adam(model.parameters(), lr = base_lr_rate, weight_decay = weight_decay, amsgrad = True)

    # Disable these when loading the original pretrained model parameters
    #model.load_state_dict(checkpoint['model_state_dict'])
    #optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    writer_text.add_text(tag = 'Hourglass/StartingLogs',                                                                                                  \
                    text_string = (   'LOADED MODEL PARAMETERS: {}  \n'.format(checkpoint_path)                                                           \
                                    + 'BATCH SIZE: {}  \n'.format(batch_size)                                                                             \
                                    + 'NUMBER OF EPOCHS: {}  \n'.format(epochs)                                                                           \
                                    + 'STARTING EPOCH: {}  \n'.format(starting_epoch)                                                                     \
                                    + 'TRAINING ITERATIONS / EPOCH: {}  \n'.format(len(train_loader))                                                     \
                                    + 'VALIDATION ITERATIONS / EPOCH: {}  \n'.format(len(val_loader))                                                     \
                                    + 'LEARNING RATE: {}  \n'.format(base_lr_rate)                                                                        \
                                    + 'WEIGHT DECAY (L2 REGULARIZATION): {}  \n'.format(weight_decay)                                                     \
                                    + 'MEAN AP CALCULATION EPOCH INTERVAL: {}  \n'.format(mean_ap_epoch_interval)                                         \
                                    + 'MEAN AP IOU THRESHOLDS: [{}%, {}%]  \n'.format(int(mean_ap_thresholds[0] * 100), int(mean_ap_thresholds[1] * 100)) \
                                    + '[BEST/LAST] AVERAGE TRAINING LOSS: [{:5f} / {:5f}]  \n'.format(
                                            best_epoch_average_train_loss, last_epoch_average_train_loss)                                                 \
                                    + '[BEST/LAST] AVERAGE VALIDATION LOSS: [{:5f} / {:5f}]  \n'.format(
                                            best_epoch_average_val_loss, last_epoch_average_val_loss)                                                     \
                                    + '[BEST/LAST] MEAN AVERAGE PRECISION (IOU = {}%): [{:2f} / {:2f}]  \n'.format(
                                            int(mean_ap_thresholds[0] * 100), best_mAP_50, last_mAP_50)                                                   \
                                    + '[BEST/LAST] MEAN AVERAGE PRECISION (IOU = {}%): [{:2f} / {:2f}]  \n'.format(
                                            int(mean_ap_thresholds[1] * 100), best_mAP_75, last_mAP_75)),                                                 \
                    global_step = None, walltime = None) 

    # Start the training
    for current_epoch in range(starting_epoch, epochs):
        writer_text.add_text(tag = 'Hourglass/RunningLogs', text_string = 'EPOCH: {}/{}'.format((current_epoch + 1), epochs), global_step = (current_epoch + 1), walltime = None)
               
        epoch_since                 = time.time()

        writer_epoch                = SummaryWriter('../Tensorboard/cornernet_hourglass_training_avg_loss_per_iteration_epoch_{}/'.format(current_epoch + 1))

        detections_json             = []
        pred_detections_images      = []
        gt_detections_images        = []

        current_train_iter          = 0
        current_val_iter            = 0
        current_mAP_iter            = 0
        
        running_train_loss          = 0.0
        current_average_train_loss  = 0.0
        running_val_loss            = 0.0
        current_average_val_loss    = 0.0

        current_mAP_50              = 0.0
        current_mAP_75              = 0.0
        current_breakdown_50        = []
        current_breakdown_75        = []

        is_saved                    = False

        for phase in ['train', 'val']:

            # Train loop
            if phase == 'train':
                train_epoch_since = time.time()
                
                model.train()

                for train_data in train_loader:
                    
                    current_train_iter += 1

                    images, tl_tags, br_tags, tl_heatmaps, br_heatmaps, tag_masks, tl_regrs, br_regrs, names, orig_images = train_data

                    xs = [images, tl_tags, br_tags]
                    ys = [tl_heatmaps, br_heatmaps, tag_masks, tl_regrs, br_regrs]                  
            
                    outs = model(*xs, mode = 'Train')          
            
                    #scheduler = poly_lr_scheduler(optimizer = optimizer, init_lr = base_lr_rate, iter = current_iter, lr_decay_iter = 1, 
                    #                          max_iter = max_iter, power = power)                                                          # max_iter = len(train_loader)
            
                    optimizer.zero_grad()
            
                    loss = criterion(outs, ys)

                    running_train_loss += loss.item()
                    current_average_train_loss = running_train_loss / current_train_iter

                    writer_epoch.add_scalar(tag = 'Hourglass/TrainingIterationAverageLoss'.format(current_epoch + 1), scalar_value = current_average_train_loss, global_step = current_train_iter)
                    
                    loss.backward(retain_graph = False)
            
                    optimizer.step()
                
                last_epoch_average_train_loss = current_average_train_loss

                writer_avg_train_loss.add_scalar(tag = 'Hourglass/Overfitting', scalar_value = last_epoch_average_train_loss, global_step = (current_epoch + 1))

                train_time_elapsed = time.time() - train_epoch_since
            
            # Validation loop
            elif phase == 'val':
                val_epoch_since = time.time()   
                
                model.eval()
                
                with torch.no_grad():
                    for val_data in val_loader:
                        
                        current_val_iter += 1
                        
                        images, tl_tags, br_tags, tl_heatmaps, br_heatmaps, tag_masks, tl_regrs, br_regrs, names, orig_images = val_data

                        xs = [images, tl_tags, br_tags]
                        ys = [tl_heatmaps, br_heatmaps, tag_masks, tl_regrs, br_regrs]                    
                        
                        outs = model(*xs, mode = 'Val')
                        
                        val_loss = criterion(outs, ys)

                        running_val_loss += val_loss.item()
                        current_average_val_loss = running_val_loss / current_val_iter

                        writer_epoch.add_scalar(tag = 'Hourglass/ValidationIterationAverageLoss'.format((current_epoch + 1)), scalar_value = current_average_val_loss, global_step = current_val_iter)
                    
                    last_epoch_average_val_loss = current_average_val_loss

                    writer_avg_valid_loss.add_scalar(tag = 'Hourglass/Overfitting', scalar_value = last_epoch_average_val_loss, global_step = (current_epoch + 1))

                    val_time_elapsed = time.time() - val_epoch_since                                      
        
        # Calculating mAP in every (mean_ap_epoch_interval) epoch
        if ((current_epoch % mean_ap_epoch_interval) == 0):
            mAP_epoch_since = time.time()

            annot_image_saving_iterations = np.random.randint(low = 1, high = (len(mAP_loader) - 1), size = 4)
            annot_image_saving_iterations = [1, 2, 3, 4]

            model.eval()

            with torch.no_grad():
                for mAP_data in mAP_loader:
                    mAP_batch_since = time.time()
                    
                    current_mAP_iter += 1

                    images, tl_tags, br_tags, tl_heatmaps, br_heatmaps, tag_masks, tl_regrs, br_regrs, names, orig_images = mAP_data

                    detections = model(images, mode = 'mAP', ae_threshold = ae_threshold, top_k = top_k, kernel = nms_kernel)

                    generate_detections(detections = detections, names = names, orig_images = orig_images,              \
                        detections_json = detections_json, nms_threshold = nms_threshold, min_score = min_score,        \
                        current_iter = current_mAP_iter, annot_image_saving_iterations = annot_image_saving_iterations, \
                        pred_detections_images = pred_detections_images, gt_detections_images = gt_detections_images)
            
                save_detections(detections_json, (current_epoch + 1))

                grid_image = make_grid_image(gt_detections_images = gt_detections_images, pred_detections_images = pred_detections_images)

                current_mAP_50, current_mAP_75, current_breakdown_50, current_breakdown_75, cat_list = evaluate_detections( current_epoch = (current_epoch + 1),
                                                                                                                            mean_ap_thresholds = mean_ap_thresholds) 

                writer_map_50.add_scalar(tag = 'Hourglass/MEAN_AP', scalar_value = current_mAP_50, global_step = (current_epoch + 1))
                writer_map_75.add_scalar(tag = 'Hourglass/MEAN_AP', scalar_value = current_mAP_75, global_step = (current_epoch + 1))
                writer_cat_ap_50.add_scalars(main_tag = 'Hourglass/CATEGORY_AP_50/', 
                    tag_scalar_dict =  {cat_list[0].upper(): current_breakdown_50[0],
                                        cat_list[1].upper(): current_breakdown_50[1],
                                        cat_list[2].upper(): current_breakdown_50[2],
                                        cat_list[3].upper(): current_breakdown_50[3],
                                        cat_list[4].upper(): current_breakdown_50[4],
                                        cat_list[5].upper(): current_breakdown_50[5],
                                        cat_list[6].upper(): current_breakdown_50[6],
                                        cat_list[7].upper(): current_breakdown_50[7],
                                        cat_list[8].upper(): current_breakdown_50[8],
                                        cat_list[9].upper(): current_breakdown_50[9]},
                    global_step = (current_epoch + 1))
                writer_cat_ap_75.add_scalars(main_tag = 'Hourglass/CATEGORY_AP_75/', 
                    tag_scalar_dict =  {cat_list[0].upper(): current_breakdown_75[0],
                                        cat_list[1].upper(): current_breakdown_75[1],
                                        cat_list[2].upper(): current_breakdown_75[2],
                                        cat_list[3].upper(): current_breakdown_75[3],
                                        cat_list[4].upper(): current_breakdown_75[4],
                                        cat_list[5].upper(): current_breakdown_75[5],
                                        cat_list[6].upper(): current_breakdown_75[6],
                                        cat_list[7].upper(): current_breakdown_75[7],
                                        cat_list[8].upper(): current_breakdown_75[8],
                                        cat_list[9].upper(): current_breakdown_75[9]},
                    global_step = (current_epoch + 1))
                
                last_mAP_50 = current_mAP_50
                last_mAP_75 = current_mAP_75
                last_breakdown_50 = current_breakdown_50
                last_breakdown_75 = current_breakdown_75                 

                writer_image.add_images(tag = 'Hourglass/PREDICTIONS/EPOCH_{}'.format(current_epoch + 1), img_tensor = grid_image, global_step = 0, dataformats='CHW') 

            mAP_time_elapsed = time.time() - mAP_epoch_since

        # Saving model parameters if average_val_loss or mAP is improved
        if((last_epoch_average_val_loss < best_epoch_average_val_loss) or (last_mAP_50 > best_mAP_50) or (last_mAP_75 > best_mAP_75)):
            is_saved = True

            if(last_epoch_average_val_loss < best_epoch_average_val_loss):
                best_epoch_average_val_loss   = last_epoch_average_val_loss
                best_epoch_average_train_loss = last_epoch_average_train_loss

            if(last_mAP_50 > best_mAP_50):
                best_mAP_50             = last_mAP_50
                best_breakdown_50       = last_breakdown_50             

            if(last_mAP_75 > best_mAP_75):
                best_mAP_75             = last_mAP_75
                best_breakdown_75       = last_breakdown_75   

            PATH = '../ModelParams/Hourglass/cornernet_hourglass_pretrained-epoch{}.pth'.format(current_epoch + 1)                        
            torch.save({
                    'epoch': current_epoch,
                    'iter': current_train_iter,
                    'lr': base_lr_rate,
                    'weight_decay': weight_decay,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_train_loss': best_epoch_average_train_loss,
                    'best_val_loss': best_epoch_average_val_loss,
                    'best_map_50': best_mAP_50,
                    'best_map_75': best_mAP_75,
                    'best_breakdown_50': best_breakdown_50,
                    'best_breakdown_75': best_breakdown_75,
                    'last_train_loss': last_epoch_average_train_loss,
                    'last_val_loss': last_epoch_average_val_loss,
                    'last_map_50': last_mAP_50,
                    'last_map_75': last_mAP_75,
                    'last_breakdown_50': last_breakdown_50,
                    'last_breakdown_75': last_breakdown_75
                    }, PATH)
            
            writer_text.add_text(tag = 'SavingLogs',                                                                           \
                        text_string = (   '!!! IMPROVEMENT !!! MODEL PARAMETERS HAVE BEEN SAVED !!!  \n'                       \
                                        + '[BEST/LAST] AVERAGE TRAINING LOSS: [{:5f} / {:5f}]  \n'.format(
                                                best_epoch_average_train_loss, last_epoch_average_train_loss)                  \
                                        + '[BEST/LAST] AVERAGE VALIDATION LOSS: [{:5f} / {:5f}]  \n'.format(
                                                best_epoch_average_val_loss, last_epoch_average_val_loss)                      \
                                        + '[BEST/LAST] MEAN AVERAGE PRECISION (IOU = {}%): [{:2f} / {:2f}]  \n'.format(
                                                int(mean_ap_thresholds[0] * 100), best_mAP_50, last_mAP_50)                    \
                                        + '[BEST/LAST] MEAN AVERAGE PRECISION (IOU = {}%): [{:2f} / {:2f}]  \n'.format(
                                                int(mean_ap_thresholds[1] * 100), best_mAP_75, last_mAP_75)),                  \
                        global_step = (current_epoch + 1), walltime = None)

        else:
             writer_text.add_text(tag = 'SavingLogs',                                                                          \
                        text_string = (   '!!! NO IMPROVEMENT !!!  \n'                                                         \
                                        + '[BEST/LAST] AVERAGE TRAINING LOSS: [{:5f} / {:5f}]  \n'.format(
                                                best_epoch_average_train_loss, last_epoch_average_train_loss)                  \
                                        + '[BEST/LAST] AVERAGE VALIDATION LOSS: [{:5f} / {:5f}]  \n'.format(
                                                best_epoch_average_val_loss, last_epoch_average_val_loss)                      \
                                        + '[BEST/LAST] MEAN AVERAGE PRECISION (IOU = {}%): [{:2f} / {:2f}]  \n'.format(
                                                int(mean_ap_thresholds[0] * 100), best_mAP_50, last_mAP_50)                    \
                                        + '[BEST/LAST] MEAN AVERAGE PRECISION (IOU = {}%): [{:2f} / {:2f}]  \n'.format(
                                                int(mean_ap_thresholds[1] * 100), best_mAP_75, last_mAP_75)),                  \
                        global_step = (current_epoch + 1), walltime = None)
        
        writer_hparams.add_hparams(hparam_dict = {'EPOCH': str(current_epoch + 1),
                                                  'BATCH SIZE': str(batch_size),
                                                  'OPTIMIZER': 'ADAM (AMSGRAD)',
                                                  'LEARNING RATE': str(base_lr_rate),
                                                  'WEIGHT DECAY': str(weight_decay),
                                                  'TRAIN LOSS': '{:5f}'.format(last_epoch_average_train_loss),
                                                  'VAL LOSS': '{:5f}'.format(last_epoch_average_val_loss),
                                                  'MEAN AP [AP50 / AP75]': '{:2f} / {:2f}'.format(last_mAP_50, last_mAP_75),
                                                  cat_list[0].upper() + ' [AP50 / AP75]': '{:2f} / {:2f}'.format(last_breakdown_50[0], last_breakdown_75[0]),
                                                  cat_list[1].upper() + ' [AP50 / PA75]': '{:2f} / {:2f}'.format(last_breakdown_50[1], last_breakdown_75[1]),
                                                  cat_list[2].upper() + ' [AP50 / AP75]': '{:2f} / {:2f}'.format(last_breakdown_50[2], last_breakdown_75[2]),
                                                  cat_list[3].upper() + ' [AP50 / AP75]': '{:2f} / {:2f}'.format(last_breakdown_50[3], last_breakdown_75[3]),
                                                  cat_list[4].upper() + ' [AP50 / AP75]': '{:2f} / {:2f}'.format(last_breakdown_50[4], last_breakdown_75[4]),
                                                  cat_list[5].upper() + ' [AP50 / AP75]': '{:2f} / {:2f}'.format(last_breakdown_50[5], last_breakdown_75[5]),
                                                  cat_list[6].upper() + ' [AP50 / AP75]': '{:2f} / {:2f}'.format(last_breakdown_50[6], last_breakdown_75[6]),
                                                  cat_list[7].upper() + ' [AP50 / AP75]': '{:2f} / {:2f}'.format(last_breakdown_50[7], last_breakdown_75[7]),
                                                  cat_list[8].upper() + ' [AP50 / AP75]': '{:2f} / {:2f}'.format(last_breakdown_50[8], last_breakdown_75[8]),
                                                  cat_list[9].upper() + ' [AP50 / AP75]': '{:2f} / {:2f}'.format(last_breakdown_50[9], last_breakdown_75[9]),
                                                  'SAVED': str(is_saved)},
                                    metric_dict = {'Hourglass/VV_LEARNING_RATE': base_lr_rate})
        
        epoch_time_elapsed = time.time() - epoch_since

        # Epoch ending logs
        writer_text.add_text(tag = 'Hourglass/RunningLogs',                                                                    \
                        text_string = (   '[BEST/LAST] AVERAGE TRAINING LOSS: [{:5f} / {:5f}]  \n'.format(
                                                best_epoch_average_train_loss, last_epoch_average_train_loss)                  \
                                        + '[BEST/LAST] AVERAGE VALIDATION LOSS: [{:5f} / {:5f}]  \n'.format(
                                                best_epoch_average_val_loss, last_epoch_average_val_loss)                      \
                                        + '[BEST/LAST] MEAN AVERAGE PRECISION (IOU = {}%): [{:5f} / {:5f}]  \n'.format(
                                                int(mean_ap_thresholds[0] * 100), best_mAP_50, last_mAP_50)                    \
                                        + '[BEST/LAST] MEAN AVERAGE PRECISION (IOU = {}%): [{:5f} / {:5f}]  \n'.format(
                                                int(mean_ap_thresholds[1] * 100), best_mAP_75, last_mAP_75)                    \
                                        + 'EPOCH TIME: {}:{}  \n'.format(
                                                int(epoch_time_elapsed // 60 // 60), int(epoch_time_elapsed // 60 % 60))       \
                                        + 'TRAINING TIME: {}:{}  \n'.format(
                                                int(train_time_elapsed // 60 // 60), int(train_time_elapsed // 60 % 60))       \
                                        + 'VALIDATION TIME: {}:{}  \n'.format(
                                                int(val_time_elapsed // 60 // 60), int(val_time_elapsed // 60 % 60))           \
                                        + 'MEAN AVERAGE PRECISION TIME: {}:{}  \n'.format(
                                                int(mAP_time_elapsed // 60 // 60), int(mAP_time_elapsed // 60 % 60))),         \
                        global_step = (current_epoch + 1), walltime = None)

if __name__ == "__main__":
    train()