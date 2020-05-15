import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import time
import warnings
from dataset import Dataset
from cornernet import kp as cornernet
from losses import AELoss
from test import generate_detections, save_detections, evaluate_detections, make_grid_image
from train_utils import tensorboard_add_hparams, tensorboard_epoch_ending_logs, tensorboard_saving_logs, saving_model_params, tensorboard_add_scalars, tensorboard_starting_logs, loading_original_model_params

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

    optimizer_name                  = "ADAM (AMSGRAD)"

    server_model_params_path        = '../../../logs/cornernet/ModelParams/hourglass/cornernet_hourglass_pretrained_best.pth'
    
    # Tensorboard SummaryWriters for training logs
    writer_text                     = SummaryWriter('../../../logs/cornernet/Tensorboard/hourglass/cornernet_hourglass_training_text/')
    writer_avg_train_loss           = SummaryWriter('../../../logs/cornernet/Tensorboard/hourglass/cornernet_hourglass_training_avg_train_loss_per_epoch/')
    writer_avg_valid_loss           = SummaryWriter('../../../logs/cornernet/Tensorboard/hourglass/cornernet_hourglass_training_avg_valid_loss_per_epoch/')
    writer_map_50                   = SummaryWriter('../../../logs/cornernet/Tensorboard/hourglass/cornernet_hourglass_training_mean_ap_50/')
    writer_map_75                   = SummaryWriter('../../../logs/cornernet/Tensorboard/hourglass/cornernet_hourglass_training_mean_ap_75/')
    writer_cat_ap_50                = SummaryWriter('../../../logs/cornernet/Tensorboard/hourglass/cornernet_hourglass_training_categories_ap_50/')
    writer_cat_ap_75                = SummaryWriter('../../../logs/cornernet/Tensorboard/hourglass/cornernet_hourglass_training_categories_ap_75/')
    writer_image                    = SummaryWriter('../../../logs/cornernet/Tensorboard/hourglass/cornernet_hourglass_training_image/')
    writer_hparams                  = SummaryWriter('../../../logs/cornernet/Tensorboard/hourglass/cornernet_hourglass_training_hparams/')
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
    checkpoint_path                 = '../../../logs/cornernet/ModelParams/hourglass/CornerNet_500000.pkl'
    model                           = loading_original_model_params(checkpoint_path, model)

    criterion                       = AELoss(pull_weight = 1e-1, push_weight = 1e-1)
    optimizer                       = optim.Adam(model.parameters(), lr = base_lr_rate, weight_decay = weight_decay, amsgrad = True)

    # Disable these when loading the original pretrained model parameters
    #model.load_state_dict(checkpoint['model_state_dict'])
    #optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if torch.cuda.device_count() > 1:
        print("LET'S USE", torch.cuda.device_count(), "GPUS!")
        model = nn.DataParallel(model)
    
    tensorboard_starting_logs(writer_text, checkpoint_path, batch_size, epochs, starting_epoch, train_loader, val_loader, base_lr_rate, weight_decay, mean_ap_epoch_interval, mean_ap_thresholds,
                                best_epoch_average_train_loss, last_epoch_average_train_loss, best_epoch_average_val_loss, last_epoch_average_val_loss, best_mAP_50, last_mAP_50,
                                best_mAP_75, last_mAP_75)

    # Start the training
    for current_epoch in range(starting_epoch, epochs):
        writer_text.add_text(tag = 'Hourglass/RunningLogs', text_string = 'EPOCH: {}/{}'.format((current_epoch + 1), epochs), global_step = (current_epoch + 1), walltime = None)
               
        epoch_since                 = time.time()

        writer_epoch                = SummaryWriter('../../../logs/cornernet/Tensorboard/hourglass/cornernet_hourglass_training_avg_loss_per_iteration_epoch_{}/'.format(current_epoch + 1))

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
            #annot_image_saving_iterations = [1, 2, 3, 4]
            
            if torch.cuda.device_count() > 1:
                module = model.module
            else:
                module = model
            
            module.eval()

            with torch.no_grad():
                for mAP_data in mAP_loader:
                    mAP_batch_since = time.time()
                    
                    current_mAP_iter += 1
                    
                    images, tl_tags, br_tags, tl_heatmaps, br_heatmaps, tag_masks, tl_regrs, br_regrs, names, orig_images = mAP_data
                    
                    detections = module(images, mode = 'mAP', ae_threshold = ae_threshold, top_k = top_k, kernel = nms_kernel)

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

                tensorboard_add_scalars(writer_cat_ap_50, 'Hourglass/CATEGORY_AP_50/', cat_list, current_breakdown_50, current_epoch)
                tensorboard_add_scalars(writer_cat_ap_75, 'Hourglass/CATEGORY_AP_75/', cat_list, current_breakdown_75, current_epoch)
                
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

            PATH = '../ModelParams/hourglass/cornernet_hourglass_pretrained-epoch{}.pth'.format(current_epoch + 1)
            saving_model_params(PATH, current_epoch, current_train_iter, base_lr_rate, weight_decay, model, optimizer, best_epoch_average_train_loss, best_epoch_average_val_loss,
                            best_mAP_50, best_mAP_75, best_breakdown_50, best_breakdown_75, last_epoch_average_train_loss, last_epoch_average_val_loss, last_mAP_50, last_mAP_75,
                            last_breakdown_50, last_breakdown_75)

            PATH = server_model_params_path
            saving_model_params(PATH, current_epoch, current_train_iter, base_lr_rate, weight_decay, model, optimizer, best_epoch_average_train_loss, best_epoch_average_val_loss,
                            best_mAP_50, best_mAP_75, best_breakdown_50, best_breakdown_75, last_epoch_average_train_loss, last_epoch_average_val_loss, last_mAP_50, last_mAP_75,
                            last_breakdown_50, last_breakdown_75)                        
            
            title = '!!! IMPROVEMENT !!! MODEL PARAMETERS HAVE BEEN SAVED !!!'            
            tensorboard_saving_logs(writer_text, title, best_epoch_average_train_loss, last_epoch_average_train_loss, best_epoch_average_val_loss, last_epoch_average_val_loss,
                                        mean_ap_thresholds, best_mAP_50, last_mAP_50, best_mAP_75, last_mAP_75, current_epoch)

        else:
            title = '!!! NO IMPROVEMENT !!!'            
            tensorboard_saving_logs(writer_text, title, best_epoch_average_train_loss, last_epoch_average_train_loss, best_epoch_average_val_loss, last_epoch_average_val_loss,
                                        mean_ap_thresholds, best_mAP_50, last_mAP_50, best_mAP_75, last_mAP_75, current_epoch)
        
        '''
        # SummaryWriter does not have an add_hparams method in torch==1.1.0 (added in 1.3.0)
        tensorboard_add_hparams(writer_hparams, current_epoch, batch_size, optimizer_name, base_lr_rate, weight_decay, last_epoch_average_train_loss, last_epoch_average_val_loss,
                                last_mAP_50, last_mAP_75, cat_list, last_breakdown_50, last_breakdown_75, is_saved)
        '''

        epoch_time_elapsed = time.time() - epoch_since

        # Epoch ending logs
        tensorboard_epoch_ending_logs(writer_text, best_epoch_average_train_loss, last_epoch_average_train_loss, best_epoch_average_val_loss, last_epoch_average_val_loss, mean_ap_thresholds,
                                        best_mAP_50, last_mAP_50, best_mAP_75, last_mAP_75, epoch_time_elapsed, train_time_elapsed, val_time_elapsed, mAP_time_elapsed, current_epoch)

if __name__ == "__main__":
    train(batch_size = 32, epochs = 150)