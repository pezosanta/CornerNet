import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import time
import glob
import warnings
from Dataset import Dataset
from CornerNet import kp as cornernet
from losses import AELoss
from test_v2 import calculate_mAP, save_detections

warnings.filterwarnings(action = 'once')

def train(batch_size = 14, epochs = 40):
    since = time.time()

    # Model Hyperparameters
    n = 5
    nstack = 2
    dims    = [256, 256, 384, 384, 384, 512]
    modules = [2, 2, 2, 2, 2, 4]
    out_dim = 10

    # Hyperparams for creating bounding boxes
    top_k = 100
    ae_threshold = 0.5
    nms_threshold = 0.5
    nms_kernel = 3
 
    base_lr_rate = 0.00025
    weight_decay = 0.000016

    starting_epoch = 0
    starting_iter = 0
    best_average_val_loss = 10000.0
    
    writer = SummaryWriter('logs/hourglass/')
    '''
    ####################### LOAD CHECKPOINTS #######################
    CHECKPOINT_PATH = '../ModelParams/Hourglass/cornernet_hourglass_pretrained-epoch{}.pth'.format(4)
    checkpoint = torch.load(CHECKPOINT_PATH)
    starting_epoch = 0 #checkpoint['epoch']
    starting_iter = 0 #checkpoint['iter'] + 1
    best_average_val_loss = checkpoint['val_loss']
    ################################################################    
    '''
    # Train Dataset and train dataloader
    train_dataset = Dataset(mode = 'Train')
    train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = False)
    
    # Val Dataset and val dataloader
    val_dataset = Dataset(mode = 'Val')
    val_loader = DataLoader(val_dataset, batch_size = batch_size, shuffle = False)

    mAP_dataset = Dataset(mode = 'Val')    
    mAP_loader = DataLoader(mAP_dataset, batch_size = 1, shuffle = False)
    
    train_loader_iter = iter(train_loader)
    val_loader_iter = iter(val_loader)
   
    model = cornernet(n = n, nstack = nstack, dims = dims, modules = modules, out_dim = out_dim).cuda()

    
    # Load the original pretrained MSCOCO weights in the first epoch
    CHECKPOINT_PATH = '../ModelParams/Hourglass/CornerNet_500000.pkl'
    pretrained_dict = torch.load(CHECKPOINT_PATH)
    prefix = 'module.'
    n_clip = len(prefix)
    adapted_dict = {k[n_clip:]: v for k, v in pretrained_dict.items()
                if (k.startswith(prefix + 'pre') or k.startswith(prefix + 'kps') or k.startswith(prefix + 'inter') or k.startswith(prefix + 'cnv'))}
    model.load_state_dict(adapted_dict, strict = False)

    model_dict = model.state_dict()
    count = 0
    for name, param in adapted_dict.items():
        if name not in model_dict:
            continue
        #print(model_dict_2[name])
        if(torch.equal(model_dict[name], param)):
            print(name)
            count += 1
    print('COUNT: ' + str(count))
    

    #model.load_state_dict(checkpoint['model_state_dict'])

    writer.add_text(tag = 'StartingLogs',                                                                     \
                    text_string = (   'BATCH SIZE: {}  \n'.format(batch_size)                                 \
                                    + 'NUMBER OF EPOCHS: {}  \n'.format(epochs)                               \
                                    + 'TRAINING ITERATIONS / EPOCH: {}  \n'.format(len(train_loader))         \
                                    + 'VALIDATION ITERATIONS / EPOCH: {}  \n'.format(len(val_loader))         \
                                    + 'BEST AVERAGE VALIDATION LOSS: {}  \n'.format(best_average_val_loss)    \
                                    + 'LOADED MODEL PARAMETERS: {}'.format(CHECKPOINT_PATH)),                 \
                    global_step = None, walltime = None)

    criterion = AELoss(pull_weight = 1e-1, push_weight = 1e-1)

    optimizer = optim.Adam(model.parameters(), lr = base_lr_rate, weight_decay = weight_decay, amsgrad = True)
    #optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    #for state in optimizer.state.values():
    #  for k, v in state.items():
    #    if torch.is_tensor(v):
    #        state[k] = v.cuda()

    for current_epoch in range(starting_epoch, epochs):
        print('STAAAAAAAAAAAAAAAAAAAAAAAAAAAAAART')

        writer.add_text(tag = 'RunningLogs', text_string = 'EPOCH: {}/{}'.format((current_epoch + 1), epochs), global_step = (current_epoch + 1), walltime = None)
               
        epoch_since = time.time()

        detections_json = []

        current_train_iter = 0
        current_val_iter = 0
        current_mAP_iter = 0
        
        running_train_loss = 0.0
        average_train_loss = 0.0
        running_val_loss = 0.0
        average_val_loss = 0.0

        for phase in ['train', 'val']:
            if phase == 'train':
                
                model.train()

                for train_data in train_loader:
                    train_batch_since = time.time()
                    
                    print('Train: {}'.format(current_train_iter))
                        
                    current_train_iter += 1

                    images, tl_tags, br_tags, tl_heatmaps, br_heatmaps, tag_masks, tl_regrs, br_regrs, names, orig_images, transformed_images = train_data

                    xs = [images, tl_tags, br_tags]
                    ys = [tl_heatmaps, br_heatmaps, tag_masks, tl_regrs, br_regrs]                  
            
                    outs = model(*xs, mode = 'Train')          
            
                    #scheduler = poly_lr_scheduler(optimizer = optimizer, init_lr = base_lr_rate, iter = current_iter, lr_decay_iter = 1, 
                    #                          max_iter = max_iter, power = power)                                                          # max_iter = len(train_loader)
            
                    optimizer.zero_grad()
            
                    loss = criterion(outs, ys)

                    running_train_loss += loss.item()
                    average_train_loss = running_train_loss / current_train_iter

                    writer.add_scalar(tag = 'TrainRunAvgLoss/EPOCH {}'.format(current_epoch + 1), scalar_value = average_train_loss, global_step = current_train_iter)
                    
                    loss.backward(retain_graph = False)
            
                    optimizer.step()
            
                    train_time_elapsed = time.time() - train_batch_since
            
                writer.add_scalar(tag = 'TrainEpochAvgLoss', scalar_value = average_train_loss, global_step = (current_epoch + 1))

            elif phase == 'val':   
                
                model.eval()
                
                with torch.no_grad():
                    for val_data in val_loader:
                        
                        val_batch_since = time.time()
                        
                        print('Valid: {}'.format(current_val_iter))
                        
                        current_val_iter += 1
                        
                        images, tl_tags, br_tags, tl_heatmaps, br_heatmaps, tag_masks, tl_regrs, br_regrs, names, orig_images, transformed_images = val_data

                        xs = [images, tl_tags, br_tags]
                        ys = [tl_heatmaps, br_heatmaps, tag_masks, tl_regrs, br_regrs]                    
                        
                        outs = model(*xs, mode = 'Train')
                        
                        val_loss = criterion(outs, ys)

                        running_val_loss += val_loss.item()
                        average_val_loss = running_val_loss / current_val_iter

                        writer.add_scalar(tag = 'ValRunAvgLoss/EPOCH {}'.format((current_epoch + 1)), scalar_value = average_val_loss, global_step = current_val_iter)
                        
                        val_time_elapsed = time.time() - val_batch_since                                      
                    
                    if(average_val_loss < best_average_val_loss):
                      writer.add_text(tag = 'ModelParams', text_string = '!!! SAVE !!!  \nPREVIOUS BEST AVERAGE VAL LOSS: {}  \nNEW BEST AVERAGE VAL LOSS: {}'.format(best_average_val_loss, average_val_loss), global_step = (current_epoch + 1), walltime = None)

                      best_average_val_loss = average_val_loss                        

                      PATH = '../ModelParams/Hourglass/cornernet_hourglass_pretrained-epoch{}.pth'.format(current_epoch + 1)                        
                      torch.save({
                                'epoch': current_epoch,
                                'iter': current_train_iter,
                                'model_state_dict': model.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'train_loss': average_train_loss,
                                'val_loss': average_val_loss
                                }, PATH)
                    else:
                      writer.add_text(tag = 'ModelParams', text_string = '!!! NO SAVE !!!  \nBEST AVERAGE VAL LOSS: {}  \nCURRENT AVERAGE VAL LOSS: {}'.format(best_average_val_loss, average_val_loss), global_step = (current_epoch + 1), walltime = None)

                    writer.add_scalar(tag = 'ValEpochAvgLoss', scalar_value = average_val_loss, global_step = (current_epoch + 1))
        
        if ((current_epoch % 5) == 0):

            model.eval()

            with torch.no_grad():
                for mAP_data in mAP_loader:
                    mAP_batch_since = time.time()

                    if current_mAP_iter % 500 == 0:
                        print('mAP: {}'.format(current_mAP_iter))

                    current_mAP_iter += 1

                    images, tl_tags, br_tags, tl_heatmaps, br_heatmaps, tag_masks, tl_regrs, br_regrs, names, orig_images, transformed_images = mAP_data

                    detections = model(images, mode = 'mAP', ae_threshold = ae_threshold, top_k = top_k, kernel = nms_kernel)

                    calculate_mAP(detections = detections, names = names, orig_images = orig_images, \
                        transformed_images = transformed_images, detections_json = detections_json, nms_threshold = nms_threshold)
            
                save_detections(detections_json, current_epoch)    

        epoch_time_elapsed = time.time() - epoch_since

        writer.add_text(tag = 'RunningLogs',                                                                      \
                        text_string = (   'CURRENT AVERAGE TRAINING LOSS: {}  \n'.format(average_train_loss)      \
                                        + 'CURRENT AVERAGE VALIDATION LOSS: {}  \n'.format(average_val_loss)      \
                                        + 'BEST AVERAGE VALIDATION LOSS: {}  \n'.format(best_average_val_loss)    \
                                        + 'EPOCH TIME: {}:{}  \n'.format((epoch_time_elapsed // 60 // 60),        \
                                                                            (epoch_time_elapsed // 60 % 60))),     \
                        global_step = (current_epoch + 1), walltime = None)

if __name__ == "__main__":
    train()

# TODO:
#       Dataset átnézése, detection-ök átgondolása
#       test_v2 véglegesítése, az mAP számítás becsatornázása
#       Tensorboardra kitenni az mAP-t is.