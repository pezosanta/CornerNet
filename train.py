import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import time
import glob
import warnings
from Dataset import Dataset
from CornerNet import kp as Cornernet
from losses import AELoss

warnings.filterwarnings(action = 'once')

# test from vscode

def train(batch_size = 4, epochs = 30):
    since = time.time()

    n = 5
    nstack = 2
    dims    = [256, 256, 384, 384, 384, 512]
    modules = [2, 2, 2, 2, 2, 4]
    out_dim = 10
    
    base_lr_rate = 0.00025
    weight_decay = 0.0 #0.0016

    starting_epoch = 0
    starting_iter = 0
    best_average_val_loss = 10000.0
    
    
    ####################### LOAD CHECKPOINTS #######################
    #CHECKPOINT_PATH = '/home/pezosanta/Deep Learning/Supervised Learning/CornerNet/ModelParams/train_valid_pretrained_cornernet-epoch{}-iter{}.pth'.format(3, 5067)
    CHECKPOINT_PATH = './ModelParams/train_valid_pretrained_cornernet-epoch{}-iter{}.pth'.format(3, 5067)
    checkpoint = torch.load(CHECKPOINT_PATH)
    starting_epoch = checkpoint['epoch']
    starting_iter = checkpoint['iter'] + 1
    best_average_val_loss = checkpoint['val_loss']
    ################################################################
    

    # Train Dataset és train dataloader
    train_dataset = Dataset(mode = 'Train')
    train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
    
    # Val Dataset és val dataloader
    val_dataset = Dataset(mode = 'Val')
    val_loader = DataLoader(val_dataset, batch_size = batch_size, shuffle = True)
    
    train_loader_iter = iter(train_loader)
    val_loader_iter = iter(val_loader)

    max_iter = len(train_loader) #17465
    num_iter_until_val = 10000   
    
    print('-' * 30)
    print('BATCH SIZE: ' + str(batch_size))
    print('NUMBER OF TRAINING DATA BATCHES: ' + str(len(train_loader)))
    print('NUMBER OF VALIDATION DATA BATCHES: ' + str(len(val_loader)))
    print('NUMBER OF EPOCHS: ' + str(epochs))
    print('BEST AVERAGE VAL LOSS: ' + str(best_average_val_loss))
    print('-' * 30)
   
    model = Cornernet(n = n, nstack = nstack, dims = dims, modules = modules, out_dim = out_dim).cuda()
    #model = CornerNet(n = n, nstack = nstack, dims = dims, modules = modules, out_dim = out_dim).cuda()

    '''
    # Load the original pretrained weights in the first epoch
    CHECKPOINT_PATH = '/content/CornerNet_500000.pkl'
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
    '''

    model.load_state_dict(checkpoint['model_state_dict'])

    criterion = AELoss(pull_weight = 1e-1, push_weight = 1e-1)

    optimizer = optim.Adam(model.parameters(), lr = base_lr_rate, weight_decay = weight_decay)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    #for state in optimizer.state.values():
    #  for k, v in state.items():
    #    if torch.is_tensor(v):
    #        state[k] = v.cuda()

    running_iter = 0

    for current_epoch in range(starting_epoch, epochs):
        #print('-' * 30)
        print('EPOCH {}/{}'.format(current_epoch, epochs))
        print('-' * 30)
        
        epoch_since = time.time()
        
        train_loss = 0.0

        #print('-' * 30)
        print('! TRAINING STEP !')
        print('-' * 30)        
        
        #for i in range(20):
        for current_train_iter in range(starting_iter, max_iter):
            train_batch_since = time.time()
            model.train()  # Set model to training mode                      

            '''                      
            if((current_train_iter % 250) == 0):
                #print('-' * 30)
                print('EPOCH ({}), TRAIN ITER {}/{}'.format(current_epoch, current_train_iter, max_iter))
                print('TRAIN LOSS IS CALCULATED: {}'.format(loss.item()))
                print('-' * 30)
            '''

            try:
                images, tl_tags, br_tags, tl_heatmaps, br_heatmaps, tag_masks, tl_regrs, br_regrs = next(train_loader_iter)
            except StopIteration:
                train_loader_iter = iter(train_loader)
                images, tl_tags, br_tags, tl_heatmaps, br_heatmaps, tag_masks, tl_regrs, br_regrs = next(train_loader_iter)

            #images, tl_tags, br_tags, tl_heatmaps, br_heatmaps, tag_masks, tl_regrs, br_regrs = train_data
            xs = [images, tl_tags, br_tags]
            ys = [tl_heatmaps, br_heatmaps, tag_masks, tl_regrs, br_regrs]                  
            
            outs = model(*xs)
            
            #print('-' * 30)
            #print('CUDA MEMORY ALLOCATED: ' + str(torch.cuda.memory_allocated(torch.device('cuda:0'))/(1024**3)))
            #print('-' * 30)            
            
            #scheduler = poly_lr_scheduler(optimizer = optimizer, init_lr = base_lr_rate, iter = current_iter, lr_decay_iter = 1, 
            #                          max_iter = max_iter, power = power)
            
            optimizer.zero_grad()
            
            #print('-' * 30)
            #print('ZERO GRAD DONE')
            #print('-' * 30)
            
            loss = criterion(outs, ys)

            if((current_train_iter % 250) == 0):
                #print('-' * 30)
                print('EPOCH ({}), TRAIN ITER {}/{}'.format(current_epoch, current_train_iter, max_iter))
                print('TRAIN LOSS IS CALCULATED: {}'.format(loss.item()))
                print('-' * 30)
            
            #print('-' * 30)
            #print('TRAIN LOSS IS CALCULATED: {}'.format(loss.item()))
            #print('-' * 30)
                            
            loss.backward(retain_graph = False)
            
            #print('-' * 30)
            #print('BACKPROPAGATION DONE')
            #print('-' * 30)
            
            optimizer.step()
            
            #print('-' * 30)
            #print('OPTIMIZATION STEP DONE')
            #print('-' * 30)
            
            #print('-' * 30)
            #print('CUDA MEMORY ALLOCATED: ' + str(torch.cuda.memory_allocated(torch.device('cuda:0'))/(1024**3)))
            #print('-' * 30)
            
            train_time_elapsed = time.time() - train_batch_since
            
            #print('-' * 30)
            #print('TRAINING BATCH TIME IN SEC: ' + str(train_time_elapsed))
            #print('-' * 30)
            
            #!nvidia-smi
            
            train_loss = loss.item()
            
            # VALIDATION IN EVERY 6000 TRAIN ITERATION 
            if(((running_iter + 1) % num_iter_until_val) == 0):           
                
                current_val_iter = 0
                running_val_loss = 0.0
                current_average_val_loss = 0.0  

                #print('-' * 30)
                print('! VALIDATION STEP !')
                print('-' * 30)
                
                model.eval()
                
                with torch.no_grad():
                    for val_data in val_loader:
                    #for i in range(20):

                        val_batch_since = time.time()
                        
                        images, tl_tags, br_tags, tl_heatmaps, br_heatmaps, tag_masks, tl_regrs, br_regrs = val_data
                        '''
                        if((current_val_iter % 250) == 0):
                            #print('-' * 30)
                            print('EPOCH ({}), VALIDATION ITER {}/{}'.format(current_epoch, current_val_iter, len(val_loader)))
                            print('VAL LOSS IS CALCULATED ! LOSS = {}'.format(loss.item()))
                            print('-' * 30)
                        '''
                        '''
                        try:
                            images, tl_tags, br_tags, tl_heatmaps, br_heatmaps, tag_masks, tl_regrs, br_regrs = next(val_loader_iter)
                        except StopIteration:
                            val_loader_iter = iter(val_loader)
                            images, tl_tags, br_tags, tl_heatmaps, br_heatmaps, tag_masks, tl_regrs, br_regrs = next(val_loader_iter)
                        #images, tl_tags, br_tags, tl_heatmaps, br_heatmaps, tag_masks, tl_regrs, br_regrs = val_data
                        '''

                        xs = [images, tl_tags, br_tags]
                        ys = [tl_heatmaps, br_heatmaps, tag_masks, tl_regrs, br_regrs]                    
                        
                        outs = model(*xs)
                        
                        loss = criterion(outs, ys)

                        if((current_val_iter % 250) == 0):
                            #print('-' * 30)
                            print('EPOCH ({}), VALIDATION ITER {}/{}'.format(current_epoch, current_val_iter, len(val_loader)))
                            print('VAL LOSS IS CALCULATED ! LOSS = {}'.format(loss.item()))
                            print('-' * 30)
                        
                        #print('-' * 30)
                        #print('VAL LOSS IS CALCULATED ! LOSS = {}'.format(loss.item()))
                        #print('-' * 30)
                        
                        #print('-' * 30)
                        #print('CUDA MEMORY ALLOCATED: ' + str(torch.cuda.memory_allocated(torch.device('cuda:0'))/(1024**3)))
                        #print('-' * 30)
                        
                        val_time_elapsed = time.time() - val_batch_since
                        
                        #print('-' * 30)
                        #print('VALIDATION BATCH TIME IN SEC: ' + str(val_time_elapsed))
                        #print('-' * 30)
                        
                        #!nvidia-smi
                        
                        running_val_loss = running_val_loss + loss.item()

                        current_val_iter += 1
                    
                    current_average_val_loss = running_val_loss / len(val_loader)                
                    
                    if(current_average_val_loss < best_average_val_loss):
                        
                        print('!!! SAVE !!! \nPREVIOUS BEST AVERAGE VAL LOSS: {} \nNEW BEST AVERAGE VAL LOSS: {}'.format(best_average_val_loss, current_average_val_loss))

                        best_average_val_loss = current_average_val_loss                        

                        PATH = './ModelParams/train_valid_pretrained_cornernet-epoch{}-iter{}.pth'.format(current_epoch, current_train_iter)                        
                        torch.save({
                                    'epoch': current_epoch,
                                    'iter': current_train_iter,
                                    'model_state_dict': model.state_dict(),
                                    'optimizer_state_dict': optimizer.state_dict(),
                                    'train_loss': train_loss,
                                    'val_loss': current_average_val_loss
                                    }, PATH)
                    else:
                        print('!!! NO SAVE !!! \nBEST AVERAGE VAL LOSS: {} \nCURRENT AVERAGE VAL LOSS: {}'.format(best_average_val_loss, current_average_val_loss))

                    print('-' * 30)
                    print('! TRAINING STEP !')
                    print('-' * 30)

            running_iter += 1
        
        epoch_time_elapsed = time.time() - epoch_since

        starting_iter = 0

        print('-' * 30)
        print('TRAINING LOSS: ' + str(train_loss))
        print('BEST_AVERAGE VAL LOSS: ' + str(best_average_val_loss))
        print('EPOCH TIME IN MINUTES: ' + str(epoch_time_elapsed // 60))
        print('EPOCH TIME IN HOURS: ' + str(epoch_time_elapsed // 60 / 60))
        print('-' * 30)     

if __name__ == "__main__":
    train()
