import torch
from torch.utils.tensorboard import SummaryWriter

def loading_original_model_params(checkpoint_path, model):
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

    return model

def tensorboard_starting_logs(writer, checkpoint_path, batch_size, epochs, starting_epoch, train_loader, val_loader, base_lr_rate, weight_decay, mean_ap_epoch_interval, mean_ap_thresholds,
                                best_epoch_average_train_loss, last_epoch_average_train_loss, best_epoch_average_val_loss, last_epoch_average_val_loss, best_mAP_50, last_mAP_50,
                                best_mAP_75, last_mAP_75):
    writer.add_text(tag = 'Hourglass/StartingLogs',                                                                                                       \
                    text_string = (   'LOADED MODEL PARAMETERS: {}  \n'.format(checkpoint_path)                                                           \
                                    + 'BATCH SIZE: {}  \n'.format(batch_size)                                                                             \
                                    + 'NUMBER OF GPUS: {}  \n'.format(torch.cuda.device_count())                                                          \
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

def tensorboard_add_scalars(writer, tag, cat_list, current_breakdown, current_epoch):
    writer.add_scalars(main_tag = tag, 
                        tag_scalar_dict =  {cat_list[0].upper(): current_breakdown[0],
                                            cat_list[1].upper(): current_breakdown[1],
                                            cat_list[2].upper(): current_breakdown[2],
                                            cat_list[3].upper(): current_breakdown[3],
                                            cat_list[4].upper(): current_breakdown[4],
                                            cat_list[5].upper(): current_breakdown[5],
                                            cat_list[6].upper(): current_breakdown[6],
                                            cat_list[7].upper(): current_breakdown[7],
                                            cat_list[8].upper(): current_breakdown[8],
                                            cat_list[9].upper(): current_breakdown[9]},
                        global_step = (current_epoch + 1))

def saving_model_params(PATH, current_epoch, current_train_iter, base_lr_rate, weight_decay, model, optimizer, best_epoch_average_train_loss, best_epoch_average_val_loss,
                            best_mAP_50, best_mAP_75, best_breakdown_50, best_breakdown_75, last_epoch_average_train_loss, last_epoch_average_val_loss, last_mAP_50, last_mAP_75,
                            last_breakdown_50, last_breakdown_75):                        
    torch.save({
            'epoch': current_epoch,
            'iter': current_train_iter,
            'lr': base_lr_rate,
            'weight_decay': weight_decay,
            'model_state_dict': model.module.state_dict(),
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

def tensorboard_saving_logs(writer, title, best_epoch_average_train_loss, last_epoch_average_train_loss, best_epoch_average_val_loss, last_epoch_average_val_loss,
                                mean_ap_thresholds, best_mAP_50, last_mAP_50, best_mAP_75, last_mAP_75, current_epoch):
    writer.add_text(tag = 'Hourglass/SavingLogs',                                                                             \
                            text_string = (   title + '  \n'                                                                       \
                                            + '[BEST/LAST] AVERAGE TRAINING LOSS: [{:5f} / {:5f}]  \n'.format(
                                                    best_epoch_average_train_loss, last_epoch_average_train_loss)                  \
                                            + '[BEST/LAST] AVERAGE VALIDATION LOSS: [{:5f} / {:5f}]  \n'.format(
                                                    best_epoch_average_val_loss, last_epoch_average_val_loss)                      \
                                            + '[BEST/LAST] MEAN AVERAGE PRECISION (IOU = {}%): [{:2f} / {:2f}]  \n'.format(
                                                    int(mean_ap_thresholds[0] * 100), best_mAP_50, last_mAP_50)                    \
                                            + '[BEST/LAST] MEAN AVERAGE PRECISION (IOU = {}%): [{:2f} / {:2f}]  \n'.format(
                                                    int(mean_ap_thresholds[1] * 100), best_mAP_75, last_mAP_75)),                  \
                            global_step = (current_epoch + 1), walltime = None)

def tensorboard_add_hparams(writer, current_epoch, batch_size, optimizer_name, base_lr_rate, weight_decay, last_epoch_average_train_loss, last_epoch_average_val_loss,
                                last_mAP_50, last_mAP_75, cat_list, last_breakdown_50, last_breakdown_75, is_saved):
    writer.add_hparams(hparam_dict = {  'EPOCH': str(current_epoch + 1),
                                        'SAVED': str(is_saved),
                                        'BATCH SIZE': str(batch_size),
                                        'OPTIMIZER': optimizer_name,
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
                                        cat_list[9].upper() + ' [AP50 / AP75]': '{:2f} / {:2f}'.format(last_breakdown_50[9], last_breakdown_75[9])},
                        metric_dict = {'Hourglass/W_LEARNING_RATE': base_lr_rate})

def tensorboard_epoch_ending_logs(writer, best_epoch_average_train_loss, last_epoch_average_train_loss, best_epoch_average_val_loss, last_epoch_average_val_loss, mean_ap_thresholds,
                                    best_mAP_50, last_mAP_50, best_mAP_75, last_mAP_75, epoch_time_elapsed, train_time_elapsed, val_time_elapsed, mAP_time_elapsed, current_epoch):
    writer.add_text(tag = 'Hourglass/RunningLogs',                                                                         \
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