import torch
from torch.utils.data import DataLoader
import timm
from datasets.dataset import NPY_datasets, NPY_datasets_test
from tensorboardX import SummaryWriter
from models.vmunet.vmunet import VMUNet

from engine import *
import os
import sys

from utils import *
from configs.config_setting import setting_config

import warnings
warnings.filterwarnings("ignore")



def main(config):

    print('#----------Creating logger----------#')
    sys.path.append(config.work_dir + '/')
    log_dir = os.path.join(config.work_dir, 'log')
    checkpoint_dir = os.path.join(config.work_dir, 'checkpoints')
    resume_model = os.path.join(checkpoint_dir, 'latest.pth')
    # best_dice  = checkpoint.get('best_dice', best_dice)
    # best_epoch = checkpoint.get('best_epoch', best_epoch)
    outputs = os.path.join(config.work_dir, 'outputs')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    if not os.path.exists(outputs):
        os.makedirs(outputs)

    global logger
    logger = get_logger('train', log_dir)
    global writer
    writer = SummaryWriter(config.work_dir + 'summary')

    log_config_info(config, logger)





    print('#----------GPU init----------#')
    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_id
    set_seed(config.seed)
    torch.cuda.empty_cache()





    print('#----------Preparing dataset----------#')
    # train_dataset = NPY_datasets(config.data_path, config, train=True)
    # train_loader = DataLoader(train_dataset,
    #                             batch_size=config.batch_size, 
    #                             shuffle=True,
    #                             pin_memory=True,
    #                             num_workers=config.num_workers)
    # val_dataset = NPY_datasets(config.data_path, config, train=False)
    # val_loader = DataLoader(val_dataset,
    #                             batch_size=1,
    #                             shuffle=False,
    #                             pin_memory=True, 
    #                             num_workers=config.num_workers,
    #                             drop_last=True)





    print('#----------Prepareing Model----------#')
    model_cfg = config.model_config
    if config.network == 'vmunet':
        model = VMUNet(
            num_classes=model_cfg['num_classes'],
            input_channels=model_cfg['input_channels'],
            depths=model_cfg['depths'],
            depths_decoder=model_cfg['depths_decoder'],
            drop_path_rate=model_cfg['drop_path_rate'],
            load_ckpt_path=model_cfg['load_ckpt_path'],
        )
        # model.load_from()
        
        if model_cfg.get('load_ckpt_path') and os.path.exists(model_cfg['load_ckpt_path']):
            model.load_from()
        else:
            print('[VMUNet] No pretrained checkpoint provided; training from scratch.')

    else: raise Exception('network in not right!')
    model = model.cuda()

    cal_params_flops(model, 256, logger)





    print('#----------Prepareing loss, opt, sch and amp----------#')
    criterion = config.criterion
    optimizer = get_optimizer(config, model)
    scheduler = get_scheduler(config, optimizer)
    start_epoch = 1 

    # --- Dice-based tracking & early stopping ---
    best_dice = 0.0        
    best_epoch = 0           
    dice_patience = getattr(config, 'early_stopping_patience', 30)  # fallback to 30 if not in config  


    print('#----------Set other params----------#')
    min_loss = 999           # keep if you still want to log loss; not used for early stop now
    min_epoch = 1


    if config.only_test_and_save_figs:
        checkpoint = torch.load(config.best_ckpt_path, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint)
        config.work_dir = config.img_save_path
        if not os.path.exists(config.work_dir + 'outputs/'):
            os.makedirs(config.work_dir + 'outputs/')
        loss = test_one_epoch(
                val_loader,
                model,
                criterion,
                logger,
                config,
            )
        return




    if os.path.exists(resume_model):
        print('#----------Resume Model and Other params----------#')
        checkpoint = torch.load(resume_model, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        saved_epoch = checkpoint['epoch']
        start_epoch += saved_epoch
        min_loss, min_epoch, loss = checkpoint['min_loss'], checkpoint['min_epoch'], checkpoint['loss']

        best_dice  = checkpoint.get('best_dice', best_dice)
        best_epoch = checkpoint.get('best_epoch', best_epoch)

        log_info = f'resuming model from {resume_model}. resume_epoch: {saved_epoch}, min_loss: {min_loss:.4f}, min_epoch: {min_epoch}, loss: {loss:.4f}'
        logger.info(log_info)




    step = 0
    print('#----------Training----------#')
    # for epoch in range(start_epoch, config.epochs + 1):

    #     torch.cuda.empty_cache()
    #     print(f'Epoch {epoch}/{config.epochs}:')
    #     step = train_one_epoch(
    #         train_loader,
    #         model,
    #         criterion,
    #         optimizer,
    #         scheduler,
    #         epoch,
    #         step,
    #         logger,
    #         config,
    #         writer
    #     )

    #     # loss = val_one_epoch(
    #     #         val_loader,
    #     #         model,
    #     #         criterion,
    #     #         epoch,
    #     #         logger,
    #     #         config
    #     #     )

    #     val_loss, val_dice = val_one_epoch(
    #                 val_loader,
    #                 model,
    #                 criterion,
    #                 epoch,
    #                 logger,
    #                 config
    #             )


    #     # if loss < min_loss:
    #     #     torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'best.pth'))
    #     #     min_loss = loss
    #     #     min_epoch = epoch

    #     # --- Save best based on Val Dice (higher is better) ---
    #     if val_dice > best_dice:
    #         logger.info(f"\tSaving best model: Dice {best_dice:.4f} -> {val_dice:.4f}")
    #         print(f"\tSaving best model: Dice {best_dice:.4f} -> {val_dice:.4f}")
    #         best_dice  = val_dice
    #         best_epoch = epoch
    #         torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'best.pth'))
    #     else:
    #         logger.info(f"\tDice did not improve: curr {val_dice:.4f}, best {best_dice:.4f} @ epoch {best_epoch}")
    #         print(f"\tDice did not improve: curr {val_dice:.4f}, best {best_dice:.4f} @ epoch {best_epoch}")


    #     # torch.save(
    #     #     {
    #     #         'epoch': epoch,
    #     #         'min_loss': min_loss,
    #     #         'min_epoch': min_epoch,
    #     #         'loss': loss,
    #     #         'model_state_dict': model.state_dict(),
    #     #         'optimizer_state_dict': optimizer.state_dict(),
    #     #         'scheduler_state_dict': scheduler.state_dict(),
    #     #     }, os.path.join(checkpoint_dir, 'latest.pth')) 

    #     torch.save(
    #             {
    #                 'epoch': epoch,
    #                 'min_loss': min_loss,
    #                 'min_epoch': min_epoch,
    #                 'loss': val_loss,                       # store the current val loss
    #                 'best_dice': best_dice,                 # NEW: store best Dice so far
    #                 'best_epoch': best_epoch,               # NEW: store epoch of best Dice
    #                 'model_state_dict': model.state_dict(),
    #                 'optimizer_state_dict': optimizer.state_dict(),
    #                 'scheduler_state_dict': scheduler.state_dict(),
    #             },
    #             os.path.join(checkpoint_dir, 'latest.pth')
    #         )

    #     # --- Early stopping on Dice ---
    #     early_stopping_count = epoch - best_epoch
    #     logger.info(f"\tEarly stopping patience: {early_stopping_count}/{dice_patience}")
    #     print(f"\tEarly stopping patience: {early_stopping_count}/{dice_patience}")
    #     if early_stopping_count >= dice_patience:
    #         logger.info('\tEarly stopping triggered (Dice)!')
    #         print('\tEarly stopping triggered (Dice)!')
    #         break



    # if os.path.exists(os.path.join(checkpoint_dir, 'best.pth')):
    #     print('#----------Testing----------#')

    #     ######################################
    #     test_dataset = NPY_datasets_test(config.data_path, config, test=True)
    #     test_loader = DataLoader(test_dataset,
    #                             batch_size=1,
    #                             shuffle=False,
    #                             pin_memory=True, 
    #                             num_workers=config.num_workers,
    #                             drop_last=True)
        
    #     ######################################
    #     best_weight = torch.load(config.work_dir + 'checkpoints/best.pth', map_location=torch.device('cpu'))
    #     model.load_state_dict(best_weight)
    #     loss = test_one_epoch(
    #             test_loader,
    #             model,
    #             criterion,
    #             logger,
    #             config,
    #         )
    #     # os.rename(
    #     #     os.path.join(checkpoint_dir, 'best.pth'),
    #     #     os.path.join(checkpoint_dir, f'best-epoch{min_epoch}-loss{min_loss:.4f}.pth')
    #     # ) 
    #     os.rename(
    #         os.path.join(checkpoint_dir, 'best.pth'),
    #         os.path.join(checkpoint_dir, f'best-epoch{best_epoch}-dice{best_dice:.4f}.pth')
    #     )

    
    print('#----------Testing----------#')

    ######################################
    test_dataset = NPY_datasets_test(config.data_path, config, test=True)
    test_loader = DataLoader(test_dataset,
                            batch_size=1,
                            shuffle=False,
                            pin_memory=True, 
                            num_workers=config.num_workers,
                            drop_last=True)
    
    ######################################
    checkpt_path = '/content/drive/MyDrive/Prashant/VM-UNet-ISIC18-Prashant/results/vmunet_isic18_Monday_15_September_2025_05h_01m_45s/checkpoints/best.pth'
    best_weight = torch.load(checkpt_path, map_location=torch.device('cpu'))
    model.load_state_dict(best_weight)
    loss = test_one_epoch(
            test_loader,
            model,
            criterion,
            logger,
            config,
        )
    # os.rename(
    #     os.path.join(checkpoint_dir, 'best.pth'),
    #     os.path.join(checkpoint_dir, f'best-epoch{min_epoch}-loss{min_loss:.4f}.pth')
    # ) 
    # os.rename(
    #     os.path.join(checkpoint_dir, 'best.pth'),
    #     os.path.join(checkpoint_dir, f'best-epoch{best_epoch}-dice{best_dice:.4f}.pth')
    # )
     


if __name__ == '__main__':
    config = setting_config
    main(config)