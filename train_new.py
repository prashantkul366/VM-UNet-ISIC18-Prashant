import torch
from torch import nn
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from loader import *
from loader import Dataset

# from models.H_vmunet import H_vmunet
from models.vmunet.vmunet import VMUNet
from engine import *
import os
import sys
os.environ["CUDA_VISIBLE_DEVICES"] = "0" # "0, 1, 2, 3"

from utils import *
from configs.config_setting import setting_config

import warnings
warnings.filterwarnings("ignore")

def main(config):

    print('#----------Creating logger----------#')
    sys.path.append(config.work_dir + '/')
    print(f'work dir: {config.work_dir}')
    log_dir = os.path.join(config.work_dir, 'log')
    checkpoint_dir = os.path.join(config.work_dir, 'checkpoints')
    resume_model = os.path.join(checkpoint_dir, 'latest.pth')
    # best_dice = checkpoint.get('best_dice', best_dice)
    outputs = os.path.join(config.work_dir, 'outputs')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    if not os.path.exists(outputs):
        os.makedirs(outputs)

    global logger
    logger = get_logger('train', log_dir)

    log_config_info(config, logger)





    print('#----------GPU init----------#')
    set_seed(config.seed)
    gpu_ids = [0]# [0, 1, 2, 3]
    torch.cuda.empty_cache()





    print('#----------Preparing dataset----------#')
    # train_dataset = isic_loader(path_Data = config.data_path, train = True)
    H, W = config.input_size_h, config.input_size_w
    root = config.data_path
    train_dataset = Dataset(
                        root=config.data_path, split="train",
                        images_dir="images", masks_dir="masks",
                        train_augs=True, target_size=(H, W)
                    )
    train_loader = DataLoader(train_dataset,
                                batch_size=config.batch_size, 
                                shuffle=True,
                                pin_memory=True,
                                num_workers=config.num_workers)
    
    # val_dataset = isic_loader(path_Data = config.data_path, train = False)
    # val_split = "val"
    # val_dataset = Dataset(
    #                 root=config.data_path, split="val",
    #                 images_dir="images", masks_dir="masks",
    #                 train_augs=False, target_size=(H, W)
    #             )
    
    val_split = "val"
    val_dataset = Dataset(
                    root=config.data_path, split="val",
                    images_dir="images", masks_dir="masks",
                    train_augs=False, target_size=(H, W)
                )
    val_loader = DataLoader(val_dataset,
                                batch_size=1,
                                shuffle=False,
                                pin_memory=True, 
                                num_workers=config.num_workers,
                                drop_last=True)
    # test_dataset = isic_loader(path_Data = config.data_path, train = False, Test = True)
    test_dataset = Dataset(
                        root=config.data_path, split="test",
                        images_dir="images", masks_dir="masks",
                        train_augs=False, target_size=(H, W)
                    )
    test_loader = DataLoader(test_dataset,
                                batch_size=1,
                                shuffle=False,
                                pin_memory=True, 
                                num_workers=config.num_workers,
                                drop_last=True)




    print('#----------Prepareing Models----------#')
    model_cfg = config.model_config
    # model = H_vmunet(num_classes=model_cfg['num_classes'], 
    #                 input_channels=model_cfg['input_channels'], 
    #                 c_list=model_cfg['c_list'], 
    #                 split_att=model_cfg['split_att'], 
    #                 bridge=model_cfg['bridge'],
    #                 drop_path_rate=model_cfg['drop_path_rate'])
    
    model = VMUNet(
            num_classes=model_cfg['num_classes'],
            input_channels=model_cfg['input_channels'],
            depths=model_cfg['depths'],
            depths_decoder=model_cfg['depths_decoder'],
            drop_path_rate=model_cfg['drop_path_rate'],
            load_ckpt_path=model_cfg['load_ckpt_path'],
        )
    model = torch.nn.DataParallel(model.cuda(), device_ids=gpu_ids, output_device=gpu_ids[0])





    print('#----------Prepareing loss, opt, sch and amp----------#')
    criterion = config.criterion
    optimizer = get_optimizer(config, model)
    scheduler = get_scheduler(config, optimizer)
    scaler = GradScaler()





    print('#----------Set other params----------#')
    min_loss = 999
    start_epoch = 1
    min_epoch = 1

    patience = 100            
    best_dice = -1.0
    epochs_no_improve = 0




    if os.path.exists(resume_model):
        print('#----------Resume Model and Other params----------#')
        checkpoint = torch.load(resume_model, map_location=torch.device('cpu'))
        model.module.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        saved_epoch = checkpoint['epoch']
        start_epoch += saved_epoch
        min_loss, min_epoch, loss = checkpoint['min_loss'], checkpoint['min_epoch'], checkpoint['loss']

        log_info = f'resuming model from {resume_model}. resume_epoch: {saved_epoch}, min_loss: {min_loss:.4f}, min_epoch: {min_epoch}, loss: {loss:.4f}'
        logger.info(log_info)





    print('#----------Training----------#')
    for epoch in range(start_epoch, config.epochs + 1):

        torch.cuda.empty_cache()

        train_one_epoch(
            train_loader,
            model,
            criterion,
            optimizer,
            scheduler,
            epoch,
            logger,
            config,
            scaler=scaler
        )

        # loss = val_one_epoch(
        #         val_loader,
        #         model,
        #         criterion,
        #         epoch,
        #         logger,
        #         config
        #     )

        val_loss, val_dice = val_one_epoch(
            val_loader,
            model,
            criterion,
            epoch,
            logger,
            config
        )


        # if loss < min_loss:
        #     torch.save(model.module.state_dict(), os.path.join(checkpoint_dir, 'best.pth'))
        #     min_loss = loss
        #     min_epoch = epoch

        # Save best by Dice
        improved = val_dice > best_dice + 1e-8
        if improved:
            best_dice = val_dice
            epochs_no_improve = 0
            torch.save(model.module.state_dict(), os.path.join(checkpoint_dir, 'best.pth'))
            print(f'Validation Dice improved to {best_dice:.4f}, saving model to {os.path.join(checkpoint_dir, "best.pth")}')
        else:
            epochs_no_improve += 1

        print_msg = f'Validation Dice: {val_dice:.4f}, Best Dice: {best_dice:.4f} (at epoch {min_epoch}) epochs_no_improve: {epochs_no_improve}'
        print(print_msg)
        # torch.save(
        #     {
        #         'epoch': epoch,
        #         'min_loss': min_loss,
        #         'min_epoch': min_epoch,
        #         'loss': loss,
        #         'model_state_dict': model.module.state_dict(),
        #         'optimizer_state_dict': optimizer.state_dict(),
        #         'scheduler_state_dict': scheduler.state_dict(),
        #     }, os.path.join(checkpoint_dir, 'latest.pth')) 

        torch.save(
            {
                'epoch': epoch,
                'min_loss': min_loss,
                'min_epoch': min_epoch,
                'loss': val_loss,
                'best_dice': best_dice,
                'model_state_dict': model.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            }, os.path.join(checkpoint_dir, 'latest.pth')
        )

        if epochs_no_improve >= patience:
            stop_msg = (f"Early stopping at epoch {epoch} "
                        f"(no Dice improvement for {patience} validation epochs). "
                        f"Best Dice: {best_dice:.4f}")
            print(stop_msg)
            logger.info(stop_msg)
            break

    if os.path.exists(os.path.join(checkpoint_dir, 'best.pth')):
        print('#----------Testing----------#')
        best_weight = torch.load(os.path.join(checkpoint_dir, 'best.pth'),
                                 map_location=torch.device('cpu'))
        model.module.load_state_dict(best_weight)
        loss = test_one_epoch(
            test_loader,
            model,
            criterion,
            logger,
            config,
        )
        # include min_epoch and min_loss in the filename (from training tracking)
        os.rename(
            os.path.join(checkpoint_dir, 'best.pth'),
            os.path.join(checkpoint_dir, f'best-epoch{min_epoch}-loss{min_loss:.4f}.pth')
        )
    


if __name__ == '__main__':
    config = setting_config
    main(config)