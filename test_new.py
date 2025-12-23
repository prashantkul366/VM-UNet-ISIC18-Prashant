import torch
from torch import nn
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from loader import *

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
from thop import profile, clever_format



def main(config):

    print('#----------Creating logger----------#')
    sys.path.append(config.work_dir + '/')
    log_dir = os.path.join(config.work_dir, 'log')
    checkpoint_dir = os.path.join(config.work_dir, 'checkpoints')
    # resume_model = os.path.join('')
    # resume_model = '/content/drive/MyDrive/Prashant/H-vmunet_prashant_ISIC/results/H_vmunet_ISIC2018_Thursday_04_December_2025_06h_08m_08s/checkpoints/best.pth'
    # resume_model = '/content/drive/MyDrive/Prashant/H-vmunet_prashant_ISIC/results/H_vmunet_ISIC2018_Thursday_04_December_2025_06h_08m_08s/checkpoints/best.pth'
    # resume_model = '/content/drive/MyDrive/Prashant/H-vmunet_prashant_ISIC/results/H_vmunet_BUSI_Friday_05_December_2025_07h_20m_32s/checkpoints/best.pth'

    # resume_model = '/content/drive/MyDrive/Prashant/VM-UNet-ISIC18-Prashant/results/vmunet_CVC-ClinicDB_80_20_Thursday_18_December_2025_11h_33m_09s/checkpoints/best.pth'
    # resume_model = '/content/drive/MyDrive/Prashant/VM-UNet-ISIC18-Prashant/results/vmunet_BUSI_Friday_19_December_2025_05h_25m_02s/checkpoints/best.pth'
    resume_model = "/content/drive/MyDrive/Prashant/VM-UNet-ISIC18-Prashant/results/vmunet_ISIC2018_Monday_22_December_2025_05h_15m_38s/checkpoints/best.pth"
    
    outputs = os.path.join(config.work_dir, 'outputs')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    if not os.path.exists(outputs):
        os.makedirs(outputs)

    global logger
    logger = get_logger('test', log_dir)

    log_config_info(config, logger)





    print('#----------GPU init----------#')
    set_seed(config.seed)
    gpu_ids = [0]# [0, 1, 2, 3]
    torch.cuda.empty_cache()
    


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


    # ---------- Model complexity (Params & GFLOPs) ----------
    # model.eval()
    # dummy = torch.zeros(
    #     1,
    #     config.input_channels,
    #     config.input_size_h,
    #     config.input_size_w,
    #     device='cuda'
    # )
    # with torch.no_grad():
    #     macs, params = profile(model.module, inputs=(dummy,), verbose=False)

    
    # ---------- Model complexity (Params & GFLOPs) ----------
    # model_for_profile = H_vmunet(
    #     num_classes=model_cfg['num_classes'], 
    #     input_channels=model_cfg['input_channels'], 
    #     c_list=model_cfg['c_list'], 
    #     split_att=model_cfg['split_att'], 
    #     bridge=model_cfg['bridge'],
    #     drop_path_rate=model_cfg['drop_path_rate']
    # ).cuda()

    model_for_profile = VMUNet(
            num_classes=model_cfg['num_classes'],
            input_channels=model_cfg['input_channels'],
            depths=model_cfg['depths'],
            depths_decoder=model_cfg['depths_decoder'],
            drop_path_rate=model_cfg['drop_path_rate'],
            load_ckpt_path=model_cfg['load_ckpt_path'],
        ).cuda()

    model_for_profile.eval()

    dummy = torch.zeros(
        1,
        config.input_channels,
        config.input_size_h,
        config.input_size_w,
        device='cuda'
    )

    with torch.no_grad():
        macs, params = profile(model_for_profile, inputs=(dummy,), verbose=False)
        
    macs_str, params_str = clever_format([macs, params], "%.3f") 
    gmacs = macs / 1e9
    gflops = 2.0 * gmacs  # FLOPs ≈ 2 * MACs


    print(f"#----------Model complexity----------#")
    print(f"Params: {params_str} | MACs: {macs_str} | GFLOPs (≈2*MACs): {gflops:.3f} GFLOPs")
    logger.info(f"Params: {params_str} | MACs: {macs_str} | GFLOPs (≈2*MACs): {gflops:.3f} GFLOPs")


    print('#----------Preparing dataset----------#')
    # test_dataset = isic_loader(path_Data = config.data_path, train = False, Test = True)
    # test_loader = DataLoader(test_dataset,
    #                             batch_size=1,
    #                             shuffle=False,
    #                             pin_memory=True, 
    #                             num_workers=config.num_workers,
    #                             drop_last=True)

    H, W = config.input_size_h, config.input_size_w
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

    print('#----------Prepareing loss, opt, sch and amp----------#')
    criterion = config.criterion
    optimizer = get_optimizer(config, model)
    scheduler = get_scheduler(config, optimizer)
    scaler = GradScaler()





    print('#----------Set other params----------#')
    min_loss = 999
    start_epoch = 1
    min_epoch = 1


    print('#----------Testing----------#')
    best_weight = torch.load(resume_model, map_location=torch.device('cpu'))
    model.module.load_state_dict(best_weight)

    loss = test_one_epoch(
            test_loader,
            model,
            criterion,
            logger,
            config,
        )



if __name__ == '__main__':
    config = setting_config
    main(config)