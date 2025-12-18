# import numpy as np
# from tqdm import tqdm
# import torch
# from torch.cuda.amp import autocast as autocast
# from sklearn.metrics import confusion_matrix
# from utils import save_imgs
# import time

# def train_one_epoch(train_loader,
#                     model,
#                     criterion, 
#                     optimizer, 
#                     scheduler,
#                     epoch, 
#                     step,
#                     logger, 
#                     config,
#                     writer):
#     '''
#     train model for one epoch
#     '''
#     # switch to train mode
#     model.train() 
 
#     loss_list = []

#     for iter, data in enumerate(train_loader):
#         step += iter
#         optimizer.zero_grad()
#         images, targets = data
#         images, targets = images.cuda(non_blocking=True).float(), targets.cuda(non_blocking=True).float()

#         out = model(images)
#         loss = criterion(out, targets)

#         loss.backward()
#         optimizer.step()
        
#         loss_list.append(loss.item())

#         now_lr = optimizer.state_dict()['param_groups'][0]['lr']

#         writer.add_scalar('loss', loss, global_step=step)

#         if iter % config.print_interval == 0:
#             log_info = f'train: epoch {epoch}, iter:{iter}, loss: {np.mean(loss_list):.4f}, lr: {now_lr}'
#             print(log_info)
#             logger.info(log_info)
#     scheduler.step() 
#     return step


# def val_one_epoch(test_loader,
#                     model,
#                     criterion, 
#                     epoch, 
#                     logger,
#                     config):
#     # switch to evaluate mode
#     model.eval()
#     preds = []
#     gts = []
#     loss_list = []
#     with torch.no_grad():
#         for data in tqdm(test_loader):
#             img, msk = data
#             img, msk = img.cuda(non_blocking=True).float(), msk.cuda(non_blocking=True).float()

#             out = model(img)
#             loss = criterion(out, msk)

#             loss_list.append(loss.item())
#             gts.append(msk.squeeze(1).cpu().detach().numpy())
#             if type(out) is tuple:
#                 out = out[0]
#             out = out.squeeze(1).cpu().detach().numpy()
#             preds.append(out) 

#     # if epoch % config.val_interval == 0:
#     #     preds = np.array(preds).reshape(-1)
#     #     gts = np.array(gts).reshape(-1)

#     #     y_pre = np.where(preds>=config.threshold, 1, 0)
#     #     y_true = np.where(gts>=0.5, 1, 0)

#     #     confusion = confusion_matrix(y_true, y_pre)
#     #     TN, FP, FN, TP = confusion[0,0], confusion[0,1], confusion[1,0], confusion[1,1] 

#     #     accuracy = float(TN + TP) / float(np.sum(confusion)) if float(np.sum(confusion)) != 0 else 0
#     #     sensitivity = float(TP) / float(TP + FN) if float(TP + FN) != 0 else 0
#     #     specificity = float(TN) / float(TN + FP) if float(TN + FP) != 0 else 0
#     #     f1_or_dsc = float(2 * TP) / float(2 * TP + FP + FN) if float(2 * TP + FP + FN) != 0 else 0
#     #     miou = float(TP) / float(TP + FP + FN) if float(TP + FP + FN) != 0 else 0

#     #     log_info = f'val epoch: {epoch}, loss: {np.mean(loss_list):.4f}, miou: {miou}, f1_or_dsc: {f1_or_dsc}, accuracy: {accuracy}, \
#     #             specificity: {specificity}, sensitivity: {sensitivity}, confusion_matrix: {confusion}'
#     #     print(log_info)
#     #     logger.info(log_info)

#     # else:
#     #     log_info = f'val epoch: {epoch}, loss: {np.mean(loss_list):.4f}'
#     #     print(log_info)
#     #     logger.info(log_info)

#     preds = np.array(preds).reshape(-1)
#     gts = np.array(gts).reshape(-1)

#     y_pre = np.where(preds>=config.threshold, 1, 0)
#     y_true = np.where(gts>=0.5, 1, 0)

#     confusion = confusion_matrix(y_true, y_pre)
#     TN, FP, FN, TP = confusion[0,0], confusion[0,1], confusion[1,0], confusion[1,1] 

#     accuracy = float(TN + TP) / float(np.sum(confusion)) if float(np.sum(confusion)) != 0 else 0
#     sensitivity = float(TP) / float(TP + FN) if float(TP + FN) != 0 else 0
#     specificity = float(TN) / float(TN + FP) if float(TN + FP) != 0 else 0
#     f1_or_dsc = float(2 * TP) / float(2 * TP + FP + FN) if float(2 * TP + FP + FN) != 0 else 0
#     miou = float(TP) / float(TP + FP + FN) if float(TP + FP + FN) != 0 else 0

#     log_info = f'val epoch: {epoch}, loss: {np.mean(loss_list):.4f}, miou: {miou}, f1_or_dsc: {f1_or_dsc}, accuracy: {accuracy}, \
#             specificity: {specificity}, sensitivity: {sensitivity}, confusion_matrix: {confusion}'
#     print(log_info)
#     logger.info(log_info)
    
#     # return np.mean(loss_list)
#     return float(np.mean(loss_list)), float(f1_or_dsc)


# class AverageMeter(object):
#     """Computes and stores the average and current value"""

#     def __init__(self):
#         self.reset()

#     def reset(self):
#         self.val = 0
#         self.avg = 0
#         self.sum = 0
#         self.count = 0

#     def update(self, val, n=1):
#         self.val = val
#         self.sum += val * n
#         self.count += n
#         self.avg = self.sum / self.count


# def test_one_epoch(test_loader,
#                     model,
#                     criterion,
#                     logger,
#                     config,
#                     test_data_name=None):
#     # switch to evaluate mode
#     model.eval()
#     preds = []
#     gts = []
#     loss_list = []
#     gpu_time_meter = AverageMeter()
#     with torch.no_grad():
#         for i, data in enumerate(tqdm(test_loader)):
#             img, msk = data
#             img, msk = img.cuda(non_blocking=True).float(), msk.cuda(non_blocking=True).float()

#             start_time = time.time()
#             out = model(img)
#             end_time = time.time()
#             gpu_time_meter.update(end_time - start_time, img.size(0))

#             loss = criterion(out, msk)

#             loss_list.append(loss.item())
#             msk = msk.squeeze(1).cpu().detach().numpy()
#             gts.append(msk)
#             if type(out) is tuple:
#                 out = out[0]
#             out = out.squeeze(1).cpu().detach().numpy()
#             preds.append(out) 
#             if i % config.save_interval == 0:
#                 save_imgs(img, msk, out, i, config.work_dir + 'outputs/', config.datasets, config.threshold, test_data_name=test_data_name)

#         preds = np.array(preds).reshape(-1)
#         gts = np.array(gts).reshape(-1)

#         y_pre = np.where(preds>=config.threshold, 1, 0)
#         y_true = np.where(gts>=0.5, 1, 0)

#         confusion = confusion_matrix(y_true, y_pre)
#         TN, FP, FN, TP = confusion[0,0], confusion[0,1], confusion[1,0], confusion[1,1] 

#         accuracy = float(TN + TP) / float(np.sum(confusion)) if float(np.sum(confusion)) != 0 else 0
#         sensitivity = float(TP) / float(TP + FN) if float(TP + FN) != 0 else 0
#         specificity = float(TN) / float(TN + FP) if float(TN + FP) != 0 else 0
#         f1_or_dsc = float(2 * TP) / float(2 * TP + FP + FN) if float(2 * TP + FP + FN) != 0 else 0
#         miou = float(TP) / float(TP + FP + FN) if float(TP + FP + FN) != 0 else 0

#         if test_data_name is not None:
#             log_info = f'test_datasets_name: {test_data_name}'
#             print(log_info)
#             logger.info(log_info)
#         log_info = f'test of best model, loss: {np.mean(loss_list):.4f},miou: {miou}, f1_or_dsc: {f1_or_dsc}, accuracy: {accuracy}, \
#                 specificity: {specificity}, sensitivity: {sensitivity}, confusion_matrix: {confusion}'
#         log_info += f', GPU_time: {gpu_time_meter.avg:.4f} sec per image'
#         print(log_info)
#         logger.info(log_info)

#     return np.mean(loss_list)


import numpy as np
from tqdm import tqdm
import torch
from torch.cuda.amp import autocast as autocast
from sklearn.metrics import confusion_matrix
from utils import save_imgs
import os
from PIL import Image

def save_pred_mask_only(pred_np, idx, save_dir, threshold=0.5):
    """
    pred_np: numpy array from model (H, W) or (1, H, W)
    saves a binary 0/255 PNG mask as <idx>.png
    """
    os.makedirs(save_dir, exist_ok=True)

    # squeeze channel dimension if present
    if pred_np.ndim == 3 and pred_np.shape[0] == 1:
        pred_np = pred_np[0]

    # binarize
    mask_bin = (pred_np >= threshold).astype(np.uint8) * 255  # 0 or 255

    # convert to image and save as PNG (lossless, "high quality")
    im = Image.fromarray(mask_bin)
    # print("Saving mask:", os.path.join(save_dir, f"{idx:04d}_mask.png"))
    # im.save(os.path.join(save_dir, f"{idx:04d}_mask.png"))

    save_path = os.path.join(save_dir, f"{idx}.png")
    print("Saving mask:", save_path)
    im.save(save_path)


def train_one_epoch(train_loader,
                    model,
                    criterion, 
                    optimizer, 
                    scheduler,
                    epoch, 
                    logger, 
                    config, 
                    scaler=None):
    '''
    train model for one epoch
    '''
    # switch to train mode
    model.train() 
 
    loss_list = []

    for iter, data in enumerate(train_loader):
        optimizer.zero_grad()
        images, targets = data
        images, targets = images.cuda(non_blocking=True).float(), targets.cuda(non_blocking=True).float()
        if config.amp:
            with autocast():
                out = model(images)
                loss = criterion(out, targets)      
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            out = model(images)
            loss = criterion(out, targets)
            loss.backward()
            optimizer.step()
        
        loss_list.append(loss.item())

        now_lr = optimizer.state_dict()['param_groups'][0]['lr']
        if iter % config.print_interval == 0:
            log_info = f'train: epoch {epoch}, iter:{iter}, loss: {np.mean(loss_list):.4f}, lr: {now_lr}'
            print(log_info)
            logger.info(log_info)
    scheduler.step() 


# def val_one_epoch(test_loader,
#                     model,
#                     criterion, 
#                     epoch, 
#                     logger,
#                     config):
#     # switch to evaluate mode
#     model.eval()
#     preds = []
#     gts = []
#     loss_list = []
#     with torch.no_grad():
#         for data in tqdm(test_loader):
#             img, msk = data
#             img, msk = img.cuda(non_blocking=True).float(), msk.cuda(non_blocking=True).float()
#             out = model(img)
#             loss = criterion(out, msk)
#             loss_list.append(loss.item())
#             gts.append(msk.squeeze(1).cpu().detach().numpy())
#             if type(out) is tuple:
#                 out = out[0]
#             out = out.squeeze(1).cpu().detach().numpy()
#             preds.append(out) 

#     if epoch % config.val_interval == 0:
#         preds = np.array(preds).reshape(-1)
#         gts = np.array(gts).reshape(-1)

#         y_pre = np.where(preds>=config.threshold, 1, 0)
#         y_true = np.where(gts>=0.5, 1, 0)

#         confusion = confusion_matrix(y_true, y_pre)
#         TN, FP, FN, TP = confusion[0,0], confusion[0,1], confusion[1,0], confusion[1,1] 

#         accuracy = float(TN + TP) / float(np.sum(confusion)) if float(np.sum(confusion)) != 0 else 0
#         sensitivity = float(TP) / float(TP + FN) if float(TP + FN) != 0 else 0
#         specificity = float(TN) / float(TN + FP) if float(TN + FP) != 0 else 0
#         f1_or_dsc = float(2 * TP) / float(2 * TP + FP + FN) if float(2 * TP + FP + FN) != 0 else 0
#         miou = float(TP) / float(TP + FP + FN) if float(TP + FP + FN) != 0 else 0

#         log_info = (
#             f'val epoch: {epoch}, loss: {np.mean(loss_list):.4f}, '
#             f'iou: {miou:.4f}, dice: {f1_or_dsc:.4f}, accuracy: {accuracy:.4f}, '
#             f'specificity: {specificity:.4f}, sensitivity: {sensitivity:.4f}, '
#             f'confusion_matrix: {confusion}'
#         )
#         print(log_info)
#         logger.info(log_info)

#         # log_info = f'val epoch: {epoch}, loss: {np.mean(loss_list):.4f}, miou: {miou}, f1_or_dsc: {f1_or_dsc}, accuracy: {accuracy}, \
#         #         specificity: {specificity}, sensitivity: {sensitivity}, confusion_matrix: {confusion}'
#         # print(log_info)
#         # logger.info(log_info)

#     else:
#         log_info = f'val epoch: {epoch}, loss: {np.mean(loss_list):.4f}'
#         print(log_info)
#         logger.info(log_info)
    
#     return np.mean(loss_list)


def val_one_epoch(test_loader,
                  model,
                  criterion, 
                  epoch, 
                  logger,
                  config):
    model.eval()
    preds, gts, loss_list = [], [], []

    with torch.no_grad():
        for data in tqdm(test_loader):
            img, msk = data
            img, msk = img.cuda(non_blocking=True).float(), msk.cuda(non_blocking=True).float()
            out = model(img)
            loss = criterion(out, msk)
            loss_list.append(loss.item())

            if isinstance(out, tuple):
                out = out[0]
            gts.append(msk.squeeze(1).cpu().numpy())
            preds.append(out.squeeze(1).cpu().numpy())
            

    # ---- compute metrics every time ----
    preds = np.array(preds).reshape(-1)
    gts   = np.array(gts).reshape(-1)

    y_pre  = np.where(preds >= config.threshold, 1, 0)
    y_true = np.where(gts   >= 0.5,            1, 0)

    confusion = confusion_matrix(y_true, y_pre, labels=[0,1])
    TN, FP, FN, TP = confusion[0,0], confusion[0,1], confusion[1,0], confusion[1,1]

    accuracy    = float(TN + TP) / float(np.sum(confusion)) if float(np.sum(confusion)) != 0 else 0.0
    sensitivity = float(TP) / float(TP + FN) if float(TP + FN) != 0 else 0.0
    specificity = float(TN) / float(TN + FP) if float(TN + FP) != 0 else 0.0
    dice        = float(2 * TP) / float(2 * TP + FP + FN) if float(2 * TP + FP + FN) != 0 else 0.0
    iou         = float(TP) / float(TP + FP + FN) if float(TP + FP + FN) != 0 else 0.0

    mean_loss = float(np.mean(loss_list))

    # (optional) keep your val_interval just for printing frequency
    if (epoch % getattr(config, "val_interval", 1)) == 0:
        log_info = (f'val epoch: {epoch}, loss: {mean_loss:.4f}, '
                    f'iou: {iou:.4f}, dice: {dice:.4f}, accuracy: {accuracy:.4f}, '
                    f'specificity: {specificity:.4f}, sensitivity: {sensitivity:.4f}, '
                    f'confusion_matrix: {confusion}')
        print(log_info)
        logger.info(log_info)

    # ---- return BOTH loss and dice ----
    return mean_loss, dice


def test_one_epoch(test_loader,
                    model,
                    criterion,
                    logger,
                    config,
                    test_data_name=None):
    # switch to evaluate mode
    model.eval()
    preds = []
    gts = []
    loss_list = []
    print("#----------Testing----------#")
    with torch.no_grad():
        for i, data in enumerate(tqdm(test_loader)):
            # img, msk = data
            img, msk, filename = data
            img, msk = img.cuda(non_blocking=True).float(), msk.cuda(non_blocking=True).float()
            out = model(img)
            loss = criterion(out, msk)
            loss_list.append(loss.item())
            msk = msk.squeeze(1).cpu().detach().numpy()
            gts.append(msk)
            if type(out) is tuple:
                out = out[0]
            out = out.squeeze(1).cpu().detach().numpy()
            preds.append(out) 
            print(f"Saving prediction mask for index: {i}")
            # save_imgs(img, msk, out, i, config.work_dir + 'outputs/', config.datasets, config.threshold, test_data_name=test_data_name)
            save_dir = os.path.join(config.work_dir, "pred_masks")
            os.makedirs(save_dir, exist_ok=True)
            save_pred_mask_only(
                pred_np=out,          # numpy (H, W) or (1, H, W)
                idx=filename[0],
                save_dir=save_dir,
                threshold=config.threshold,
            )

        print("#----------Calculating metrics----------#")
        preds = np.array(preds).reshape(-1)
        gts = np.array(gts).reshape(-1)

        y_pre = np.where(preds>=config.threshold, 1, 0)
        y_true = np.where(gts>=0.5, 1, 0)

        confusion = confusion_matrix(y_true, y_pre)
        TN, FP, FN, TP = confusion[0,0], confusion[0,1], confusion[1,0], confusion[1,1] 

        accuracy = float(TN + TP) / float(np.sum(confusion)) if float(np.sum(confusion)) != 0 else 0
        sensitivity = float(TP) / float(TP + FN) if float(TP + FN) != 0 else 0
        specificity = float(TN) / float(TN + FP) if float(TN + FP) != 0 else 0
        f1_or_dsc = float(2 * TP) / float(2 * TP + FP + FN) if float(2 * TP + FP + FN) != 0 else 0
        miou = float(TP) / float(TP + FP + FN) if float(TP + FP + FN) != 0 else 0

        if test_data_name is not None:
            log_info = f'test_datasets_name: {test_data_name}'
            print(log_info)
            logger.info(log_info)
        # log_info = f'test of best model, loss: {np.mean(loss_list):.4f},miou: {miou}, f1_or_dsc: {f1_or_dsc}, accuracy: {accuracy}, \
        #         specificity: {specificity}, sensitivity: {sensitivity}, confusion_matrix: {confusion}'
        # print(log_info)
        # logger.info(log_info)
        log_info = (
            f'test of best model, loss: {np.mean(loss_list):.4f}, '
            f'iou: {miou:.4f}, dice: {f1_or_dsc:.4f}, accuracy: {accuracy:.4f}, '
            f'specificity: {specificity:.4f}, sensitivity: {sensitivity:.4f}, '
            f'confusion_matrix: {confusion}'
        )
        print(log_info)
        logger.info(log_info)


    return np.mean(loss_list)
