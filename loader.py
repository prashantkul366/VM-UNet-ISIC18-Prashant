from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
import random
import os
from PIL import Image
from einops.layers.torch import Rearrange
from scipy.ndimage.morphology import binary_dilation
from torch.utils.data import Dataset
from torchvision import transforms
from scipy import ndimage
from utils import *
import cv2
from torchvision.transforms import functional as TF
from torchvision.transforms import InterpolationMode

from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import numpy as np


def _open_image_with_tifffile(path):
    """
    Robust open for images (handles many TIFF flavours).
    Returns a PIL.Image in RGB (for images) or L (if caller later converts).
    """
    try:
        # try tifffile first (best for tiffs)
        import tifffile
        arr = tifffile.imread(path)  # numpy array (H,W) or (H,W,3) or (H,W,4)
        if arr is None:
            raise RuntimeError("tifffile returned None")
        # If grayscale, convert to 3-channel
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        # If more than 3 channels (e.g., RGBA), drop extra channels
        if arr.shape[-1] > 3:
            arr = arr[..., :3]
        return Image.fromarray(arr).convert("RGB")
    except Exception:
        # fallback: try imageio.v3
        try:
            import imageio.v3 as iio
            arr = iio.imread(path)
            if arr.ndim == 2:
                arr = np.stack([arr, arr, arr], axis=-1)
            if arr.shape[-1] > 3:
                arr = arr[..., :3]
            return Image.fromarray(arr).convert("RGB")
        except Exception:
            # fallback: try cv2
            try:
                import cv2
                im_cv = cv2.imread(path, cv2.IMREAD_UNCHANGED)
                if im_cv is None:
                    raise RuntimeError("cv2.imread returned None")
                # cv2 loads BGR; convert to RGB
                if im_cv.ndim == 2:
                    im_cv = np.stack([im_cv, im_cv, im_cv], axis=-1)
                else:
                    im_cv = cv2.cvtColor(im_cv, cv2.COLOR_BGR2RGB)
                return Image.fromarray(im_cv).convert("RGB")
            except Exception as e:
                raise RuntimeError(f"All readers failed for {path}: {e}")


# ===== normalize over the dataset 
def dataset_normalized(imgs):
    imgs_normalized = np.empty(imgs.shape)
    imgs_std = np.std(imgs)
    imgs_mean = np.mean(imgs)
    imgs_normalized = (imgs-imgs_mean)/imgs_std
    for i in range(imgs.shape[0]):
        imgs_normalized[i] = ((imgs_normalized[i] - np.min(imgs_normalized[i])) / (np.max(imgs_normalized[i])-np.min(imgs_normalized[i])))*255
    return imgs_normalized


## Temporary
class isic_loader(Dataset):
    """ dataset class for Brats datasets
    """
    def __init__(self, path_Data, train = True, Test = False):
        super(isic_loader, self)
        self.train = train
        if train:
          self.data   = np.load(path_Data+'data_train.npy')
          self.mask   = np.load(path_Data+'mask_train.npy')
        else:
          if Test:
            self.data   = np.load(path_Data+'data_test.npy')
            self.mask   = np.load(path_Data+'mask_test.npy')
          else:
            self.data   = np.load(path_Data+'data_val.npy')
            self.mask   = np.load(path_Data+'mask_val.npy')          
        
        self.data   = dataset_normalized(self.data)
        self.mask   = np.expand_dims(self.mask, axis=3)
        self.mask   = self.mask/255.

    def __getitem__(self, indx):
        img = self.data[indx]
        seg = self.mask[indx]
        if self.train:
            if random.random() > 0.5:
                img, seg = self.random_rot_flip(img, seg)
            if random.random() > 0.5:
                img, seg = self.random_rotate(img, seg)
        
        seg = torch.tensor(seg.copy())
        img = torch.tensor(img.copy())
        img = img.permute( 2, 0, 1)
        seg = seg.permute( 2, 0, 1)

        return img, seg
    
    def random_rot_flip(self,image, label):
        k = np.random.randint(0, 4)
        image = np.rot90(image, k)
        label = np.rot90(label, k)
        axis = np.random.randint(0, 2)
        image = np.flip(image, axis=axis).copy()
        label = np.flip(label, axis=axis).copy()
        return image, label
    
    def random_rotate(self,image, label):
        angle = np.random.randint(20, 80)
        image = ndimage.rotate(image, angle, order=0, reshape=False)
        label = ndimage.rotate(label, angle, order=0, reshape=False)
        return image, label


               
    def __len__(self):
        return len(self.data)
    


def _to_tensor(img_np):  # HWC [0,1] -> CHW float32
    t = torch.from_numpy(img_np.astype(np.float32))
    if t.ndim == 2:        # H W  -> 1 H W
        t = t.unsqueeze(0)
    else:                  # H W C -> C H W
        t = t.permute(2, 0, 1)
    return t

class Dataset(Dataset):
    """
    Generic file-based loader for roots like:
      root/
        train/ images/, masks/
        val/   images/, masks/
        test/  images/, masks/
    Works with both ISIC-style and BUSI-style folders.
    """
    def __init__(self, root, split="train",
             images_dir="images", masks_dir="masks",
             train_augs=True,
             target_size=(256, 256)):     
        super().__init__()
        super().__init__()
        self.root = root
        self.split = split
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.train_augs = train_augs and (split == "train")
        self.target_size = target_size

        base = os.path.join(root, split)
        
        img_dir = os.path.join(base, images_dir)
        msk_dir = os.path.join(base, masks_dir)
        print(f"Loading {split} data from {base}...")
        print(f"  Images: {img_dir} with length {len(os.listdir(img_dir))} files")
        print(f"  Masks:  {msk_dir} with length {len(os.listdir(msk_dir))} files")
        if not os.path.isdir(img_dir):
            raise FileNotFoundError(f"Images folder not found: {img_dir}")
        if not os.path.isdir(msk_dir):
            raise FileNotFoundError(f"Masks folder not found: {msk_dir}")

        # match by stem
        def stems(d):
            return {os.path.splitext(f)[0]: f
                    for f in os.listdir(d)
                    if f.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff"))}

        imgs = stems(img_dir)
        msks = stems(msk_dir)
        common = sorted(list(set(imgs.keys()) & set(msks.keys())))
        if len(common) == 0:
            raise RuntimeError(f"No matching image/mask pairs in {img_dir} and {msk_dir}")

        self.pairs = [(os.path.join(img_dir, imgs[k]), os.path.join(msk_dir, msks[k]))
                      for k in common]

    def __len__(self):
        return len(self.pairs)

    def _random_rot_flip(self, image, label):
        k = np.random.randint(0, 4)
        image = np.rot90(image, k)
        label = np.rot90(label, k)
        axis = np.random.randint(0, 2)
        image = np.flip(image, axis=axis).copy()
        label = np.flip(label, axis=axis).copy()
        return image, label

    def _random_rotate(self, image, label):
        angle = np.random.randint(20, 80)  # keep your original range
        image = ndimage.rotate(image, angle, order=1, reshape=False)  # bilinear for image
        label = ndimage.rotate(label, angle, order=0, reshape=False)  # nearest for mask
        return image, label

    def __getitem__(self, idx):
        img_path, msk_path = self.pairs[idx]

        # Load PIL
        # img = Image.open(img_path).convert("RGB")
        img = _open_image_with_tifffile(img_path)  
        # img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        # msk = Image.open(msk_path).convert("L")
        msk = _open_image_with_tifffile(msk_path).convert("L")

        # --- RESIZE to fixed size ---
        H, W = self.target_size
        img = TF.resize(img, (H, W), interpolation=InterpolationMode.BILINEAR, antialias=True)
        msk = TF.resize(msk, (H, W), interpolation=InterpolationMode.NEAREST)

        # To numpy, [0,1]
        img = np.asarray(img, dtype=np.float32) / 255.0
        msk = (np.asarray(msk, dtype=np.float32) > 127).astype(np.float32)

        # Augs (keep simple to avoid shape changes)
        if self.train_augs:
            if random.random() > 0.5:
                img = np.ascontiguousarray(np.flip(img, axis=1))
                msk = np.ascontiguousarray(np.flip(msk, axis=1))

        # To tensors
        img_t = torch.from_numpy(img).permute(2, 0, 1).contiguous()   # [3,H,W]
        msk_t = torch.from_numpy(msk).unsqueeze(0).contiguous()       # [1,H,W]

        img_name = os.path.basename(img_path).split('.')[0]

        # return img_t, msk_t , img_name
        return img_t, msk_t

