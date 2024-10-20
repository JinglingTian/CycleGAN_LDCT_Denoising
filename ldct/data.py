import torch
from torch.utils.data import Dataset
import os
import numpy as np
from torchvision import transforms
from torchvision.transforms import functional as tvF
from torchvision.io import read_image
from .utils import min_max_norm,otsu

# 配对数据集
class LDCT_Dataset(Dataset):
    def __init__(self,
                 fd_path,
                 ld_path,
                 resize = None,
                 crop_size=None,
                 otsu=False
                 ):
        super().__init__()
        self.fd_imgs = [os.path.join(fd_path,f) for f in os.listdir(fd_path)]
        self.ld_imgs = [os.path.join(ld_path,f) for f in os.listdir(ld_path)]
        self.fd_imgs.sort()
        self.ld_imgs.sort()
        assert len(self.fd_imgs)==len(self.ld_imgs)

        self.crop_size = crop_size
        self.resize = resize
        self.otsu = otsu
    def __len__(self):
        return len(self.fd_imgs)
    
    def get_img_mask(self,fd_path,ld_path):
        img_fd = np.load(fd_path)
        img_ld = np.load(ld_path)

        if self.otsu:
            mask = otsu(img_fd)
            img_fd = img_fd*mask
            img_ld = img_ld*mask

        img_fd = torch.tensor(img_fd).unsqueeze(0).float()
        img_ld = torch.tensor(img_ld).unsqueeze(0).float()
        # 更改尺寸
        if self.resize is not None:
            img_fd = tvF.resize(img_fd,self.resize,antialias=True)
            img_ld = tvF.resize(img_ld,self.resize,antialias=True)

        # 随机剪裁
        if self.crop_size is not None:
            crop_params = transforms.RandomCrop.get_params(img_fd, self.crop_size)
            img_fd = tvF.crop(img_fd,*crop_params)
            img_ld = tvF.crop(img_ld,*crop_params)

        # 归一化
        img_fd, img_ld = min_max_norm(img_fd), min_max_norm(img_ld)


        return img_fd,img_ld
    
    def __getitem__(self, index):
        fd_path = self.fd_imgs[index]
        ld_path = self.ld_imgs[index]
        img_fd,img_ld = self.get_img_mask(fd_path,ld_path)
        return img_ld.float(),img_fd.float()



# 非配对数据集
class LDCT_CycleGan_Dataset(Dataset):
    def __init__(self,
                 fd_path,
                 ld_path,
                 resize = None,
                 crop_size=None,
                 shuffile=False
                 ):
        super().__init__()
        self.fd_imgs = [os.path.join(fd_path,f) for f in os.listdir(fd_path)]
        self.ld_imgs = [os.path.join(ld_path,f) for f in os.listdir(ld_path)]
        self.fd_imgs.sort()
        self.ld_imgs.sort()
        assert len(self.fd_imgs)==len(self.ld_imgs)

        self.resize = resize
        self.crop_size = crop_size
        self.shuffile = shuffile

    def __len__(self):
        return len(self.fd_imgs)

    def get_img(self,img_path):
        img = torch.tensor(np.load(img_path)).unsqueeze(0).float()
        # 更改尺寸
        if self.resize is not None:
            img = tvF.resize(img,self.resize,antialias=True)
        # 随机剪裁
        if self.crop_size is not None:
            crop_params = transforms.RandomCrop.get_params(img, self.crop_size)
            img = tvF.crop(img,*crop_params)
        # 归一化
        img = min_max_norm(img)
        return img
    
    def __getitem__(self, index):
        fd_path = self.fd_imgs[index]
        ld_path = self.ld_imgs[index]
        if self.shuffile:
            fd_path = np.random.choice(self.fd_imgs)
            ld_path = np.random.choice(self.ld_imgs)
        img_fd,img_ld = self.get_img(fd_path),self.get_img(ld_path)
        return img_ld.float(),img_fd.float()
