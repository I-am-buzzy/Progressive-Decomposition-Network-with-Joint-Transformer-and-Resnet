from torch.utils.data.dataset import Dataset
from PIL import Image
import torchvision
import torchvision.transforms.functional as TF
import os
import torch
import torch.nn.functional as F
import fnmatch
import numpy as np
import random
from utils import randrot, randfilp
from natsort import natsorted


class MSRSData(torch.utils.data.Dataset):  #Load dataset with infrared folder path and visible folder path  TODO: remove ground truth reference
    def __init__(self, opts, is_train=True, crop=lambda x: x):
        super(MSRSData, self).__init__()
        self.is_train = is_train
        if is_train:
            self.vis_folder = os.path.join(opts.dataroot, 'train', 'vi')
            self.ir_folder = os.path.join(opts.dataroot, 'train', 'ir')
            self.label_folder = os.path.join(opts.dataroot, 'train', 'label')     #1083个
            self.bi_folder = os.path.join(opts.dataroot, 'train', 'bi')            
            self.bd_folder = os.path.join(opts.dataroot, 'train', 'bd')
            self.mask_folder = os.path.join(opts.dataroot, 'train', 'mask')
        else:
            self.vis_folder = os.path.join(opts.dataroot, 'test', 'vi')
            self.ir_folder = os.path.join(opts.dataroot, 'test', 'ir')
            self.label_folder = os.path.join(opts.dataroot, 'test', 'label')  #361个
            #self.label_folder = os.path.join(opts.dataroot, 'test', 'ccd')   #自己加的为了算出CCD的语义分割
        self.crop = torchvision.transforms.RandomCrop(256)   #随机裁剪成256*256
        # gain infrared and visible images list
        self.ir_list = natsorted(os.listdir(self.label_folder))     #python自然排序natsort 从小到大排列
        print('len(self.ir_list):', len(self.ir_list))   #1083
        #self.ST = SpatialTransformer(self.crop.size[0],self.crop.size[0],False)

    def __getitem__(self, index):     #这个方法用于定义对象的索引操作，即通过索引值获取对象中的元素
        # gain image path
        image_name = self.ir_list[index]      #索引0，1，2、、、、、1083
        vis_path = os.path.join(self.vis_folder, image_name)
        ir_path = os.path.join(self.ir_folder, image_name)
        label_path = os.path.join(self.label_folder, image_name)        
        # read image as type Tensor
        vis = self.imread(path=vis_path)   #torch.Size([1, 3, 480, 640])
        ir = self.imread(path=ir_path, vis_flage=False)         #代码在下面   #torch.Size([1, 1, 480, 640])
        label = self.imread(path=label_path, label=True)        #torch.Size([1, 1, 480, 640])
        if self.is_train:
            bi_path = os.path.join(self.bi_folder, image_name)
            bd_path = os.path.join(self.bd_folder, image_name)
            mask_path = os.path.join(self.mask_folder, image_name)
            bi = self.imread(path=bi_path, label=True)            #torch.Size([1, 1, 480, 640])
            bd = self.imread(path=bd_path, label=True)            #torch.Size([1, 1, 480, 640])
            mask = self.imread(path=mask_path, vis_flage=False)       #torch.Size([1, 1, 480, 640])
        
        if self.is_train:         # 训练图像进行一定的数据增强，包括翻转，旋转，以及随机裁剪等
            vis_ir = torch.cat([vis, ir, label, bi, bd, mask], dim=1)        #torch.Size([1, 8, 480, 640])
            if vis_ir.shape[-1] <= 256 or vis_ir.shape[-2] <= 256:
                vis_ir = TF.resize(vis_ir, 256)
            vis_ir = randfilp(vis_ir)       #翻转
            vis_ir = randrot(vis_ir)
            patch = self.crop(vis_ir)      #裁剪成256*256

            vis, ir, label, bi, bd, mask = torch.split(patch, [3, 1, 1, 1, 1, 1], dim=1)    #torch.split()作用将tensor分成块结构
            label = label.type(torch.LongTensor)
            bi = bi / 255.0
            bd = bd / 255.0
            bi = bi.type(torch.LongTensor)
            bd = bd.type(torch.LongTensor)
            # ir 单通道, vis RGB三通道 torch.Size([1, 256, 256])，torch.Size([3, 256, 256])
            return ir.squeeze(0), vis.squeeze(0), label.squeeze(0), bi.squeeze(0), bd.squeeze(0), mask.squeeze(0)  #三通道
        else: 
            label = label.type(torch.LongTensor)
            return ir.squeeze(0), vis.squeeze(0), label.squeeze(0), image_name

    def __len__(self):
        return len(self.ir_list)


    @staticmethod
    def imread(path, label=False, vis_flage=True):
        if label:
            img = Image.open(path)
            im_ts = TF.to_tensor(img).unsqueeze(0) * 255
        else:
            if vis_flage:   # visible images; RGB channel
                img = Image.open(path).convert('RGB')   #彩色图像
                im_ts = TF.to_tensor(img).unsqueeze(0)   #unsqueeze()函数起升维的作用,参数表示在哪个地方加一个维度。
            else:           # infrared images single channel
                img = Image.open(path).convert('L')    #灰度图像
                im_ts = TF.to_tensor(img).unsqueeze(0)
        return im_ts
    

class FusionData(torch.utils.data.Dataset):     #Load dataset with infrared folder path and visible folder path
    def __init__(self, opts, crop=lambda x: x):
        super(FusionData, self).__init__()          
        #self.vis_folder = os.path.join(opts.dataroot, 'test', 'vi')
        #self.ir_folder = os.path.join(opts.dataroot, 'test', 'ir')
        self.vis_folder = os.path.join(opts.dataroot, 'vi')
        self.ir_folder = os.path.join(opts.dataroot, 'ir')
        #self.fuse_folder = os.path.join(opts.dataroot, 'fuse')

        # self.vis_folder = os.path.join('/data/timer/Segmentation/SegNext/datasets/MSRS/RGB')
        # self.ir_folder = os.path.join('/data/timer/Segmentation/SegNext/datasets/MSRS/Thermal')
        self.ir_list = natsorted(os.listdir(self.ir_folder))
        print(len(self.ir_list))

    def __getitem__(self, index):
        # gain image path
        image_name = self.ir_list[index]
        vis_path = os.path.join(self.vis_folder, image_name)
        ir_path = os.path.join(self.ir_folder, image_name)
        #fuse_path = os.path.join(self.fuse_folder, image_name)
        # read image as type Tensor
        vis, w, h = self.imread(path=vis_path)
        ir, w, h = self.imread(path=ir_path, vis_flage=False)
        #fuse, w, h = self.imread(path=fuse_path, vis_flage=False)
        return ir.squeeze(0), vis.squeeze(0), image_name, w, h

    def __len__(self):
        return len(self.ir_list)


    @staticmethod
    def imread(path, label=False, vis_flage=True):
        if label:
            img = Image.open(path)
            # 获取图像大小
            width, height = img.size

            # 调整图像大小到32的倍数
            new_width = width - (width % 32)
            new_height = height - (height % 32)
            img = img.resize((new_width, new_height))
            im_ts = TF.to_tensor(img).unsqueeze(0) * 255
        else:
            if vis_flage:    # visible images; RGB channel
                img = Image.open(path).convert('RGB')
                # 获取图像大小
                width, height = img.size

                # 调整图像大小到32的倍数
                new_width = width - (width % 32)
                new_height = height - (height % 32)
                img = img.resize((new_width, new_height))
                im_ts = TF.to_tensor(img).unsqueeze(0)
            else:   # infrared images single channel
                img = Image.open(path).convert('L') 
                # 获取图像大小
                width, height = img.size
                # 调整图像大小到32的倍数
                new_width = width - (width % 32)
                new_height = height - (height % 32)
                img = img.resize((new_width, new_height))
                im_ts = TF.to_tensor(img).unsqueeze(0)
        return im_ts, width, height
