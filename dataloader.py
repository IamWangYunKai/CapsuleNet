#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#import torch
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset

IMAGE_WIDTH = 128
IMAGE_HEIGHT = 128

class MyDataset(Dataset):
    def __init__(self, file_path=None, if_valid=False):
        self.data_path = '/home/wang/DataSet/yq_road/'
        self.file_path = file_path
        see_frames, unsee_frames, road_types = self.read_data(self.file_path)
        
        if if_valid:
            transforms_ = [transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH), Image.BICUBIC),
                           transforms.ToTensor(),
                           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                           ]
        else:
            transforms_ = [transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH), Image.BICUBIC),
                           transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                           transforms.ToTensor(),
                           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                           ]
        self.img_transformer = transforms.Compose(transforms_)
        
        self.ids = []
        self.type = []
        
        for i in range(len(see_frames)):
            for j in range(see_frames[i], unsee_frames[i]):
                self.ids.append(j)
                self.type.append(road_types[i])
                
        self.dataLen = len(self.ids)
        #print('data len:', self.dataLen)
        
    def __getitem__(self, index):
        img_path = self.data_path+str(self.ids[index])+'.jpg'
        label = self.type[index]
        img = Image.open(img_path)
        img = img.convert("RGB")
        img = self.img_transformer(img)
        return (img, label)

    def __len__(self):
        return self.dataLen
    
    def read_data(self, file_name):
        see_frames = []
        unsee_frames = []
        road_types = []
        with open(file_name, 'r') as file:
            lines = file.readlines()[1:]
            for line in lines:
                spline = line.split()
                see_frames.append(int(spline[0]))
                unsee_frames.append(int(spline[1]))
                road_types.append(int(spline[3]))
        return see_frames, unsee_frames, road_types

if __name__ == '__main__':
    ds = MyDataset(file_path='animation.txt')
    dataloader = DataLoader(dataset=ds,
                            batch_size=4,
                            shuffle=True,
                            num_workers=16)
    
    for index, (x, y) in enumerate(dataloader):
        print(x.shape, y.shape)