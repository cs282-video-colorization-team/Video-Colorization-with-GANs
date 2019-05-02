import os
import torch
from torchvision import transforms
from torch.utils import data
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

class Movie(Dataset):
    def __init__(self, image_path, transform, mode):
        '''
        mode: train or test
        cls_list: 'all' or list of classes, e.g. ['Drama','Horror',...]
        '''
        self.image_path = image_path
        self.transform = transform
        self.mode = mode

        self.data_files_name = []

        file_path = self.image_path
        self.data_files_name = [file for file in os.listdir(file_path) if os.path.splitext(file)[1] == '.png']

    def __getitem__(self, index):
        filename = os.path.join(self.image_path, self.data_files_name[index])
        color_img = Image.open(filename)
        gray_img = Image.open(filename).convert('LA')

        if self.mode == 'train' or self.mode == 'val':
            if self.transform:
                gray_img, color_img = self.transform(gray_img), self.transform(color_img)
            return gray_img, color_img
        else:
            if self.transform:
                gray_img = self.transform(gray_img)
            return gray_img

    def __len__(self):
        return len(self.data_files_name)

# def get_loader(image_path, batch_size=16, mode='train', num_workers=1):

#     transform_gray = []
#     transform_gray.append(transforms.Grayscale())
#     transform_gray.append(transforms.ToTensor())
#     #transform_gray.append(transforms.Normalize(mean=(0.5), std=(0.5)))
#     transform_gray = transforms.Compose(transform_gray)

#     if mode == 'train':
#         transform_color = []
#         transform_color.append(transforms.ToTensor())
#         transform_color.append(transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
#         transform_color = transforms.Compose(transform_color)
#     else:
#         transform_color = None

#     dataset = Movie(image_path, transform_color, transform_gray, mode)

#     data_loader = data.DataLoader(dataset=dataset,
#                                   batch_size=batch_size,
#                                   shuffle=(mode=='train'),
#                                   num_workers=num_workers)

#     return data_loader
