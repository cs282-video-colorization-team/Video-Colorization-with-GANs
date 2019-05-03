import os
import torch
from torchvision import transforms
from torch.utils import data
from torch.utils.data import Dataset
from PIL import Image

class Movie(Dataset):
    def __init__(self, image_path, large, transform_color, transform_gray, mode):
        '''
        mode: train or test
        cls_list: 'all' or list of classes, e.g. ['Drama','Horror',...]
        '''
        if large:
            self.IMAGE_SIZE = (480, 480)
        else:
            self.IMAGE_SIZE = (224, 224)
        self.image_path = image_path
        self.transform_color = transform_color
        self.transform_gray = transform_gray
        self.mode = mode

        self.data_files_name = []

        file_path = self.image_path
        self.data_files_name = [file for file in os.listdir(file_path) if os.path.splitext(file)[1] == '.png']

    def __getitem__(self, index):
        image = Image.open(os.path.join(self.image_path + self.data_files_name[index]))
        # image.thumbnail(self.IMAGE_SIZE)
        image = image.resize(self.IMAGE_SIZE)
        #print (image, os.path.join(self.image_path + self.data_files_name[index]))

        if self.mode == 'train':
            print (self.transform_gray(image).shape, self.transform_color(image).shape)
            return self.transform_gray(image), self.transform_color(image)

        elif self.mode == 'val':
            print ('val', self.transform_gray(image).shape, self.transform_color(image).shape)
            return self.transform_gray(image), self.transform_color(image)
        elif self.mode == 'test':
            return self.transform_gray(image)

    def __len__(self):
        return len(self.data_files_name)

class MovieTime(Dataset):
    def __init__(self, image_path, transform_color, transform_gray, mode):

        self.image_path = image_path
        self.transform_color = transform_color
        self.transform_gray = transform_gray
        self.IMAGE_RESIZE = (480, 480)
        self.mode = mode

        self.data_files_name = [file for file in os.listdir(self.file_path) if os.path.splitext(file)[1] == '.png']

    def __getitem__(self, index):

        now_file_name = self.data_files_name[index]

        timestamp = int(now_file_name.split('.')[0])

        if timestamp > 0:
            prev_timestamp_str = str(timestamp - 1)
            prev_file_name = '0' * (5-len(prev_timestamp_str)) + prev_timestamp_str + '.png'

        if timestamp < len(self.data_files_name) - 1:
            next_timestamp_str = str(timestamp + 1)
            next_file_name = '0' * (5-len(next_timestamp_str)) + next_timestamp_str + '.png'


        prev_file_name = self.data_files_name[index]
        next_file_name = self.data_files_name[index]

        image = Image.open(os.path.join(self.image_path + self.data_files_name[index]))
        image = image.resize(self.IMAGE_RESIZE)

        if self.mode == 'train':
            return self.transform_gray(image), self.transform_color(image)
        elif self.mode == 'val':
            return self.transform_gray(image), self.transform_color(image)
        elif self.mode == 'test':
            return self.transform_gray(image)

    def __len__(self):
        return len(self.data_files_name)


def get_loader(image_path, batch_size=16, large=True, mode='train', num_workers=1):
    # if large:
    #     crop = transforms.CenterCrop(480)
    # else:
    #     crop = transforms.CenterCrop(224)
    transform_gray = []
    transform_gray.append(transforms.Grayscale())
    # transform_gray.append(crop)
    transform_gray.append(transforms.ToTensor())
    transform_gray.append(transforms.Normalize(mean=[0.5], std=[0.5]))
    transform_gray = transforms.Compose(transform_gray)

    if mode == 'train' or mode == 'val':
        transform_color = []
        # transform_color.append(crop)
        transform_color.append(transforms.ToTensor())
        transform_color.append(transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
        transform_color = transforms.Compose(transform_color)
    else:
        transform_color = None

    dataset = Movie(image_path, large, transform_color, transform_gray, mode)

    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=(mode=='train'),
                                  num_workers=num_workers,
                                  drop_last=True)

    return data_loader


def get_movie_time_loader(image_path, batch_size=16, mode='train', num_workers=1):

    transform_gray = []
    transform_gray.append(transforms.Grayscale())
    transform_gray.append(transforms.ToTensor())
    transform_gray = transforms.Compose(transform_gray)

    if mode == 'train' or mode == 'val':
        transform_color = []
        transform_color.append(transforms.ToTensor())
        transform_color.append(transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
        transform_color = transforms.Compose(transform_color)
    else:
        transform_color = None

    dataset = MovieTime(image_path, transform_color, transform_gray, mode)

    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=(mode=='train'),
                                  num_workers=num_workers,
                                  drop_last=True)

    return data_loader