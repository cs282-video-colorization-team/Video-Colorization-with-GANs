import os
import torch
from torchvision import transforms
from torch.utils import data
from torch.utils.data import Dataset
from PIL import Image

class MovieTime(Dataset):
    def __init__(self, image_path, transform_color, transform_gray, mode):
        self.image_path = image_path
        self.transform_color = transform_color
        self.transform_gray = transform_gray
        self.IMAGE_RESIZE = (480, 480)
        self.mode = mode

        self.data_files_name = [file for file in os.listdir(self.image_path) if os.path.splitext(file)[1] == '.png']

    def __getitem__(self, index):

        now_file_name = self.data_files_name[index]
        now_timestamp = int(now_file_name.split('.')[0])

        if now_timestamp > 0:
            prev_timestamp_str = str(now_timestamp - 1)
            prev_file_name = '0' * (5-len(prev_timestamp_str)) + prev_timestamp_str + '.png'
        else:
            prev_file_name = now_file_name

        if now_timestamp < len(self.data_files_name) - 1:
            next_timestamp_str = str(now_timestamp + 1)
            next_file_name = '0' * (5-len(next_timestamp_str)) + next_timestamp_str + '.png'
        else:
            next_file_name = now_file_name

        # _now, _prev, _next
        image_now = Image.open(os.path.join(self.image_path + now_file_name))
        image_now = image_now.resize(self.IMAGE_RESIZE)

        image_prev = Image.open(os.path.join(self.image_path + prev_file_name))
        image_prev = image_prev.resize(self.IMAGE_RESIZE)

        image_next = Image.open(os.path.join(self.image_path + next_file_name))
        image_next = image_next.resize(self.IMAGE_RESIZE)

        if self.mode == 'train':
            return self.transform_gray(image_now), self.transform_gray(image_prev), self.transform_gray(image_next), self.transform_color(image)
        elif self.mode == 'val':
            return self.transform_gray(image_now), self.transform_gray(image_prev), self.transform_gray(image_next), self.transform_color(image)
        elif self.mode == 'test':
            return self.transform_gray(image_now), self.transform_gray(image_prev), self.transform_gray(image_next)

    def __len__(self):
        return len(self.data_files_name)

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