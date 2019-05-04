import torch
import os
from gan_model import *
from movie_data_loader import *
from movie_time_data_loader import *
from torch.autograd import Variable
from PIL import Image
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import torchvision
import argparse


parser = argparse.ArgumentParser(description='Test Colorization using GAN')
parser.add_argument('--path', type=str,
    help='Root path for dataset')
parser.add_argument('--savepath', type=str,
    help='Save Root path for dataset')
parser.add_argument('--modelpath', type=str,
    help='Model Root path')
parser.add_argument('--time', type=str,
    help='baseline or with time', choices=['baseline', 'time'])
parser.add_argument('--mode', default='val', type=str,
    help='Mode of dataloader, if only gray image, choose test, else val', choices=['val','test'])

 
def main():
    global args
    args = parser.parse_args()

    PATH = args.modelpath

    save_path = args.savepath
    ori_path = args.path

    checkpoint_G = torch.load(PATH)
    ngf = checkpoint_G['ngf']
    Large = checkpoint_G['Large']
    model_G = ConvGen(ngf)
    model_G.load_state_dict(checkpoint_G['state_dict'])

    model_G.eval()

    if args.time=='baseline':
        val_loader = get_loader(ori_path,
            batch_size=1,
            large=Large,
            mode=args.mode,
            num_workers=4,
            )
    else:
        val_loader = get_movie_time_loader(ori_path,
            batch_size=1,
            mode=args.mode,
            start_index = 1,
            num_workers=4,
            )
    val_bs = 1 #val_loader.batch_size
    cnt = 1
    with torch.no_grad(): # Fuck torch.no_grad!! Gradient will accumalte if you don't set torch.no_grad()!!
        if args.time=='baseline':
            for i, (data, target) in enumerate(val_loader):
                data, target = Variable(data), Variable(target)
                # print(data.shape)
                fake =  model_G(data).data
                # print("fake shape: ", fake.shape)
                for i in range(val_bs):
                    # #method 1
                    # transform = transforms.Compose([transforms.ToPILImage(), transforms.Resize(size=(360,480)), transforms.ToTensor()]);tmp_img = transform(fake[i])
                    # torchvision.utils.save_image(tmp_img, 'output/'+ '%05d.png' %(cnt))
                    # cnt += 1

                    # #method 2
                    p = transforms.ToPILImage()(fake[i].cpu())
                    p = p.resize((480,360))
                    p.save('output/'+ '%05d.png' %(cnt))
                    cnt += 1
        else:
            for i, (_now, _prev, _next, target, target_lab) in enumerate(val_loader):
                _now, _prev, _next, target, target_lab = Variable(_now), Variable(_prev), Variable(_next), Variable(target), Variable(target_lab)

                # validate with fake
                fake, fake_lab =  model_G(_now, _prev, _next).data

                for i in range(val_bs):

                    p = transforms.ToPILImage()(fake[i].cpu())
                    p = p.resize((480,360))
                    p.save('output/'+ '%05d.png' %(cnt))
                    cnt += 1



if __name__ == '__main__':
    main()
