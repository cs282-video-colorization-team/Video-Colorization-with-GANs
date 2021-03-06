import torch
import os
from gan_model import *
from gan_model_time import *
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
import matplotlib

parser = argparse.ArgumentParser(description='Test Colorization using GAN')
parser.add_argument('--path', type=str,
    help='Root path for dataset')
parser.add_argument('--savepath', type=str,
    help='Save Root path for dataset')
parser.add_argument('--modelpath', type=str,
    help='Model Root path')
parser.add_argument('--time', type=str,
    help='baseline or with time', choices=['baseline', 'time'])
parser.add_argument('--start_index', default=1, type=int,
                    help='start_index of the filename')
parser.add_argument('--is_color', action="store_true",
                    help='say it if it is a image with color')

# parser.add_argument('--mode', default='val', type=str,
#     help='Mode of dataloader, if only gray image, choose test, else val', choices=['val','test'])

 
def main():
    global args
    args = parser.parse_args()

    PATH = args.modelpath

    save_path = args.savepath
    ori_path = args.path

    checkpoint_G = torch.load(PATH)
    ngf = checkpoint_G['ngf']
    Large = checkpoint_G['Large']
    use_self_attn = checkpoint_G['use_self_attn']
    if args.time=='baseline':
        model_G = ConvGen(ngf)
    else:
        model_G = ConvGenTime(ngf, use_self_attn)
    model_G.load_state_dict(checkpoint_G['state_dict'])

    model_G.eval()

    val_bs = checkpoint_G['batch_size']

    if args.time=='baseline':
        test_loader = get_loader(ori_path,
            batch_size=val_bs,
            large=Large,
            mode='test',
            num_workers=4
            )
    else:
        test_loader = get_movie_time_loader(ori_path,
            batch_size=val_bs,
            mode='test',
            num_workers=4, 
            start_index=args.start_index
            )
        
    with torch.no_grad(): # Fuck torch.no_grad!! Gradient will accumalte if you don't set torch.no_grad()!!
        if args.time=='baseline':
            for i, (data, filename) in enumerate(test_loader):
                print("Processing image", filename)
                # print("filename:", filename)
                # print("filename type:", type(filename))
                data = Variable(data)
                # print(data.shape)
                fake =  model_G(data).data
                # print("fake shape: ", fake.shape)
                for j in range(val_bs):
                    # #method 1
                    # transform = transforms.Compose([transforms.ToPILImage(), transforms.Resize(size=(360,480)), transforms.ToTensor()]);tmp_img = transform(fake[i])
                    # torchvision.utils.save_image(tmp_img, 'output/'+ '%05d.png' %(cnt))

                    # #method 2
                    # p = transforms.ToPILImage()(fake[j].cpu())
                    # p = p.resize((480,360))
                    # p.save(save_path + filename[j])

                    # #method 3
                    pred = fake[j].cpu().numpy()
                    pred_rgb = (np.transpose(pred, (1,2,0)).astype(np.float64) + 1) / 2.
                    matplotlib.image.imsave(save_path + filename[j], pred_rgb)
        else:
            for i, (_now, _prev, _next, filename) in enumerate(test_loader):
                print("Processing image", filename)
                _now, _prev, _next = Variable(_now), Variable(_prev), Variable(_next)

                # validate with fake
                fake =  model_G(_now, _prev, _next).data

                for j in range(val_bs):

                    # p = transforms.ToPILImage()(fake[j].cpu())
                    # p = p.resize((480,360))
                    # p.save(save_path + filename[j])
                    pred = fake[j].cpu().numpy()
                    pred_rgb = (np.transpose(pred, (1,2,0)).astype(np.float64) + 1) / 2.
                    matplotlib.image.imsave(save_path + filename[j], pred_rgb)



if __name__ == '__main__':
    main()
