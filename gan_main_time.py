# from transform import ReLabel, ToLabel, ToSP, Scale
from gan_model_time import ConvGenTime
from gan_model import ConvDis
from gan_model_time import PatchDis
from utils import *
import gan_model_time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils import data
import torch.nn.init as init
import torch.nn.functional as F
import torchvision
from torchvision import datasets, models, transforms
from skimage import color
from movie_time_data_loader import *

import time
import os
import sys
from PIL import Image
import argparse
import numpy as np
import matplotlib.pyplot as plt
import datetime

import matplotlib


parser = argparse.ArgumentParser(description='Colorization using GAN')
parser.add_argument('--path', type=str,
                    help='Root path for dataset')
parser.add_argument('--large', action="store_true",
                    help='Use larger images?')
parser.add_argument('--batch_size', default=4, type=int,
                    help='Batch size: default 4')
parser.add_argument('--lr', default=1e-4, type=float,
                    help='Learning rate for optimizer')
parser.add_argument('--weight_decay', default=0, type=float,
                    help='Weight decay for optimizer')
parser.add_argument('--num_epoch', default=20, type=int,
                    help='Number of epochs')
parser.add_argument('--lamb', default=100, type=int,
                    help='Lambda for L1 Loss')
parser.add_argument('--test', default='', type=str,
                    help='Path to the model, for testing')
parser.add_argument('--model_G', default='', type=str,
                    help='Path to resume for Generator model')
parser.add_argument('--model_D', default='', type=str,
                    help='Path to resume for Discriminator model')
parser.add_argument('--ngf', default=32, type=int,
                    help='# of gen filters in first conv layer')
parser.add_argument('--ndf', default=32, type=int,
                    help='# of discrim filters in first conv layer')

parser.add_argument('--numG', default=1, type=int, help='G trains numG times when D trains per time')
parser.add_argument('--numD', default=1, type=int, help='D trains numD times when G trains per time')
parser.add_argument('--patchGAN', action='store_true', help='Use patchGAN in Discriminator')
parser.add_argument('--use_lsgan', action='store_true', help='Use LSGAN in loss criterion')
parser.add_argument('--use_self_attn', action='store_true', help='Use self attention in both G and D')

# parser.add_argument('-p', '--plot', action="store_true",
#                     help='Plot accuracy and loss diagram?')
parser.add_argument('-s','--save', action="store_true",
                    help='Save model?')
parser.add_argument('--gpu', default=0, type=int,
                    help='Which GPU to use?')


def main():
    global args, date
    args = parser.parse_args()
    date = str(datetime.datetime.now()).replace(':', '_').replace(' ', '_').replace('.', '_')

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    model_G = ConvGenTime(args.ngf, args.use_self_attn)
    if args.patchGAN:
        model_D = PatchDis(large=args.large, ndf=args.ndf, use_self_attn=args.use_self_attn)
    else:
        model_D = ConvDis(large=args.large, ndf=args.ndf, use_self_attn=args.use_self_attn)

    start_epoch_G = start_epoch_D = 0
    if args.model_G:
        print('Resume model G: %s' % args.model_G)
        checkpoint_G = torch.load(resume)
        model_G.load_state_dict(checkpoint_G['state_dict'])
        start_epoch_G = checkpoint_G['epoch']
    if args.model_D:
        print('Resume model D: %s' % args.model_D)
        checkpoint_D = torch.load(resume)
        model_D.load_state_dict(checkpoint_D['state_dict'])
        start_epoch_D = checkpoint_D['epoch']
    assert start_epoch_G == start_epoch_D
    if args.model_G == '' and args.model_D == '':
        print('No Resume')
        start_epoch = 0

    model_G.cuda()
    model_D.cuda()

    # optimizer
    optimizer_G = optim.Adam(model_G.parameters(),
                             lr=args.lr, betas=(0.5, 0.999),
                             eps=1e-8, weight_decay=args.weight_decay)
    optimizer_D = optim.Adam(model_D.parameters(),
                             lr=args.lr, betas=(0.5, 0.999),
                             eps=1e-8, weight_decay=args.weight_decay)
    if args.model_G:
        optimizer_G.load_state_dict(checkpoint_G['optimizer'])
    if args.model_D:
        optimizer_D.load_state_dict(checkpoint_D['optimizer'])

    # loss function
    global criterionGAN
    #criterion = nn.BCELoss()
    criterionGAN = gan_model_time.GANLoss(use_lsgan=args.use_lsgan, tensor =torch.cuda.FloatTensor)

    global L1
    L1 = nn.L1Loss()

    # dataset
    data_root = args.path

    train_loader = get_movie_time_loader(os.path.join(data_root, 'train/'),
                             batch_size=args.batch_size,
                             large=args.large,
                             mode='train',
                             start_index = 1,
                             num_workers=4,
                             shuffle=True
                            )

    val_loader = get_movie_time_loader(os.path.join(data_root, 'val/'),
                            batch_size=args.batch_size,
                            large=args.large,
                            mode='val',
                            start_index = 10000,
                            num_workers=4,
                            shuffle=True
                            )

    global val_bs
    val_bs = val_loader.batch_size

    # set up plotter, path, etc.
    global iteration, print_interval, plotter, plotter_basic
    iteration = 0
    print_interval = args.numG * 5
    plotter = Plotter_GAN_TV()
    plotter_basic = Plotter_GAN()

    global img_path
    size = ''
    if args.large: size = 'Large'
    img_path = 'img/model_time/%s/GAN_%s_L1%d_bs%d_%s_lr%s_ngf%d_ndf%d_numG%d_numD%d/' \
               % (date, size, args.lamb, args.batch_size, 'Adam', str(args.lr), args.ngf, args.ndf, args.numG, args.numD)
    model_path = 'model/model_time/%s/GAN_%s_L1%d_bs%d_%s_lr%s_ngf%d_ndf%d_numG%d_numD%d/' \
               % (date, size, args.lamb, args.batch_size, 'Adam', str(args.lr), args.ngf, args.ndf, args.numG, args.numD)
    if not os.path.exists(img_path):
        os.makedirs(img_path)
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    # start loop
    start_epoch = 0

    for epoch in range(start_epoch, args.num_epoch):
        print('Epoch {}/{}'.format(epoch, args.num_epoch - 1))
        print('-' * 20)

        if epoch == 0:
            val_lerrG, val_errD = validate(val_loader, model_G, model_D, optimizer_G, optimizer_D, epoch=-1)
        # train
        train_errG, train_errD = train(train_loader, model_G, model_D, optimizer_G, optimizer_D, epoch, iteration)
        # validate
        val_lerrG, val_errD = validate(val_loader, model_G, model_D, optimizer_G, optimizer_D, epoch)

        plotter.train_update(train_errG, train_errD)
        plotter.val_update(val_lerrG, val_errD)
        plotter.draw(img_path + 'train_val.png')

        if args.save and (epoch % 10 == 1):
            print('Saving check point')
            save_checkpoint({'epoch': epoch + 1,
                             'state_dict': model_G.state_dict(),
                             'optimizer': optimizer_G.state_dict(),
                             'Large': args.large,
                             'ngf': args.ngf,
                             'batch_size': args.batch_size,
                             'ndf': args.ndf,
                             'numG': args.numG,
                             'numD': args.numD,
                             'use_self_attn': args.use_self_attn,
                             },
                             filename=model_path+'G_epoch%d.pth.tar' \
                             % epoch)

    # Save the latest model
    print('Save the lastest model')
    save_checkpoint({'epoch': epoch + 1,
                     'state_dict': model_G.state_dict(),
                     'optimizer': optimizer_G.state_dict(),
                     'Large': args.large,
                     'ngf': args.ngf,
                     'batch_size': args.batch_size,
                     'ndf': args.ndf,
                     'numG': args.numG,
                     'numD': args.numD,
                     'use_self_attn': args.use_self_attn,
                     },
                     filename=model_path+'Last_G_epoch%d.pth.tar' \
                     % epoch)


    save_checkpoint({'epoch': epoch + 1,
                     'state_dict': model_D.state_dict(),
                     'optimizer': optimizer_D.state_dict(),
                     'Large': args.large,
                     'ngf': args.ngf,
                     'batch_size': args.batch_size,
                     'ndf': args.ndf,
                     'numG': args.numG,
                     'numD': args.numD,
                     'use_self_attn': args.use_self_attn,
                     },
                     filename=model_path+'Last_D_epoch%d.pth.tar' \
                     % epoch)



def train(train_loader, model_G, model_D, optimizer_G, optimizer_D, epoch, iteration):
    errorG = AverageMeter() # will be reset after each epoch
    errorD = AverageMeter() # will be reset after each epoch
    errorG_basic = AverageMeter() # basic will be reset after each print
    errorD_basic = AverageMeter() # basic will be reset after each print
    errorD_real = AverageMeter()
    errorD_fake = AverageMeter()
    errorG_GAN = AverageMeter()
    #errorG_L1_lab = AverageMeter()
    errorG_R = AverageMeter()

    model_G.train()
    model_D.train()

    real_label = 1
    fake_label = 0

    #for i, (_now, _prev, _next, target, target_lab) in enumerate(train_loader):
    for i, (_now, _prev, _next, target) in enumerate(train_loader):

        #_now, _prev, _next, target, target_lab = Variable(_now.cuda()), Variable(_prev.cuda()), Variable(_next.cuda()), Variable(target.cuda()), Variable(target_lab.cuda())
        _now, _prev, _next, target = Variable(_now.cuda()), Variable(_prev.cuda()), Variable(_next.cuda()), Variable(target.cuda())

        ########################
        # update D network
        ########################
        # train with real
        if (i % args.numG) == 0:
            # dr1, dr2, df1, df2, gf1, gf2 are attention scores
            model_D.zero_grad()
            output = model_D(target)
            #label = torch.FloatTensor(target.size(0)).fill_(real_label).cuda()
            #labelv = Variable(label)
            #errD_real = criterion(torch.squeeze(output), labelv)
            errD_real = criterionGAN(output, True)
            errD_real.backward()
            D_x = output.data.mean()

            # train with fake
            #fake, _ =  model_G(_now, _prev, _next)
            fake = model_G(_now, _prev, _next)
            #labelv = Variable(label.fill_(fake_label))
            output = model_D(fake.detach())
            #errD_fake = criterion(torch.squeeze(output), labelv)
            errD_fake = criterionGAN(output, False)
            errD_fake.backward()
            D_G_x1 = output.data.mean()

            errD = errD_real + errD_fake
            optimizer_D.step()

        ########################
        # update G network
        ########################

        if (i % args.numD) == 0:
            #labelv = Variable(label.fill_(real_label))
            #fake, fake_lab = model_G(_now, _prev, _next)
            fake = model_G(_now, _prev, _next)
            model_G.zero_grad()
            output = model_D(fake)
            #errG_GAN = criterion(torch.squeeze(output), labelv)
            errG_GAN = criterionGAN(output, True)
            errG_L1 = L1(fake.view(fake.size(0),-1), target.view(target.size(0),-1))
            #errG_L1_lab = L1(fake_lab.view(fake_lab.size(0),-1), target_lab.view(target_lab.size(0),-1))

            #errG = errG_GAN + args.lamb * (0.7 * errG_L1 + 0.3 * errG_L1_lab)
            errG = errG_GAN + args.lamb * errG_L1
            errG.backward()
            D_G_x2 = output.data.mean()
            optimizer_G.step()

        # store error values
        if (i % max(args.numG, args.numD)) == 0:
            errorG.update(errG, target.size(0), history=1)
            errorD.update(errD, target.size(0), history=1)
            errorG_basic.update(errG, target.size(0), history=1)
            errorD_basic.update(errD, target.size(0), history=1)
            errorD_real.update(errD_real, target.size(0), history=1)
            errorD_fake.update(errD_fake, target.size(0), history=1)
            #errorG_L1_lab.update(errG_L1_lab, target.size(0), history=1)

            errorD_real.update(errD_real, target.size(0), history=1)
            errorD_fake.update(errD_fake, target.size(0), history=1)
            errorG_GAN.update(errG_GAN, target.size(0), history=1)
            errorG_R.update(errG_L1, target.size(0), history=1)

        if i == 0:
            vis_result(_now.data, target.data, fake.data, epoch, mode = 'train')

        if iteration % print_interval == 0:
            print('Epoch%d[%d/%d]: Loss_D: %.4f(R%0.4f+F%0.4f) Loss_G: %0.4f(GAN%.4f+RGB%0.4f) D(x): %.3f D(G(z)): %.3f / %.3f' \
                % (epoch, i, len(train_loader),
                errorD_basic.avg, errorD_real.avg, errorD_fake.avg,
                errorG_basic.avg, errorG_GAN.avg, errorG_R.avg * args.lamb,
                D_x, D_G_x1, D_G_x2
                ))
            # plot image
            plotter_basic.g_update(errorG_basic.avg)
            plotter_basic.d_update(errorD_basic.avg)
            plotter_basic.draw(img_path + 'train_basic.png')
            # reset AverageMeter
            errorG_basic.reset()
            errorD_basic.reset()
            errorD_real.reset()
            errorD_fake.reset()
            errorG_GAN.reset()
            errorG_R.reset()
            #errorG_L1_lab.reset()

        iteration += 1

    return errorG.avg, errorD.avg


def validate(val_loader, model_G, model_D, optimizer_G, optimizer_D, epoch):
    errorG = AverageMeter()
    errorD = AverageMeter()

    model_G.eval()
    model_D.eval()

    real_label = 1
    fake_label = 0

    with torch.no_grad(): # Fuck torch.no_grad!! Gradient will accumalte if you don't set torch.no_grad()!!
        #for i, (_now, _prev, _next, target, target_lab) in enumerate(val_loader):
        for i, (_now, _prev, _next, target) in enumerate(val_loader):
            #_now, _prev, _next, target, target_lab = Variable(_now.cuda()), Variable(_prev.cuda()), Variable(_next.cuda()), Variable(target.cuda()), Variable(target_lab.cuda())
            _now, _prev, _next, target = Variable(_now.cuda()), Variable(_prev.cuda()), Variable(_next.cuda()), Variable(target.cuda())
            ########################
            # D network
            ########################
            # validate with real
            output = model_D(target)
            #label = torch.FloatTensor(target.size(0)).fill_(real_label).cuda()
            #labelv = Variable(label)
            
            errD_real = criterionGAN(output, True)

            # validate with fake
            #fake, fake_lab =  model_G(_now, _prev, _next)
            fake = model_G(_now, _prev, _next)
            #labelv = Variable(label.fill_(fake_label))
            output = model_D(fake.detach())
            errD_fake = criterionGAN(output, False)

            errD = errD_real + errD_fake

            ########################
            # G network
            ########################
            #labelv = Variable(label.fill_(real_label))
            output = model_D(fake)
            errG_GAN = criterionGAN(output, True)
            errG_L1 = L1(fake.view(fake.size(0),-1), target.view(target.size(0),-1))
            #errG_L1_lab = L1(fake_lab.view(fake_lab.size(0),-1), target_lab.view(target_lab.size(0),-1))

            #errG = errG_GAN + args.lamb * (0.8 * errG_L1 + 0.2 * errG_L1_lab)
            errG = errG_GAN + args.lamb * errG_L1 

            errorG.update(errG, target.size(0), history=1)
            errorD.update(errD, target.size(0), history=1)

            if i == 0:
                vis_result(_now.data, target.data, fake.data, epoch, mode = 'val')

            if i % 50 == 0:
                print('Validating Epoch %d: [%d/%d]' \
                    % (epoch, i, len(val_loader)))

        print('Validation: Loss_D: %.4f Loss_G: %.4f '\
            % (errorD.avg, errorG.avg))

    return errorG.avg, errorD.avg

def vis_result(data, target, output, epoch, mode='val'):
    '''visualize images for GAN'''
    img_list = []
    for i in range(min(32, val_bs)):
        l = torch.unsqueeze(torch.squeeze(data[i]), 0).cpu().numpy()
        # from IPython import embed; embed()
        raw = target[i].cpu().numpy()
        pred = output[i].cpu().numpy()

        raw_rgb = (np.transpose(raw, (1,2,0)).astype(np.float64) + 1) / 2.
        pred_rgb = (np.transpose(pred, (1,2,0)).astype(np.float64) + 1) / 2.

        # denormalize the grey image for visualization 
        grey = l * 0.5 + 0.5 
        grey = np.transpose(grey, (1,2,0))
        grey = np.repeat(grey, 3, axis=2).astype(np.float64)
        img_list.append(np.concatenate((grey, raw_rgb, pred_rgb), 1))
    display_len = 4 if len(img_list) >= 4 else 2
    img_list = [np.concatenate(img_list[display_len*i:display_len*(i+1)], axis=1) for i in range(len(img_list) // display_len)]
    img_list = np.concatenate(img_list, axis=0)

    if mode=='val':
        matplotlib.image.imsave(img_path + 'epoch%d_val.png' % epoch, img_list)
    else:
        matplotlib.image.imsave(img_path + 'epoch%d_train.png' % epoch, img_list)

if __name__ == '__main__':
    main()
