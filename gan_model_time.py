import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from spectral_norm import SpectralNorm
import numpy as np
from torch.autograd import Variable

class Self_Attn(nn.Module):
    """ Self Attention Layer"""
    def __init__(self, in_dim, activation):
        super(Self_Attn, self).__init__()
        self.chanel_in = in_dim
        self.activation = activation

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1) 

    def forward(self, x):
        """
            inputs:
                x: input feature maps(N, C, H, W)
            returns:
                out: self attention value + input feature
                attention: (C, W*H, W*H)
        """
        N, C, H, W = x.size()
        proj_query = self.query_conv(x).view(N, -1, H*W).permute(0, 2, 1) # (N, H*W, C)
        proj_key = self.key_conv(x).view(N, -1, H*W) # (N, C, H*W)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(N, -1, H*W) # (N, C, H*W)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(N, C, H, W)

        out = self.gamma * out + x
        return out, attention


class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                self.real_label_var = Variable(real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                self.fake_label_var = Variable(fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)


class ConvGenTime(nn.Module):
    '''Generator'''
    def __init__(self, ngf=64):
        super(ConvGenTime, self).__init__()

        self.conv1 = nn.Conv2d(1, ngf, 3, stride=2, padding=1, bias=False) # ngf = 64
        self.bn1 = nn.InstanceNorm2d(ngf, affine=True)
        self.relu1 = nn.LeakyReLU(0.1)

        self.conv2 = nn.Conv2d(ngf, ngf*2, 3, stride=2, padding=1, bias=False) # 128
        self.bn2 = nn.InstanceNorm2d(ngf*2, affine=True)
        self.relu2 = nn.LeakyReLU(0.1)

        self.conv3 = nn.Conv2d(ngf*2, ngf*4, 3, stride=2, padding=1, bias=False) # 256
        self.bn3 = nn.InstanceNorm2d(ngf*4, affine=True)
        self.relu3 = nn.LeakyReLU(0.1)

        self.conv4 = nn.Conv2d(ngf*4, ngf*8, 3, stride=2, padding=1, bias=False) # 512
        self.bn4 = nn.InstanceNorm2d(ngf*8, affine=True)
        self.relu4 = nn.LeakyReLU(0.1)

        self.conv5 = nn.Conv2d(ngf*8, ngf*8, 3, stride=2, padding=1, bias=False) # 512
        self.bn5 = nn.InstanceNorm2d(ngf*8, affine=True)
        self.relu5 = nn.LeakyReLU(0.1)

        # === 1x1 cov for prev, now, next
        self.conv1x1prev = nn.Conv2d(ngf*8, ngf, 1, stride=1, padding=0, bias=False)  # 512
        self.conv1x1now = nn.Conv2d(ngf*8, ngf*6, 1, stride=1, padding=0, bias=False)
        self.conv1x1next = nn.Conv2d(ngf*8, ngf, 1, stride=1, padding=0, bias=False)
        self.bn1x1 = nn.InstanceNorm2d(ngf*8, affine=True)
        self.relu1x1 = nn.LeakyReLU(0.1)
        # ===

        self.deconv6 = nn.ConvTranspose2d(ngf*8, ngf*8, 3, stride=2, padding=1, output_padding=1, bias=False) # 512
        self.bn6 = nn.InstanceNorm2d(ngf*8, affine=True)
        self.relu6 = nn.ReLU()

        self.deconv7 = nn.ConvTranspose2d(ngf*8, ngf*4, 3, stride=2, padding=1, output_padding=1, bias=False) # 256
        self.bn7 = nn.InstanceNorm2d(ngf*4, affine=True)
        self.relu7 = nn.ReLU()

        self.deconv8 = nn.ConvTranspose2d(ngf*4, ngf*2, 3, stride=2, padding=1, output_padding=1, bias=False) # 128
        self.bn8 = nn.InstanceNorm2d(ngf*2, affine=True)
        self.relu8 = nn.ReLU()

        self.deconv9 = nn.ConvTranspose2d(ngf*2, ngf, 3, stride=2, padding=1, output_padding=1, bias=False) # 64
        self.bn9 = nn.InstanceNorm2d(ngf, affine=True)
        self.relu9 = nn.ReLU()

        self.deconvRGB = nn.ConvTranspose2d(ngf, 3, 3, stride=2, padding=1, output_padding=1, bias=False)

        self.deconvLAB = nn.ConvTranspose2d(ngf, 3, 3, stride=2, padding=1, output_padding=1, bias=False)

        self.attn1 = Self_Attn(ngf*2, 'relu')
        self.attn2 = Self_Attn(ngf, 'relu')

        self._initialize_weights()

    def forward(self, _now, _prev, _next):


        # ==========================
        # Step 1  -START- now branch 
        # ==========================

        h_now = _now
        h_now = self.conv1(h_now)
        h_now = self.bn1(h_now)
        h_now = self.relu1(h_now) # 64,112,112 (if input is 224x224)
        pool1 = h_now

        h_now = self.conv2(h_now)
        h_now = self.bn2(h_now)
        h_now = self.relu2(h_now) # 128,56,56
        pool2 = h_now

        h_now = self.conv3(h_now) # 256,28,28
        h_now = self.bn3(h_now)
        h_now = self.relu3(h_now)
        pool3 = h_now

        h_now = self.conv4(h_now) # 512,14,14
        h_now = self.bn4(h_now)
        h_now = self.relu4(h_now)
        pool4 = h_now

        h_now = self.conv5(h_now) # 512,7,7
        h_now = self.bn5(h_now)
        h_now = self.relu5(h_now) # >>>>> Bottleneck

        h_now =self.conv1x1now(h_now) # 1x1conv

        # ==========================
        # Step 1    -END- now branch 
        # ==========================

        # ========================
        # Step 2  -START- prev branch 
        # =========================

        h_prev = _prev
        h_prev = self.conv1(h_prev)
        h_prev = self.bn1(h_prev)
        h_prev = self.relu1(h_prev) # 64,112,112 (if input is 224x224)

        h_prev = self.conv2(h_prev)
        h_prev = self.bn2(h_prev)
        h_prev = self.relu2(h_prev) # 128,56,56

        h_prev = self.conv3(h_prev) # 256,28,28
        h_prev = self.bn3(h_prev)
        h_prev = self.relu3(h_prev)

        h_prev = self.conv4(h_prev) # 512,14,14
        h_prev = self.bn4(h_prev)
        h_prev = self.relu4(h_prev)

        h_prev = self.conv5(h_prev) # 512,7,7
        h_prev = self.bn5(h_prev)
        h_prev = self.relu5(h_prev) # >>>>> Bottleneck

        h_prev = self.conv1x1prev(h_prev) # 1x1conv

        # ========================
        # Step 2    -END- prev branch 
        # ========================

        # ========================
        # Step 3  -START- next branch 
        # =========================

        h_next = _next
        h_next = self.conv1(h_next)
        h_next = self.bn1(h_next)
        h_next = self.relu1(h_next) # 64,112,112 (if input is 224x224)

        h_next = self.conv2(h_next)
        h_next = self.bn2(h_next)
        h_next = self.relu2(h_next) # 128,56,56

        h_next = self.conv3(h_next) # 256,28,28
        h_next = self.bn3(h_next)
        h_next = self.relu3(h_next)

        h_next = self.conv4(h_next) # 512,14,14
        h_next = self.bn4(h_next)
        h_next = self.relu4(h_next)

        h_next = self.conv5(h_next) # 512,7,7
        h_next = self.bn5(h_next)
        h_next = self.relu5(h_next) # >>>>> Bottleneck

        h_next = self.conv1x1next(h_next) # 1x1conv

        # ========================
        # Step 3    -END- next branch 
        # ========================


        # =========================
        # Step 4  -START- concat feature map
        # =========================
        h = torch.cat((h_prev, h_now, h_next), dim=1)
        h = self.bn1x1(h)
        h = self.relu1x1(h) # 512,7,7

        # =========================
        # Step 4  -END- concat feature map
        # =========================

        # =========================
        # Step 4  -START- UNET decoder
        # =========================
        h = self.deconv6(h)
        h = self.bn6(h)
        h = self.relu6(h) # 512,14,14
        h += pool4

        h = self.deconv7(h)
        h = self.bn7(h)
        h = self.relu7(h) # 256,28,28
        h += pool3

        h = self.deconv8(h)
        h = self.bn8(h)
        h = self.relu8(h) # 128,56,56
        h += pool2

        h, p1 = self.attn1(h)

        h = self.deconv9(h)
        h = self.bn9(h)
        h = self.relu9(h) # 64,112,112
        h += pool1

        h, p2 = self.attn2(h)

        rgb = self.deconvRGB(h)
        rgb = F.tanh(rgb) # 3,224,224

        # lab = self.deconvLAB(h)
        # lab = F.tanh(lab) # 3,224,224

        # =========================
        # Step 4  -END- UNET decoder
        # =========================

        return rgb, p1, p2#, lab

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            if isinstance(m, nn.ConvTranspose2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))


class PatchDis(nn.Module):
    '''Discriminator'''
    def __init__(self, large=False, ndf=64):
        super(PatchDis, self).__init__()

        def init_conv(insize, outsize, kernel_size, stride, padding, bias=True):
            m = nn.Conv2d(insize, outsize, kernel_size, stride=stride, padding=padding, bias=bias)
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
            return m

        self.conv1 = SpectralNorm(init_conv(3, ndf, kernel_size=4, stride=2, padding=1)) # 64
        self.relu1 = nn.LeakyReLU(0.01, inplace=True)

        self.conv2 = SpectralNorm(init_conv(ndf, ndf*2, kernel_size=4, stride=2, padding=1)) # 128
        self.relu2 = nn.LeakyReLU(0.01, inplace=True)

        self.conv3 = SpectralNorm(init_conv(ndf*2, ndf*4, kernel_size=4, stride=2, padding=1)) # 256
        self.relu3 = nn.LeakyReLU(0.01, inplace=True)

        self.conv4 = SpectralNorm(init_conv(ndf*4, ndf*8, kernel_size=4, stride=1, padding=1)) # 512
        self.relu4 = nn.LeakyReLU(0.01, inplace=True)

        self.conv5 = SpectralNorm(init_conv(ndf*8, 1, kernel_size=4, stride=1, padding=1, bias=False)) # 1

        self.attn1 = Self_Attn(ndf*4, 'relu')
        self.attn2 = Self_Attn(ndf*8, 'relu')
        
    def forward(self, x):
        # h = self.main(x)
        # output = self.conv1(h)
        h = self.conv1(x)
        h = self.relu1(h)

        #h = self.conv2(h)
        h = self.conv2(h)
        h = self.relu2(h)

        #h = self.conv3(h)
        h = self.conv3(h)
        h = self.relu3(h)

        h, p1 = self.attn1(h)

        #h = self.conv4(h)
        h = self.conv4(h)
        h = self.relu4(h)

        h, p2 = self.attn2(h)

        #h = self.conv5(h)
        h = self.conv5(h)

        return h.squeeze(), p1, p2
