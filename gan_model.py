import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from gan_model_time import Self_Attn


class ConvGen(nn.Module):
    '''Generator'''
    def __init__(self, ngf=32):
        super(ConvGen, self).__init__()

        self.conv1 = nn.Conv2d(1, ngf, 3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(ngf)
        self.relu1 = nn.LeakyReLU(0.1)

        self.conv2 = nn.Conv2d(ngf, ngf*2, 3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(ngf*2)
        self.relu2 = nn.LeakyReLU(0.1)

        self.conv3 = nn.Conv2d(ngf*2, ngf*4, 3, stride=2, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(ngf*4)
        self.relu3 = nn.LeakyReLU(0.1)

        self.conv4 = nn.Conv2d(ngf*4, ngf*8, 3, stride=2, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(ngf*8)
        self.relu4 = nn.LeakyReLU(0.1)

        self.conv5 = nn.Conv2d(ngf*8, ngf*8, 3, stride=2, padding=1, bias=False)
        self.bn5 = nn.BatchNorm2d(ngf*8)
        self.relu5 = nn.LeakyReLU(0.1)

        self.deconv6 = nn.ConvTranspose2d(ngf*8, ngf*8, 3, stride=2, padding=1, output_padding=1, bias=False)
        self.bn6 = nn.BatchNorm2d(ngf*8)
        self.relu6 = nn.ReLU()

        self.deconv7 = nn.ConvTranspose2d(ngf*8, ngf*4, 3, stride=2, padding=1, output_padding=1, bias=False)
        self.bn7 = nn.BatchNorm2d(ngf*4)
        self.relu7 = nn.ReLU()

        self.deconv8 = nn.ConvTranspose2d(ngf*4, ngf*2, 3, stride=2, padding=1, output_padding=1, bias=False)
        self.bn8 = nn.BatchNorm2d(ngf*2)
        self.relu8 = nn.ReLU()

        self.deconv9 = nn.ConvTranspose2d(ngf*2, ngf, 3, stride=2, padding=1, output_padding=1, bias=False)
        self.bn9 = nn.BatchNorm2d(ngf)
        self.relu9 = nn.ReLU()

        self.deconv10 = nn.ConvTranspose2d(ngf, 3, 3, stride=2, padding=1, output_padding=1, bias=False)
        self.bn10 = nn.BatchNorm2d(3)
        self.relu10 = nn.ReLU()

        self._initialize_weights()

    def forward(self, x):
        h = x
        h = self.conv1(h)
        h = self.bn1(h)
        h = self.relu1(h) # 64,112,112 (if input is 224x224)
        pool1 = h

        h = self.conv2(h)
        h = self.bn2(h)
        h = self.relu2(h) # 128,56,56
        pool2 = h

        h = self.conv3(h) # 256,28,28
        h = self.bn3(h)
        h = self.relu3(h)
        pool3 =h

        h = self.conv4(h) # 512,14,14
        h = self.bn4(h)
        h = self.relu4(h)
        pool4 = h

        h = self.conv5(h) # 512,7,7
        h = self.bn5(h)
        h = self.relu5(h)

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

        h = self.deconv9(h)
        h = self.bn9(h)
        h = self.relu9(h) # 64,112,112
        h += pool1

        h = self.deconv10(h)
        h = F.tanh(h) # 3,224,224

        return h

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            if isinstance(m, nn.ConvTranspose2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))


class ConvGenTime(nn.Module):
    '''Generator'''
    def __init__(self, ngf=32):
        super(ConvGenTime, self).__init__()

        self.conv1 = nn.Conv2d(1, ngf, 3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(ngf)
        self.relu1 = nn.LeakyReLU(0.1)

        self.conv2 = nn.Conv2d(ngf, ngf*2, 3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(ngf*2)
        self.relu2 = nn.LeakyReLU(0.1)

        self.conv3 = nn.Conv2d(ngf*2, ngf*4, 3, stride=2, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(ngf*4)
        self.relu3 = nn.LeakyReLU(0.1)

        self.conv4 = nn.Conv2d(ngf*4, ngf*8, 3, stride=2, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(ngf*8)
        self.relu4 = nn.LeakyReLU(0.1)

        self.conv5 = nn.Conv2d(ngf*8, ngf*8, 3, stride=2, padding=1, bias=False)
        self.bn5 = nn.BatchNorm2d(ngf*8)
        self.relu5 = nn.LeakyReLU(0.1)

        # === 1x1 cov for prev, now, next
        self.conv1x1prev = nn.Conv2d(ngf*8, ngf, 1, stride=1, padding=0, bias=False)
        self.conv1x1now = nn.Conv2d(ngf*8, ngf*6, 1, stride=1, padding=0, bias=False)
        self.conv1x1next = nn.Conv2d(ngf*8, ngf, 1, stride=1, padding=0, bias=False)
        self.bn1x1 = nn.BatchNorm2d(ngf*8)
        self.relu1x1 = nn.LeakyReLU(0.1)
        # ===

        self.deconv6 = nn.ConvTranspose2d(ngf*8, ngf*8, 3, stride=2, padding=1, output_padding=1, bias=False)
        self.bn6 = nn.BatchNorm2d(ngf*8)
        self.relu6 = nn.ReLU()

        self.deconv7 = nn.ConvTranspose2d(ngf*8, ngf*4, 3, stride=2, padding=1, output_padding=1, bias=False)
        self.bn7 = nn.BatchNorm2d(ngf*4)
        self.relu7 = nn.ReLU()

        self.deconv8 = nn.ConvTranspose2d(ngf*4, ngf*2, 3, stride=2, padding=1, output_padding=1, bias=False)
        self.bn8 = nn.BatchNorm2d(ngf*2)
        self.relu8 = nn.ReLU()

        self.deconv9 = nn.ConvTranspose2d(ngf*2, ngf, 3, stride=2, padding=1, output_padding=1, bias=False)
        self.bn9 = nn.BatchNorm2d(ngf)
        self.relu9 = nn.ReLU()

        self.deconv10 = nn.ConvTranspose2d(ngf, 3, 3, stride=2, padding=1, output_padding=1, bias=False)
        self.bn10 = nn.BatchNorm2d(3)
        self.relu10 = nn.ReLU()

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

        h_prev =self.conv1x1prev(h_prev) # 1x1conv

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

        h_next =self.conv1x1next(h_next) # 1x1conv

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

        h = self.deconv9(h)
        h = self.bn9(h)
        h = self.relu9(h) # 64,112,112
        h += pool1

        h = self.deconv10(h)
        h = F.tanh(h) # 3,224,224

        # =========================
        # Step 4  -END- UNET decoder
        # =========================

        return h

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            if isinstance(m, nn.ConvTranspose2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))


class ConvDis(nn.Module):
    '''Discriminator'''
    def __init__(self, large=False, ndf=32, use_self_attn=False):
        super(ConvDis, self).__init__()

        self.conv1 = nn.Conv2d(3, ndf, 3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(ndf)
        self.relu1 = nn.LeakyReLU(0.1)

        self.conv2 = nn.Conv2d(ndf, ndf*2, 3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(ndf*2)
        self.relu2 = nn.LeakyReLU(0.1)

        self.conv3 = nn.Conv2d(ndf*2, ndf*4, 3, stride=2, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(ndf*4)
        self.relu3 = nn.LeakyReLU(0.1)

        self.conv4 = nn.Conv2d(ndf*4, ndf*8, 3, stride=2, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(ndf*8)
        self.relu4 = nn.LeakyReLU(0.1)

        self.conv5 = nn.Conv2d(ndf*8, ndf*8, 3, stride=2, padding=1, bias=False)
        self.bn5 = nn.BatchNorm2d(ndf*8)
        self.relu5 = nn.LeakyReLU(0.1)

        if large:
            self.conv6 = nn.Conv2d(ndf*8, ndf*8, 15, stride=1, padding=0, bias=False)
        else:
            self.conv6 = nn.Conv2d(ndf*8, ndf*8, 7, stride=1, padding=0, bias=False)
        self.bn6 = nn.BatchNorm2d(ndf*8)
        self.relu6 = nn.LeakyReLU(0.1)

        self.conv7 = nn.Conv2d(ndf*8, 1, 1, stride=1, padding=0, bias=False)

        self.attn1 = Self_Attn(ndf*8, 'relu')
        self.attn2 = Self_Attn(ndf*8, 'relu')

        self._initialize_weights()

    def forward(self, x):
        h = x
        h = self.conv1(h)
        h = self.bn1(h)
        h = self.relu1(h) # 64,112,112 (if input is 224x224)

        h = self.conv2(h)
        h = self.bn2(h)
        h = self.relu2(h) # 128,56,56

        h = self.conv3(h) # 256,28,28
        h = self.bn3(h)
        h = self.relu3(h)

        h = self.conv4(h) # 512,14,14
        h = self.bn4(h)
        h = self.relu4(h)

        h = self.conv5(h) # 512,7,7
        h = self.bn5(h)
        h = self.relu5(h)

        if use_self_attn:
            h, p1 = self.attn1(h)

        h = self.conv6(h)
        h = self.bn6(h)
        h = self.relu6(h) # 512,1,1

        if use_self_attn:
            h, p2 = self.attn2(h)

        h = self.conv7(h)
        h = F.sigmoid(h)

        return h #, p1, p2

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            if isinstance(m, nn.ConvTranspose2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
