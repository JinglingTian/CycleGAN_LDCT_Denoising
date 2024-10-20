import torch
import torch.nn as nn
import torch.nn.functional as F
from .block import *


class Generator_CycleGAN(nn.Module):
    def __init__(self, input_nc=1, output_nc=1, n_residual_blocks=9):
        super(Generator_CycleGAN, self).__init__()

        # Initial convolution block       
        model = [   nn.ReflectionPad2d(3),
                    nn.Conv2d(input_nc, 64, 7),
                    nn.InstanceNorm2d(64),
                    nn.ReLU(inplace=True) ]

        # Downsampling
        in_features = 64
        out_features = in_features*2
        for _ in range(2):
            model += [  nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                        nn.InstanceNorm2d(out_features),
                        nn.ReLU(inplace=True) ]
            in_features = out_features
            out_features = in_features*2

        # Residual blocks
        for _ in range(n_residual_blocks):
            model += [ResidualBlock(in_features)]

        # Upsampling
        out_features = in_features//2
        for _ in range(2):
            model += [  nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                        nn.InstanceNorm2d(out_features),
                        nn.ReLU(inplace=True) ]
            in_features = out_features
            out_features = in_features//2

        # Output layer
        model += [  nn.ReflectionPad2d(3),
                    nn.Conv2d(64, output_nc, 7),
                    nn.Tanh() ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)
    

class Generator_Unet(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.en_conv1 = ConvBlock(1,64)
        self.en_conv2 = ConvBlock(64,128)
        self.en_conv3 = ConvBlock(128,256)
        self.en_conv4 = ConvBlock(256,512)

        self.buttom_conv = ConvBlock(512,1024)
        
        self.de_conv4 = ConvBlock(1024+512,512)
        self.de_conv3 = ConvBlock(512+256,256)
        self.de_conv2 = ConvBlock(256+128,128)
        self.de_conv1 = ConvBlock(128+64,64)

        self.out_conv = nn.Sequential(
            nn.Conv2d(64,1,1,1),
            nn.Sigmoid()
            )
    def up(self,x):
        _,_,h,w = x.shape
        return F.interpolate(x, size=(int(h*2),int(w*2)), mode="bilinear")
    def down(self, x):
        return F.max_pool2d(x, kernel_size=2)

    def forward(self,x):
        en1 = self.en_conv1(x)
        en2 = self.en_conv2(self.down(en1))
        en3 = self.en_conv3(self.down(en2))
        en4 = self.en_conv4(self.down(en3))
        
        x = self.buttom_conv(self.down(en4))

        x = self.de_conv4(torch.concat([en4,self.up(x)],dim=1))
        x = self.de_conv3(torch.concat([en3,self.up(x)],dim=1))
        x = self.de_conv2(torch.concat([en2,self.up(x)],dim=1))
        x = self.de_conv1(torch.concat([en1,self.up(x)],dim=1))

        x = self.out_conv(x)
        return x

class Generator_Unet_SE(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.en_conv1 = ConvBlock(1,64)
        self.en_conv2 = ConvBlock(64,128)
        self.en_conv3 = ConvBlock(128,256)
        self.en_conv4 = ConvBlock(256,512)

        self.buttom_conv = ConvBlock(512,1024)
        
        self.de_conv4 = ConvBlock(1024+512,512)
        self.de_conv3 = ConvBlock(512+256,256)
        self.de_conv2 = ConvBlock(256+128,128)
        self.de_conv1 = ConvBlock(128+64,64)

        self.out_conv = nn.Sequential(
            nn.Conv2d(64,1,1,1),
            nn.Sigmoid()
            )
        self.se_eca = SE(512)
    def up(self,x):
        _,_,h,w = x.shape
        return F.interpolate(x, size=(int(h*2),int(w*2)), mode="bilinear")
    def down(self, x):
        return F.max_pool2d(x, kernel_size=2)

    def forward(self,x):
        en1 = self.en_conv1(x)
        en2 = self.en_conv2(self.down(en1))
        en3 = self.en_conv3(self.down(en2))
        en4 = self.en_conv4(self.down(en3))
        
        x = self.se_eca(en4)
        x = self.buttom_conv(self.down(x))

        x = self.de_conv4(torch.concat([en4,self.up(x)],dim=1))
        x = self.de_conv3(torch.concat([en3,self.up(x)],dim=1))
        x = self.de_conv2(torch.concat([en2,self.up(x)],dim=1))
        x = self.de_conv1(torch.concat([en1,self.up(x)],dim=1))

        x = self.out_conv(x)
        return x

class Generator_Unet_ECA(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.en_conv1 = ConvBlock(1,64)
        self.en_conv2 = ConvBlock(64,128)
        self.en_conv3 = ConvBlock(128,256)
        self.en_conv4 = ConvBlock(256,512)

        self.buttom_conv = ConvBlock(512,1024)
        
        self.de_conv4 = ConvBlock(1024+512,512)
        self.de_conv3 = ConvBlock(512+256,256)
        self.de_conv2 = ConvBlock(256+128,128)
        self.de_conv1 = ConvBlock(128+64,64)

        self.out_conv = nn.Sequential(
            nn.Conv2d(64,1,1,1),
            nn.Sigmoid()
            )
        self.se_eca = ECA(512)
    def up(self,x):
        _,_,h,w = x.shape
        return F.interpolate(x, size=(int(h*2),int(w*2)), mode="bilinear")
    def down(self, x):
        return F.max_pool2d(x, kernel_size=2)

    def forward(self,x):
        en1 = self.en_conv1(x)
        en2 = self.en_conv2(self.down(en1))
        en3 = self.en_conv3(self.down(en2))
        en4 = self.en_conv4(self.down(en3))
        
        x = self.se_eca(en4)
        x = self.buttom_conv(self.down(x))

        x = self.de_conv4(torch.concat([en4,self.up(x)],dim=1))
        x = self.de_conv3(torch.concat([en3,self.up(x)],dim=1))
        x = self.de_conv2(torch.concat([en2,self.up(x)],dim=1))
        x = self.de_conv1(torch.concat([en1,self.up(x)],dim=1))

        x = self.out_conv(x)
        return x


class Generator_Unet_SEECA(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.en_conv1 = ConvBlock(1,64)
        self.en_conv2 = ConvBlock(64,128)
        self.en_conv3 = ConvBlock(128,256)
        self.en_conv4 = ConvBlock(256,512)

        self.buttom_conv = ConvBlock(512,1024)
        
        self.de_conv4 = ConvBlock(1024+512,512)
        self.de_conv3 = ConvBlock(512+256,256)
        self.de_conv2 = ConvBlock(256+128,128)
        self.de_conv1 = ConvBlock(128+64,64)

        self.out_conv = nn.Sequential(
            nn.Conv2d(64,1,1,1),
            nn.Sigmoid()
            )
        self.se_eca = nn.Sequential(
            SE(512),
            ECA(512)
        )
    def up(self,x):
        _,_,h,w = x.shape
        return F.interpolate(x, size=(int(h*2),int(w*2)), mode="bilinear")
    def down(self, x):
        return F.max_pool2d(x, kernel_size=2)

    def forward(self,x):
        en1 = self.en_conv1(x)
        en2 = self.en_conv2(self.down(en1))
        en3 = self.en_conv3(self.down(en2))
        en4 = self.en_conv4(self.down(en3))
        
        x = self.se_eca(en4)
        x = self.buttom_conv(self.down(x))

        x = self.de_conv4(torch.concat([en4,self.up(x)],dim=1))
        x = self.de_conv3(torch.concat([en3,self.up(x)],dim=1))
        x = self.de_conv2(torch.concat([en2,self.up(x)],dim=1))
        x = self.de_conv1(torch.concat([en1,self.up(x)],dim=1))

        x = self.out_conv(x)
        return x

class Generator_Unet_ASPP(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.en_conv1 = ConvBlock(1,64)
        self.en_conv2 = ConvBlock(64,128)
        self.en_conv3 = ConvBlock(128,256)
        self.en_conv4 = ConvBlock(256,512)

        self.buttom_conv = ConvBlock(512,1024)
        
        self.de_conv4 = ConvBlock(1024+512,512)
        self.de_conv3 = ConvBlock(512+256,256)
        self.de_conv2 = ConvBlock(256+128,128)
        self.de_conv1 = ConvBlock(128+64,64)

        self.out_conv = nn.Sequential(
            nn.Conv2d(64,1,1,1),
            nn.Sigmoid()
            )
        self.aspp = ASPP(1024,1024)
    def up(self,x):
        _,_,h,w = x.shape
        return F.interpolate(x, size=(int(h*2),int(w*2)), mode="bilinear")
    def down(self, x):
        return F.max_pool2d(x, kernel_size=2)

    def forward(self,x):
        en1 = self.en_conv1(x)
        en2 = self.en_conv2(self.down(en1))
        en3 = self.en_conv3(self.down(en2))
        en4 = self.en_conv4(self.down(en3))
        
        x = self.buttom_conv(self.down(en4))
        x = self.aspp(x)

        x = self.de_conv4(torch.concat([en4,self.up(x)],dim=1))
        x = self.de_conv3(torch.concat([en3,self.up(x)],dim=1))
        x = self.de_conv2(torch.concat([en2,self.up(x)],dim=1))
        x = self.de_conv1(torch.concat([en1,self.up(x)],dim=1))

        x = self.out_conv(x)
        return x


class Generator_ALL(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.en_conv1 = ConvBlock(1,64)
        self.en_conv2 = ConvBlock(64,128)
        self.en_conv3 = ConvBlock(128,256)
        self.en_conv4 = ConvBlock(256,512)

        self.buttom_conv = ConvBlock(512,1024)
        
        self.de_conv4 = ConvBlock(1024+512,512)
        self.de_conv3 = ConvBlock(512+256,256)
        self.de_conv2 = ConvBlock(256+128,128)
        self.de_conv1 = ConvBlock(128+64,64)

        self.out_conv = nn.Sequential(
            nn.Conv2d(64,1,1,1),
            nn.Sigmoid()
            )
        self.se_eca = nn.Sequential(
            SE(512),
            ECA(512)
        )
        self.aspp = ASPP(1024,1024)
    def up(self,x):
        _,_,h,w = x.shape
        return F.interpolate(x, size=(int(h*2),int(w*2)), mode="bilinear")
    def down(self, x):
        return F.max_pool2d(x, kernel_size=2)

    def forward(self,x):
        en1 = self.en_conv1(x)
        en2 = self.en_conv2(self.down(en1))
        en3 = self.en_conv3(self.down(en2))
        en4 = self.en_conv4(self.down(en3))
        
        x = self.se_eca(en4)
        x = self.buttom_conv(self.down(x))
        x = self.aspp(x)

        x = self.de_conv4(torch.concat([en4,self.up(x)],dim=1))
        x = self.de_conv3(torch.concat([en3,self.up(x)],dim=1))
        x = self.de_conv2(torch.concat([en2,self.up(x)],dim=1))
        x = self.de_conv1(torch.concat([en1,self.up(x)],dim=1))

        x = self.out_conv(x)
        return x

