import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as v_models

class Conv_BN_ReLu(nn.Module):
    def __init__(self,in_channels, out_channels, kernel_size, stride=1, padding=0,dilation=1,groups=1) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding,dilation,groups),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
    def forward(self,x):
        return self.conv(x)


class ConvBlock(nn.Module):
    def __init__(self, in_ch,out_ch,use_res=True) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch,out_ch,3,1,1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Conv2d(out_ch,out_ch,3,1,1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )
        self.res_conv = nn.Conv2d(in_ch,out_ch,1) if use_res else None
    def forward(self,x):
        if self.res_conv is None:
            return self.conv(x)
        else:
            return self.conv(x)+self.res_conv(x)

class SE(nn.Module):
    def __init__(self,in_ch) -> None:
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fex = nn.Sequential(
            nn.Conv2d(in_ch,in_ch,1),
            nn.ReLU()
        )
    def forward(self,x):
        x = x*self.fex(self.avg_pool(x))
        return x

class ECA(nn.Module):
    def __init__(self,in_ch) -> None:
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.att_conv = nn.Sequential(
            nn.Conv1d(1,8,3,1,1),
            nn.ReLU(),
            nn.Conv1d(8,16,3,1,1),
            nn.ReLU(),
            nn.Conv1d(16,1,3,1,1),
            nn.ReLU()
        )
        
    def forward(self,x):
        score = self.avg_pool(x).permute(0,2,3,1).squeeze(1)
        score = self.att_conv(score).unsqueeze(1).permute(0,3,1,2)
        return x*score

# 残差块
class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()
        conv_block = [
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
        ]
        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)



class Image_Pooling(nn.Module):
    def __init__(self,in_ch,out_ch) -> None:
        super().__init__()
        self.adapt_pool = nn.AdaptiveAvgPool2d((1,1))
        self.conv = Conv_BN_ReLu(in_ch,out_ch,1)
    def up_sample(self,x,size):
        return F.interpolate(x,size)

    def forward(self,x):
        size = x.shape[-2:]
        x = self.adapt_pool(x)
        x = self.conv(x)
        x = self.up_sample(x,size)
        return x

class ASPP(nn.Module):
    def __init__(self, in_ch, out_ch) -> None:
        super().__init__()
        self.conv_list = nn.ModuleList([
            Conv_BN_ReLu(in_ch,out_ch,1,1,0),
            Conv_BN_ReLu(in_ch,out_ch,3,1,6,dilation=6),
            Conv_BN_ReLu(in_ch,out_ch,3,1,12,dilation=12),
            Conv_BN_ReLu(in_ch,out_ch,3,1,18,dilation=18),
            Image_Pooling(in_ch,out_ch)
        ])
        self.conv = Conv_BN_ReLu(out_ch*len(self.conv_list),out_ch,1)

    def forward(self,x):
        results = []
        for c in self.conv_list:
            results.append(c(x))
        x = self.conv(torch.concat(results,1))
        return x

class CL_Model(nn.Module):
    def __init__(self,in_ch=1):
        super(CL_Model, self).__init__()
        self.in_conv = nn.Conv2d(in_ch,3,1)
        self.vgg = v_models.vgg16(weights=v_models.VGG16_Weights.IMAGENET1K_V1)
        self.vgg = nn.Sequential(*list(self.vgg.features.children())[:32])
        self.score = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2*512*8*8,1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024,512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512,256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256,1),
            nn.Sigmoid()
        )

    def forward(self, x1, x2):
        x1 = self.in_conv(x1)
        x2 = self.in_conv(x2)
        x1 = self.vgg(x1)
        x2 = self.vgg(x2)

        x = torch.concat([x1,x2],1)
        x = self.score(x)
        return x