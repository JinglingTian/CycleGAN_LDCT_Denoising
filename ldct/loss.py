import torch
import torch.nn as nn 
import torch.nn.functional as F
import torchvision.models as v_models
import torch.fft
from .net.block import CL_Model

class L1_Perc_Loss(nn.Module):
    def __init__(self, perceptual_weight=0.8, l1_weight=0.2):
        super(L1_Perc_Loss, self).__init__()
        self.layers = set([4,9,16,23,30])
        self.vgg = v_models.vgg16(weights=v_models.VGG16_Weights.IMAGENET1K_V1)
        self.vgg = nn.ModuleList(list(self.vgg.features.children())[:31])
        self.vgg.eval()
        for param in self.vgg.parameters():
            param.requires_grad = False
        self.perceptual_weight = perceptual_weight
        self.l1_weight = l1_weight
        self.l1_loss = nn.L1Loss()
        
    def forward(self, pred, target):
        pred = torch.concat([pred,pred,pred],1)
        target = torch.concat([target,target,target],1)
        l1_loss = self.l1_loss(pred, target)

        perceptual_loss = self.compute_perceptual_loss(pred, target)

        
        mixed_loss = self.perceptual_weight * perceptual_loss + self.l1_weight * l1_loss
        
        return mixed_loss
    
    def compute_perceptual_loss(self, pred, target):
        vgg_input_features = self.get_features(pred, self.layers)
        vgg_target_features = self.get_features(target, self.layers)
        loss = 0
        for layer in self.layers:
            loss += nn.MSELoss()(vgg_input_features[layer], vgg_target_features[layer])
        
        return loss
    
    def get_features(self, images, layers):
        features = {}
        for i, module in enumerate(self.vgg):
            images = module(images)
            if i in layers:
                features[i] = images
        return features


