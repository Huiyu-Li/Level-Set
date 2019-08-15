import SimpleITK as sitk
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import time

class LSNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.in_layer = nn.Sequential(
            nn.Conv3d(1, 2, kernel_size=3, padding=1),
            nn.BatchNorm3d(2),
            nn.Tanh(),
            nn.Conv3d(2, 1, kernel_size=3, padding=1),
            nn.BatchNorm3d(1),
            nn.Tanh()
        )
        self.conv = nn.Sequential(
            nn.Conv3d(1,9, kernel_size=3, padding=1),
            nn.BatchNorm3d(9),#27
            nn.Tanh()
        )
        self.out_layer = nn.Sequential(
            nn.Conv3d(1,2, kernel_size=3, padding=1),
            nn.BatchNorm3d(2),
            nn.Softmax(dim=1),#NCDHW
            nn.Conv3d(2, 1, kernel_size=3, padding=1),
            nn.BatchNorm3d(1),
            nn.Tanh()
        )
        self.conv3d_1 = nn.Conv3d(1, 1, kernel_size=3, padding=1)
        self.conv3d_3 = nn.Conv3d(1, 3, kernel_size=3, padding=1)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=1, padding=1)
        self.Softshrink = nn.Softshrink(lambd=0.5)
        self.relu = nn.ReLU()#inplace=True
        self.tanh = nn.Tanh()

    def forward(self,x,input):
        #Length(看是否需要调回原序？)
        #inf_sup
        # x = self.in_layer(x)
        x = self.tanh(x)

        x = self.conv(x)
        x = self.maxpool(x)#need to check the shape carefully

        x = torch.min(x,1,keepdim=True)[1].data.float()
        # sup_inf
        x = self.conv(x)#会不会共用权重？
        filter = -1.*torch.ones((9,9,1,1,1))
        if torch.cuda.is_available():
            filter = filter.cuda()
        x = F.conv3d(x, filter)
        x = self.maxpool(x)  # need to check the shape carefully
        x = torch.max(x, 1, keepdim=True)[1].data.float()
        # sup_inf
        x = self.conv(x)
        filter = -1. * torch.ones((9, 9, 1, 1, 1))
        if torch.cuda.is_available():
            filter = filter.cuda()
        x = F.conv3d(x, filter)
        x = self.maxpool(x)  # need to check the shape carefully
        x = torch.max(x, 1, keepdim=True)[1].data.float()
        # inf_sup
        x = self.conv(x)
        x = self.maxpool(x)  # need to check the shape carefully
        x = torch.min(x, 1, keepdim=True)[1].data.float()
        # inf_sup
        x = self.conv(x)
        x = self.maxpool(x)  # need to check the shape carefully
        x = torch.min(x, 1, keepdim=True)[1].data.float()
        # sup_inf
        x = self.conv(x)
        filter = -1. * torch.ones((9, 9, 1, 1, 1))
        if torch.cuda.is_available():
            filter = filter.cuda()
        x = F.conv3d(x, filter)
        x = self.maxpool(x)  # need to check the shape carefully
        x = torch.max(x, 1, keepdim=True)[1].data.float()
        # print('Length',x.shape)
        #Area
        v = self.Softshrink(x)#v怎行和x挂钩？
        if torch.sum(v) > 0:
            # print('torch.sum(v)> 0')
            x= self.conv3d_1(x)
            x = self.maxpool(x)# need to check the shape carefully
        elif torch.sum(v) < 0:
            # print('torch.sum(v) < 0')
            x = self.conv3d_1(x)
            filter = -1. * torch.ones((9, 9, 1, 1, 1))
            if torch.cuda.is_available():
                filter = filter.cuda()
            x = F.conv3d(x, filter)
            x = self.maxpool(x)  # need to check the shape carefully
        else:
            x = x
        # print('Area',x.shape)
        #Region
        u = self.relu(x)#??
        Hphi = (u > 0).float()
        Hinv = 1. - Hphi
        c1 = torch.sum(input * Hphi)/(torch.sum(Hphi)+1e-8)
        c2 = torch.sum(input * Hinv)/(torch.sum(Hinv)+1e-8)
        # print(c1.shape,c2.shape)
        avg_inside = (input - c1) ** 2
        avg_oustide = (input - c2) ** 2
        # print(avg_inside.shape, avg_oustide.shape)
        avg_inside = self.conv3d_1(avg_inside)
        avg_oustide = self.conv3d_1(avg_oustide)
        # print(avg_inside.shape, avg_oustide.shape)
        x = self.tanh(avg_oustide-avg_inside)

        # x = self.out_layer(x)
        return x
class diceLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.smoth = 1e-5
    def forward(self, output, target):
        smooth = 1.
        # have to use contiguous since they may from a torch.view op
        iflat = output.contiguous().view(-1)
        tflat = target.contiguous().view(-1)
        intersection = (iflat * tflat).sum()

        A_sum = torch.sum(tflat * iflat)
        B_sum = torch.sum(tflat * tflat)
        return 1 - ((2. * intersection + smooth) / (A_sum + B_sum + smooth))
def main():
    ct_array = np.ones((4,8,8))
    model = LSNet()
    input = torch.from_numpy(ct_array).view(1, 1, 4, 8, 8).float()
    out = model(input,input)

if __name__ == '__main__':
    main()