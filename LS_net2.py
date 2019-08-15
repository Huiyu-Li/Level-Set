import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import time
filters = [np.eye(3),
       np.array([[0, 1, 0]] * 3),
       np.flipud(np.eye(3)),
       np.rot90([[0, 1, 0]] * 3)]
filters = torch.Tensor(filters)
def checkerboard_level_set(image_shape, square_size=5):
    grid = np.mgrid[[slice(i) for i in image_shape]]
    grid = (grid // square_size)

    # Alternate 0/1 for even/odd numbers.
    grid = grid & 1

    checkerboard = np.bitwise_xor.reduce(grid, axis=0)
    res = np.int8(checkerboard)
    return res

def conv_dilation(input,filter):
    if len(np.unique(filter.size))>1:
        raise ValueError("Filter must be square!")
    size = filter.shape[0]#square filter size
    Pad = nn.ConstantPad2d(1, 0)
    paded = Pad(input)
    pos = torch.zeros((input.size()[0] * size, input.size()[1] * size))
    neg = torch.zeros((input.size()[0] * size, input.size()[1] * size))
    for i in range(input.size()[0]):
        for j in range(input.size()[1]):
            pos[i * size:(i + 1) * size, j * size:(j + 1) * size] = paded[i:i + size, j:j + size]
            neg[i * size:(i + 1) * size, j * size:(j + 1) * size] = filter
    add = pos + neg
    MaxPool2d = nn.MaxPool2d(size, stride=3, padding=0)
    out = MaxPool2d(add.view(1, 1, add.size()[0], add.size()[1]))
    return out[0,0,:,:]#后期要改掉

def conv_erosion(input,filter):
    if len(np.unique(filter.size))>1:
        raise ValueError("Filter must be square!")
    size = filter.size()[0]#square filter size
    Pad = nn.ConstantPad2d(1, 0)
    paded = Pad(input)
    pos = torch.zeros((input.size()[0] * size, input.size()[1] * size))
    neg = torch.zeros((input.size()[0] * size, input.size()[1] * size))
    for i in range(input.size()[0]):
        for j in range(input.size()[1]):
            pos[i * size:(i + 1) * size, j * size:(j + 1) * size] = paded[i:i + size, j:j + size]
            neg[i * size:(i + 1) * size, j * size:(j + 1) * size] = -filter
    add = pos + neg
    MaxPool2d = nn.MaxPool2d(size, stride=3, padding=0)
    out = -MaxPool2d(-add.view(1, 1, add.size()[0], add.size()[1]))
    return out[0,0,:,:]#后期要改掉

class LSNet2(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,input,origin):
        #Length(看是否需要调回原序？)
        #inf_sup np.array(dilations).min(0)
        dilations = torch.empty(1,input.shape[0],input.shape[1])
        for filter in filters:
            temp = conv_dilation(input, filter).view(1,input.shape[0],input.shape[1])
            dilations = torch.cat((dilations,temp))
        input = torch.min(dilations,dim=0)[0]
        # sup_inf np.array(erosions).max(0)
        erosions = torch.empty(1,input.shape[0],input.shape[1])
        for filter in filters:
            temp = conv_erosion(input, filter).view(1,input.shape[0],input.shape[1])
            erosions = torch.cat((erosions,temp))
        input = torch.max(erosions, dim=0)[0]
        # print(input.shape)
        # sup_inf
        erosions = torch.empty(1, input.shape[0], input.shape[1])
        for filter in filters:
            temp = conv_erosion(input, filter).view(1, input.shape[0], input.shape[1])
            erosions = torch.cat((erosions, temp))
        input = torch.max(erosions, dim=0)[0]
        # inf_sup
        dilations = torch.empty(1, input.shape[0], input.shape[1])
        for filter in filters:
            temp = conv_dilation(input, filter).view(1, input.shape[0], input.shape[1])
            dilations = torch.cat((dilations, temp))
        input = torch.min(dilations, dim=0)[0]
        # inf_sup
        dilations = torch.empty(1, input.shape[0], input.shape[1])
        for filter in filters:
            temp = conv_dilation(input, filter).view(1, input.shape[0], input.shape[1])
            dilations = torch.cat((dilations, temp))
        input = torch.min(dilations, dim=0)[0]
        # sup_inf
        erosions = torch.empty(1, input.shape[0], input.shape[1])
        for filter in filters:
            temp = conv_erosion(input, filter).view(1, input.shape[0], input.shape[1])
            erosions = torch.cat((erosions, temp))
        input = torch.max(erosions, dim=0)[0]
        #Area
        v = 1
        filter = np.ones((3,) * len(origin.shape))
        filter = torch.from_numpy(filter)
        # print('filter',filter)
        if v > 0:
            aux = conv_dilation(input, filter)
        elif v < 0:
            aux = conv_erosion(input, filter)
        if v != 0:
            threshold = np.percentile(input, 40)
            threshold_mask_balloon = origin > threshold / np.abs(v)
            input[threshold_mask_balloon] = aux[threshold_mask_balloon]
        #Region
        c0 = (origin * (1 - input)).sum() / float((1 - input).sum() + 1e-8)
        c1 = (origin * input).sum() / float(input.sum() + 1e-8)
        aux = ((origin - c1) ** 2 - (origin - c0) ** 2)
        input[aux < 0] = 1
        input[aux > 0] = 0
        return input


import matplotlib.pyplot as plt
from skimage import img_as_float32
import time
import os
import shutil
import cv2

def main():
    filePath = './Img/'
    savePath = './MCVSeg/'
    if os.path.isdir(savePath):
        shutil.rmtree(savePath)
    os.mkdir(savePath)

    fileList = os.listdir(filePath)

    for i in range(len(fileList)):  # len(fileList)
        Image = cv2.imread(os.path.join(filePath, fileList[i]), 1)  # 读入原图
        image = cv2.cvtColor(Image, cv2.COLOR_BGR2GRAY)
        image = img_as_float32(image)#*0.5

        init_level_set = checkerboard_level_set(image.shape, square_size=5)
        input = (init_level_set > 0)*1.

        model = LSNet2()
        input = torch.from_numpy(input)
        image = torch.from_numpy(image)
        ls = input
        for _ in range(3):
            ls = model(ls,image)#just one iteration

        fig = plt.figure()
        plt.imshow(image, cmap="gray")
        plt.axis('off')
        plt.contour(ls, [0.5], colors='r')
        plt.title(fileList[i]+'MAC', fontsize=12)

        fig.tight_layout()
        plt.savefig('./MCVSeg/' + fileList[i] + '.png')
        # plt.show(block=False)

if __name__ == '__main__':
    start = time.time()
    main()
    print('Time:{:.3f}'.format((time.time()-start)/60))