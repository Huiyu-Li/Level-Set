import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import os
import cv2
from skimage import data, img_as_float
from sklearn import preprocessing

class LSDataloader(Dataset):
    def __init__(self,dir_csv):
        """
        :param csv_file:path to all the images
        """
        self.image_dirs = pd.read_csv(dir_csv,header=None).iloc[:, :].values#from DataFrame to array
        self.channels = 1
        self.image_depth = 1
        self.image_height = 369
        self.image_width = 369

    def __len__(self):
        return len(self.image_dirs)

    def __getitem__(self, item):
        # (NCDHW),C must be added and N must be removed since added byitself
        input = np.empty((self.channels, self.image_depth, self.image_height,self.image_width)).astype(np.float32)
        target = np.empty((self.channels, self.image_depth, self.image_height, self.image_width)).astype(np.float32)
        Image = cv2.imread(self.image_dirs[item][0], 1)  # 读入原图
        Image = cv2.cvtColor(Image, cv2.COLOR_BGR2GRAY)
        image = img_as_float(Image)
        ct = preprocessing.scale(image,axis=0)
        # print(ct.mean(),ct.std())
        seg = Image.copy()
        # print(np.unique(seg))#[0 128 255]
        seg[seg==128] = 255
        input[0,0,:,:] = ct
        target[0,0,:,:] = seg

        # show for test
        # import matplotlib.pyplot as plt
        # plt.figure()
        # plt.subplot(121)
        # plt.imshow(ct);plt.title('MyDataloader ct')
        # plt.subplot(122)
        # plt.imshow(seg);plt.title('MyDataloader seg')
        # plt.show()

        # sample = {'image':item_image,'label':item_label}#false code!
        return (input,target)
