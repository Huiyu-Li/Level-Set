#############reduce the number of training&valid data&test data##################
import os
import math
import numpy as np
import csv
savedct_path = './Img/'

train_csv = './train.csv'
valid_csv = './valid.csv'
# test_csv = '/home/lihuiyu/Documents/Segmentation/LiTS-Vnet3D/dataprocess/test.csv'

#clear the exists file
if os.path.isfile(train_csv):
    os.remove(train_csv)
if os.path.isfile(valid_csv):
    os.remove(valid_csv)

ct_lists = os.listdir(savedct_path)
num_file = len(ct_lists)
ratio = 0.8
tn = math.ceil(num_file * ratio)
print(num_file)

# shuffle(only for image and make the corresponding label when witer into csv)
perm = np.arange(len(ct_lists))
np.random.shuffle(perm)
ct_lists = np.array(ct_lists)[perm]

train_lists = ct_lists[0:tn]#attention:[0:num_train)
valid_lists = ct_lists[tn:num_file]

with open(train_csv, 'w') as file:
    w = csv.writer(file)
    # w.writerow(('Image','Label'))#attention: the first row defult to tile
    # or pd.read_csv(image_csv,header=None)#enable the first row by using defualt tile
    for name in train_lists:
        ct_name = os.path.join(savedct_path, name)
        # seg_name = os.path.join(savedseg_path, 'segmentation-' + name.split('-')[-1])
        w.writerow((ct_name,ct_name))

with open(valid_csv, 'w') as file:
    w = csv.writer(file)
    # w.writerow(('Image','Label'))#attention: the first row defult to tile
    # or pd.read_csv(image_csv,header=None)#enable the first row by using defualt tile
    for name in valid_lists:
        ct_name = os.path.join(savedct_path, name)
        # seg_name = os.path.join(savedseg_path, 'segmentation-' + name.split('-')[-1])
        w.writerow((ct_name,ct_name))