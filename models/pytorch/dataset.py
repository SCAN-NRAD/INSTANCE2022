import os
import tqdm
from glob import glob
import nibabel as nib
import numpy as np
import itertools
import torch
import random
from torch.utils.data import Dataset
from batchgenerators.augmentations.utils import pad_nd_image
import scipy.ndimage as ndi
import logging

class INSTANCE_2022(Dataset):
    def __init__(self, cases_file = None, patch_size = 128):

        self.base_dir = '/str/nas/INSTANCE2022/'

        if cases_file is not None:
            with open(cases_file) as f:
                self.ids = [line.rstrip('\n') for line in f]
        else:
            #Load entire dataset
            self.ids = os.listdir(os.path.join(self.base_dir,'rawdata'))

        self.patch_size = patch_size

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        idx = self.ids[i]


        img_file = glob(os.path.join(self.base_dir, f'train_2/data/{idx}'))
        label_file = glob(os.path.join(self.base_dir, f'train_2/label/{idx}'))

        assert len(label_file) == 1, \
            f'Either no label or multiple labels found for the ID {idx}: {label_file}'
        assert len(img_file) == 1, \
            f'Either no image or multiple images found for the ID {idx}: {img_file}'

        img = nib.load(img_file[0])
        label = nib.load(label_file[0])
        img_affine = img.affine

        assert (img.affine.round() == label.affine.round()).prod(), \
            f'Image and label {idx} affines should be the same'

        assert img.shape == label.shape, \
            f'Image and label {idx} should be the same shape, but are {img.shape} and {label.shape}'
        
        img = img.get_fdata()
        label =label.get_fdata()

        #clamp and scale
        img[img<0] = 0
        img[img>80] = 80
        img = img / 80

        #get a random patch inside the volume of size (patch_size,patch_size,patch_size)  
        if self.patch_size > 0:
            img, label  = self.random_patch(img,label,self.patch_size,check_labels=False)

        img = np.expand_dims(img, 0)

        img = torch.from_numpy(img).type(torch.float32)
        label = torch.from_numpy(label).long()

        return {
            'image': img,
            'label': label,
            'name' : idx,
            'affine': img_affine
        }

    def random_patch(self,input_array,input_label,patch_size,check_labels=False):
        x,y,z = input_array.shape

        patchFound = False
        while not patchFound:
            min_x = min_y = min_z = 0 
            max_x, max_y, max_z = x,y,z
            
            if x > patch_size:
                min_x = np.random.randint(x - patch_size+1)
                max_x = min_x+patch_size
            if y > patch_size:
                min_y = np.random.randint(y - patch_size+1)
                max_y = min_y+patch_size
            if z > patch_size:
                min_z = np.random.randint(z - patch_size+1)
                max_z = min_z+patch_size

            output_array = pad_nd_image(input_array[min_x:max_x,min_y:max_y,min_z:max_z],(patch_size,patch_size,patch_size))
            output_label = pad_nd_image(input_label[min_x:max_x,min_y:max_y,min_z:max_z],(patch_size,patch_size,patch_size))

            if check_labels == False:
                patchFound = True
            else:
                for i in [-1,0,1]:
                    if i == 0:
                        continue
                    if (output_label == i).sum() == 0:
                        print('missing label. applying new random patch')
                        break
                    patchFound = True
        
        return output_array, output_label
