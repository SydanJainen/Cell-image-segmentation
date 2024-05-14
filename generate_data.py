import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
from torchviz import make_dot
import numpy as np
from torch.utils.data import Dataset
import os
from PIL import Image
from skimage.io import imread,imshow
from skimage.transform import resize
from tqdm import tqdm


TRAIN_PATH = 'dataset/stage1_train/'

train_files = next(os.walk(TRAIN_PATH))[1]

X_train = np.zeros((len(train_files), 128, 128, 3), dtype = np.uint8)
Y_train = np.zeros((len(train_files), 128, 128, 1))

img_height = 256
img_width = 256
img_channel = 3

def preprocess(id_, path):
    """
    input: list of ids, path of directory
    output: return X_train and y_train
    """
    
    # initialize two empty array to store
    # size is (# of training instance, img_size, img_size, img_channel)
    X_train = np.zeros((len(id_), img_height, img_width, img_channel), dtype = np.uint8)
    Y_train = np.zeros((len(id_), img_height, img_width, 1))
    
    # iterate through all the training img, save each training instance into X_train
    # using tqdm is good for us to visualize the process
    for n, id_ in tqdm(enumerate(id_), total = len(id_)):   
        cur_path = path + id_
        # read in img as array
        img = imread(cur_path + '/images/' + id_ + '.png')[:,:,:img_channel]  
        # resize data to increase the speed of training
        img = resize(img, (img_height, img_width), mode='constant', preserve_range=True)
        # save current img into X_train
        X_train[n] = img  
        # for each img, we have several masks
        # we need to iterate through each one 
        mask = np.zeros((img_height, img_width, 1))
        for mask_file in os.listdir(cur_path + '/masks/'):
            # read in current mask
            cur_mask = imread(cur_path + '/masks/' + mask_file)
            # resize it and adjust the dimension to 128x128x1
            cur_mask = np.expand_dims(resize(cur_mask, (img_height, img_width), mode = 'constant', preserve_range = True), axis = -1)
            mask = np.maximum(mask, cur_mask)
        Y_train[n] = mask
    return X_train, Y_train

X_train, Y_train = preprocess(train_files, TRAIN_PATH)

# With matplot lib show me the first image and its mask
plt.imshow(X_train[0])
plt.show()
plt.imshow(np.squeeze(Y_train[0]))
plt.show()


class CellularDataSet(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images and masks.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.images_dir = os.path.join(root_dir, 'images')
        self.masks_dir = os.path.join(root_dir, 'masks')
        self.image_names = [f for f in os.listdir(self.images_dir) if os.path.isfile(os.path.join(self.images_dir, f))]

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_name = os.path.join(self.images_dir, self.image_names[idx])
        image = Image.open(img_name).convert('RGB')
        
        # Assuming mask names match the image names but are stored as multiple files per image
        mask_path = os.path.join(self.masks_dir, self.image_names[idx].split('.')[0])
        mask_list = [os.path.join(mask_path, f) for f in os.listdir(mask_path) if os.path.isfile(os.path.join(mask_path, f))]
        mask = None
        
        # Load masks and concatenate them
        for m in mask_list:
            m_img = Image.open(m).convert('L')
            if mask is None:
                mask = np.array(m_img, dtype=np.uint8)
            else:
                mask = np.maximum(mask, np.array(m_img, dtype=np.uint8))
        
        mask = Image.fromarray(mask)
        
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask