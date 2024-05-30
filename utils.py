import glob
import json
import os
from tkinter import Image

from cv2 import imwrite, resize
import numpy as np
import torch
import tqdm


def overlay_masks(images_dir, subdir_name, target_dir):
    train_dir = os.path.join(images_dir, subdir_name)
    for mask_dirname in tqdm(glob.glob('{}/*/masks'.format(train_dir))):
        masks = []
        for image_filepath in glob.glob('{}/*'.format(mask_dirname)):
            image = np.asarray(Image.open(image_filepath))
            image = np.where(image > 0, 1, 0)
            masks.append(image)
        overlayed_masks = np.sum(masks, axis=0)
        overlayed_masks = np.where(overlayed_masks > 0, 1, 0)
        target_filepath = '/'.join(mask_dirname.replace(images_dir, target_dir).split('/')[:-1]) + '.png'
        os.makedirs(os.path.dirname(target_filepath), exist_ok=True)
        imwrite(target_filepath, overlayed_masks)

def preprocess_image(img, target_size=(128, 128)):
    img = resize(img, target_size, mode='constant')
    x = np.expand_dims(img, axis=0)
    x = x.transpose(0, 3, 1, 2)
    x = torch.FloatTensor(x)
    if torch.cuda.is_available():
        x = torch.autograd.Variable(x, volatile=True).cuda()
    else:
        x = torch.autograd.Variable(x, volatile=True)
    return x

def save_target_masks(target_filepath, *masks):
    with open(target_filepath, 'w') as file:
        json.dump(masks, file)