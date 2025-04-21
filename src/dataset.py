import os
import skimage
import numpy as np
from PIL import Image
from skimage.io import imread
from skimage.color import rgb2lab

import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

class ColorizationDataset(Dataset):
    def __init__(self, dir_dataset, image_size=320, is_train_set=True):
        """
        ---------
        Arguments
        ---------
        dir_dataset : str
            full directory path of dataset (train or valid) containing images
        image_size : int
            image size to be used for training
        is_train_set : bool
            boolean to control whether to generate a train or validation dataset object
        """
        self.dir_dataset = dir_dataset
        self.image_size = image_size
        self.list_images = sorted(os.listdir(self.dir_dataset))

        if is_train_set:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((self.image_size, self.image_size), Image.BILINEAR),
                transforms.RandomHorizontalFlip()
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((self.image_size, self.image_size), Image.BILINEAR)
            ])

    def __len__(self):
        # return number of images
        return len(self.list_images)

    def __getitem__(self, idx):
        img_rgb = imread(os.path.join(self.dir_dataset, self.list_images[idx]))
        img_rgb = self.transform(img_rgb)

        # convert to numpy array
        img_rgb = np.array(img_rgb)
        img_rgb = np.transpose(img_rgb, axes=(1, 2, 0))

        # convert rgb to lab image
        img_lab = rgb2lab(img_rgb).astype(np.float32)
        img_lab = transforms.ToTensor()(img_lab)

        # preprocess l channel, img_l belongs to [-1, 1]
        img_l = img_lab[[0], ...] / 50. - 1.

        # repeat l channel 3 times for ResNet
        img_l = torch.repeat_interleave(img_l, 3, dim=0)

        # preprocess ab channel, img_ab belongs to [-1, 1]
        img_ab = img_lab[[1, 2], ...] / 110.

        # return a dict of domain_1 and domain_2 images
        # domain_1 is l channel and
        # domain_2 is ab channel
        return {"domain_1": img_l, "domain_2": img_ab, "file_name": self.list_images[idx]}

def get_dataset_loader(dir_dataset, image_size=320, batch_size=8, is_train_set=True):
    """
    ---------
    Arguments
    ---------
    dir_dataset : str
        full directory path of dataset (train or valid) containing images
    image_size : int
        image size to be used for training
    is_train_set : bool
        boolean to control whether to generate a train or validation dataset object
    batch_size : int
        batch size to be used to generate train dataset

    -------
    Returns
    -------
    DataLoader object
    """
    dataset = ColorizationDataset(dir_dataset, image_size=image_size, is_train_set=is_train_set)
    print(f"Num images in the train dataset : {dataset.__len__()}")
    dataset_loader = DataLoader(dataset, batch_size=batch_size, shuffle=is_train_set)
    return dataset_loader
