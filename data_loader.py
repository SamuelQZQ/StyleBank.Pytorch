import torch
import torch.utils.data as data
import os
import numpy as np
import nltk
import random
from PIL import Image


dataDir = 'data_contents/VIPSL_FaceSketch/'
contentDir = dataDir + 'photos/'
styleDir0 = dataDir + 'sketches/'
styleDir1 = dataDir + 'sketches1/'


class VIPSLDataset(data.Dataset):

    def __init__(self, transform):
        self.transform = transform
        filenames = os.listdir(contentDir)
        self.len = len(filenames)

    def __getitem__(self, index):
        index = index + 1
        contentFile = str(index) + '.jpg'
        img = Image.open(contentDir + contentFile)
        img = self.transform(img)

        styleFile0 = str(index).zfill(4) + '.jpg'
        style0 = Image.open(styleDir0 + styleFile0)

        styleFile1 = str(index) + '.jpg'
        style1 = Image.open(styleDir1 + styleFile1)

        style0 = self.transform(style0)
        style1 = self.transform(style1)

        return img, style0, style1

    def __len__(self):
        return 200


def collate_fn(data):

    images, styles0, styles1 = zip(*data)

    # Merge images (from tuple of 1D tensor to 2D tensor).
    images = torch.stack(images, 0)
    styles0 = torch.stack(styles0, 0)
    styles1 = torch.stack(styles1, 0)

    return images, styles0, styles1