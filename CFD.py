import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.io as sio
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
from torchvision import transforms
import random

class CFDDataset(Dataset):
    def __init__(self, images_dir, ground_truths_dir, batch_size=256):
        self.batch_size = batch_size
        self.data = []
        
        images = []
        ground_truths = []
        sizes = [128, 256, 320]
        for filename in os.listdir(images_dir)[:118]:
            images.append(Image.open(images_dir + '/' + filename))
        for filename in os.listdir(ground_truths_dir + '/processed'):
            ground_truths.append(Image.open(ground_truths_dir + '/processed/' + filename))

        for size in sizes:
            for i in range(len(images)):
                resized_image = images[i].copy()
                resized_image = resized_image.resize((size, size))
                resized_ground_image = ground_truths[i].copy()
                resized_ground_image = resized_ground_image.resize((size, size))
                self.data.append((resized_image, resized_ground_image))

        limit = len(self.data)
        lighting_value = [-0.05, 0.05]
        for i in range(limit):
            angle = random.randint(0, 360)
            lighting = random.choice(lighting_value)
            img, truth = self.data[i]
            img = TF.rotate(img, angle)
            truth = TF.rotate(truth, angle)
            if random.random() > 0.5:
                img = TF.hflip(img)
                truth = TF.hflip(truth)
            else:
                img = TF.vflip(img)
                truth = TF.vflip(truth)
            img = TF.adjust_brightness(img, lighting)
            truth = TF.adjust_brightness(truth, lighting)
            self.data.append((img, truth))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def get_loader(self):
        train_length = round(len(self.data)*0.6)
        train_set, test_set = torch.utils.data.random_split(self, [train_length, len(self.data) - train_length])
        return (DataLoader(train_set, batch_size=self.batch_size), DataLoader(test_set, batch_size=self.batch_size))

def process_ground_truths(ground_truths_dir):
    for filename in os.listdir(ground_truths_dir):
        mat_contents = sio.loadmat(ground_truths_dir + '/' + filename)
        img_name = filename[:-4] + '.png'
        imgplot = plt.imshow(mat_contents['groundTruth'][0][0][0])
        plt.axis('off')
        plt.savefig(ground_truths_dir + '/processed/' + img_name, bbox_inches='tight', pad_inches=0)
    print("Finished processing mat files of CFD")

def get_CFD_dataloader():
    images_dir = 'data/CFD/image'
    ground_truths_dir = 'data/CFD/groundTruth'
    dataset = CFDDataset(images_dir, ground_truths_dir)
    train_loader, test_loader = dataset.get_loader()
    print(len(dataset))
    print(dataset[0])


#process_ground_truths('data/CFD/groundTruth')
get_CFD_dataloader()
