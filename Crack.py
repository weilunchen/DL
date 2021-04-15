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

class CrackDataset(Dataset):
    def __init__(self, images_dir, ground_truths, batch_size=256):
        self.batch_size = batch_size
        self.data = []

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def get_loader(self):
        return DataLoader(self, batch_size=self.batch_size)

def get_Crack_dataloader():
    train_dir = 'data/Crack500/traindata'
    test_dir = 'data/Crack500/testdata'
    val_dir = 'data/Crack500/valdata'
    dataset = CrackDataset(train_dir, val_dir, test_dir)
    train_loader, val_loader, test_loader = dataset.get_loader()
    print(len(dataset))
    print(dataset[0])

get_Crack_dataloader()
