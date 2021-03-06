import os
import numpy as np
import matplotlib.pyplot as plt

import torch 
import torchvision
import cv2 as cv

from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor
from torchvision.utils import make_grid

class Preprocessing():
    
    def __init__(self, mainFolder = "../dataSet/"):
        self.mainFolder = mainFolder

    def CheckImages(self, folderPath):
        folder = os.path.join(self.mainFolder, folderPath)
        images = []
        for filename in os.listdir(folder):
            image = cv.imread(os.path.join(folder, filename))
            if image.shape != (50, 50, 3):
                print(f"{os.path.join(folder, filename)}\t {image.shape}")
                os.remove(os.path.join(folder, filename))
            

    def DataReader(self, folder):
        dataSet = ImageFolder(self.mainFolder + folder, transform = ToTensor())
        return dataSet

    def AppendIndicies(self, data):
        indicies = []
        for i in range(0, len(data) - 1):
            indicies.append(i)
        return indicies
    
    def DataLoader(self, indicies, data, batchSize = 32):
        sampler = SubsetRandomSampler(indicies)
        dataLoader = DataLoader(data, batchSize, sampler = sampler)
        return dataLoader

    def ShowImage(self, image):
        plt.imshow(image.permute(1, 2, 0))
    
    def ShowGrid(self, dataLoader):
        for image, labels in dataLoader:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.set_xticks([]); ax.set_yticks([])
            ax.imshow(make_grid(image, 8).permute(1, 2, 0))
            break
