import matplotlib.pyplot as plt

from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor
from torchvision.utils import make_grid

class DataImport():
    def __init__(self, mainDir="./dataSet/"):
        self.mainDir = mainDir

    def Import(self, folder):
        dataSet = ImageFolder(self.mainDir + folder, transform = ToTensor())
        return dataSet

    def Show(self, imageTensor):
        return plt.imshow(imageTensor.permute(1, 2, 0))

    def ShowGrid(self, dataLoader):
        for image, label in dataLoader:
            fig, ax = plt.subplots(figsize = (10, 10))
            ax.set_xticks([]); ax.set_yticks([])
            ax.imshow(make_grid(image, 10).permute(1, 2, 0))
            break
