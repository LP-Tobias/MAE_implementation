import numpy as np
from PIL import Image
import torch

class TransformedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, index):
        img, target = self.dataset[index]
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img)  # Ensure image is in PIL format before transformations
        img = self.transform(img)
        return img, target

    def __len__(self):
        return len(self.dataset)
