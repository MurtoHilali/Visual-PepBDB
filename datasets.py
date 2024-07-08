import os
import torch
import numpy as np
import random
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class PepBDB_dataset(Dataset):
    
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.image_list = []
        self.label_list = []
        
        binding_dir = os.path.join(root_dir, 'binding')
        nonbinding_dir = os.path.join(root_dir, 'nonbinding')
        
        for img_name in os.listdir(binding_dir):
            if img_name.endswith('.jpg'):
                self.image_list.append(os.path.join(binding_dir, img_name))
                self.label_list.append(1)  # Binding class

        for img_name in os.listdir(nonbinding_dir):
            if img_name.endswith('.jpg'):
                self.image_list.append(os.path.join(nonbinding_dir, img_name))
                self.label_list.append(0)  # Nonbinding class
        
        self.image_list, self.label_list = self.shuffle_lists(self.image_list, self.label_list)
        
        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor()
        ])
        
    def shuffle_lists(self, l1, l2):
        random.seed(4)
        mapIndexPosition = list(zip(l1, l2))
        random.shuffle(mapIndexPosition)
        l1, l2, = zip(*mapIndexPosition)
        return list(l1), list(l2)
    
    def __getitem__(self, index):
        img_path = self.image_list[index]
        image = Image.open(img_path)
        image = self.transform(image)
        label = torch.tensor(self.label_list[index], dtype=torch.long)
        return image, label
    
    def __len__(self):
        return len(self.label_list)