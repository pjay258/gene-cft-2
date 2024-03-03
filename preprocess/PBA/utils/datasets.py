from torch.utils.data import DataLoader, Dataset
import pandas as pd
from PIL import Image
from torchvision import transforms
from glob import glob
import os
import torch

class CmnistDataset(Dataset):
    def __init__(self, type, bias_ratio, transform=None , root='../../dataset/'):
        """
        type: 'train', 'valid', 'test' |
        bias_ratio: 1.0, 0.995, 0.99, 0.98, 0.95, 0.0, 'unbiased'
        """
        if type == 'test' and bias_ratio != 'unbiased':
            raise Exception("Test is unbiased.")
        if type not in ['train', 'valid', 'test']:
            raise Exception("Check type")
        if bias_ratio not in [1.0, 0.995, 0.99, 0.98, 0.95, 0.0, 'unbiased']:
            raise Exception("Check bias_ratio")

        super().__init__()
        self.dataset_path = f'{root}/cmnist/{str(bias_ratio)}/{str(type)}'
        self.labels_csv = pd.read_csv(f'{root}/cmnist/{str(bias_ratio)}/{str(type)}.csv')
        if transform == None:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        else:
            self.transform = transform

    def __len__(self):
        return len(self.labels_csv)
    
    def __getitem__(self, index):
        sample = Image.open(f'{self.dataset_path}/{index}.jpg')
        sample = self.transform(sample)
        label = self.labels_csv['label'][index]

        return sample, label
    

class bFFHQDataset(Dataset):
    def __init__(self, split, transform, use_generated, root='../../dataset/'):
        super().__init__()
        self.transform = transform
        self.root = root

        if split=='train':
            self.align = glob(os.path.join(root+'bffhq/0.5pct/','align',"*","*"))
            self.conflict = glob(os.path.join(root+'bffhq/0.5pct/','conflict',"*","*"))
            if use_generated:
                self.generated_align = glob(os.path.join(root+'generated_bffhq/0.5pct/','align',"*","*"))
                self.generated_conflict = glob(os.path.join(root+'generated_bffhq/0.5pct/','conflict',"*","*"))
                self.data = self.align + self.conflict + self.generated_align + self.generated_conflict
            else:
                self.data = self.align + self.conflict

        elif split=='valid':
            self.data = glob(os.path.join(os.path.dirname(root+'bffhq/'), split, "*"))

        elif split=='test': # bring only conflict samples from testset
            self.data = glob(os.path.join(os.path.dirname(root+'bffhq/'), split, "*"))
            data_conflict = []
            for path in self.data:
                target_label = path.split('/')[-1].split('.')[0].split('_')[1] # 0: young, 1: old
                bias_label = path.split('/')[-1].split('.')[0].split('_')[2] # 0: woman, 1: man
                if target_label != bias_label:
                    data_conflict.append(path)
            self.data = data_conflict

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        target_label, bias_label = torch.LongTensor([int(self.data[index].split('_')[1]), int(self.data[index].split('_')[2].split('.')[0])])
        image = Image.open(self.data[index]).convert('RGB')

        if self.transform is not None:
            image = self.transform(image)  
        
        # image, target_label, self.data[index], ... 순서 지킬 것
        return image, target_label, self.data[index], bias_label 