from torch.utils.data import DataLoader, Dataset
import pandas as pd
from PIL import Image
from torchvision import transforms
from glob import glob
import os
import torch

class CmnistDataset_org(Dataset):
    def __init__(self, type, bias_ratio, transform=None, root='../../dataset/'):
        """
        type: 'train', 'valid', 'test' |
        bias_ratio: 1.0, 0.995, 0.99, 0.98, 0.95, 0.5, 0.0, 'unbiased'
        """
        # if type == 'test' and bias_ratio != 'unbiased':
        #     raise Exception("Test is unbiased.")
        if type not in ['train', 'valid', 'test']:
            raise Exception("Check type")
        if bias_ratio not in [1.0, 0.995, 0.99, 0.98, 0.95, 0.5, 0.0, 'unbiased']:
            raise Exception("Check bias_ratio")

        super().__init__()
        if type == 'test': 
            self.dataset_path = f'{root}/cmnist/test'
            self.labels_csv = pd.read_csv(f'{root}/cmnist/test.csv') #FIXME test 데이터 경로 지정
        else:
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
        img_path = f'{self.dataset_path}/{index}.jpg'
        a = 1
        return sample, label, img_path, a


###
class CmnistDataset(Dataset):
    def __init__(self, type, bias_ratio, transform=None, root='../../dataset/'): 
        super(CmnistDataset, self).__init__()
        self.bias_ratio = bias_ratio
        self.transform = transform
        self.root = root
        if self.transform  == None:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])

        if type=='train':
            self.align = glob(os.path.join(root, 'cmnist',str(self.bias_ratio), type,'align',"*"))  #FIXME
            self.conflict = glob(os.path.join(root, 'cmnist',str(self.bias_ratio), type,'conflict',"*"))
            self.data = self.align + self.conflict
        elif type=='valid':
            self.data = glob(os.path.join(root,'cmnist',str(self.bias_ratio),"valid","*"))            
        elif type=='test':
            self.data = glob(os.path.join(root,'cmnist',"test","*", "*"))  

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        target_label = int(self.data[index].split('_')[-2])
        image = Image.open(self.data[index]).convert('RGB')
        image = self.transform(image)
        
        # dummy
        a = 1
        # image, target_label, self.data[index], ... 순서 지킬 것
        return image, target_label, self.data[index], a

###    



class bFFHQDataset(Dataset):
    def __init__(self, split, transform, use_generated, root='../../dataset/'):
        super().__init__()
        self.transform = transform
        self.root = root

        if split=='train':
            self.align = glob(os.path.join(root+'/bffhq/0.5pct/','align',"*","*"))
            self.conflict = glob(os.path.join(root+'/bffhq/0.5pct/','conflict',"*","*"))
            if use_generated:
                self.generated_align = glob(os.path.join(root+'/generated_bffhq/0.5pct/','align',"*","*"))
                self.generated_conflict = glob(os.path.join(root+'/generated_bffhq/0.5pct/','conflict',"*","*"))
                self.data = self.align + self.conflict + self.generated_align + self.generated_conflict
            else:
                self.data = self.align + self.conflict

        elif split=='valid':
            self.data = glob(os.path.join(os.path.dirname(root+'/bffhq/'), split, "*"))

        elif split=='test': # bring only conflict samples from testset
            self.data = glob(os.path.join(os.path.dirname(root+'/bffhq/'), split, "*"))
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
    

class BARDataset(Dataset):
    def __init__(self, root, split, transform=None, percent=None, image_path_list=None):
        super(BARDataset, self).__init__()
        self.transform = transform
        self.percent = percent
        self.split = split
        self.image2pseudo = {}
        self.image_path_list = image_path_list

        self.train_align = glob(os.path.join(root,'train/align',"*/*"))
        self.train_conflict = glob(os.path.join(root,'train/conflict',f"{self.percent}/*/*"))
        self.valid = glob(os.path.join(root,'valid',"*/*"))
        self.test = glob(os.path.join(root,'test',"*/*"))

        if self.split=='train':
            self.data = self.train_align + self.train_conflict
        elif self.split=='valid':
            self.data = self.valid
        elif self.split=='test':
            self.data = self.test

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        attr = torch.LongTensor(
            [int(self.data[index].split('_')[-2]), int(self.data[index].split('_')[-1].split('.')[0])])
        image = Image.open(self.data[index]).convert('RGB')
        image_path = self.data[index]

        if 'bar/train/conflict' in image_path:
            attr[1] = (attr[0] + 1) % 6
        elif 'bar/train/align' in image_path:
            attr[1] = attr[0]

        if self.transform is not None:
            image = self.transform(image)  
        return image, attr, (image_path, index)