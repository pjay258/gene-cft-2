from torch.utils.data import DataLoader
from torchvision import transforms as T
from . import datasets

transforms_preprcs = {
    "cmnist": {
        "train": T.Compose([T.ToTensor()]),
        "valid": T.Compose([T.ToTensor()]),
        "test": T.Compose([T.ToTensor()])
        },
    
    "bar": {
        "train": T.Compose([
            T.Resize((224, 224)),
            T.RandomCrop(224, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]),
        "valid": T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]),
        "test": T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
    },

    "bffhq": {
        "train": T.Compose([
            T.Resize((224,224)),
            T.RandomCrop(224, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]),

        "valid": T.Compose([
            T.Resize((224,224)),
            T.ToTensor(),
            T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]),

        "test": T.Compose([
            T.Resize((224,224)),
            T.ToTensor(),
            T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
    },

    "dogs_and_cats": {
        "train": T.Compose([
            T.Resize((224, 224)),
            T.RandomCrop(224, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]),

        "valid": T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]),

        "test": T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]),
    },
}


def return_dataset(dataset, split, root, transform, use_generated):
    if dataset == 'bffhq':
        return datasets.bFFHQDataset(root=root,
                                     split=split,
                                     transform=transform,
                                     use_generated=use_generated)


def get_dataloader(dataset,
                   split,
                   root,
                   batch_size,
                   shuffle=False, 
                   num_workers=4,
                   use_generated=True):
    
    transform = transforms_preprcs[dataset][split] # e.g. dataset := 'bffhq', split := 'train'
    
    target_dataset = return_dataset(dataset=dataset,
                                    split=split,
                                    root=root,
                                    transform=transform,
                                    use_generated=use_generated)
    
    target_dataloader = DataLoader(dataset=target_dataset,
                                   batch_size=batch_size,
                                   shuffle=shuffle,
                                   num_workers=num_workers)

    return target_dataloader


def return_dataloaders(dataset,
                        root,
                        batch_size,
                        shuffle=True, 
                        num_workers=4,
                        use_generated=False):
    dataloaders = {
        'train': get_dataloader(dataset=dataset,
                                split='train',
                                root=root,
                                batch_size=batch_size,
                                shuffle=shuffle,
                                num_workers=num_workers,
                                use_generated=use_generated),
        'valid': get_dataloader(dataset=dataset,
                                split='valid',
                                root=root,
                                batch_size=batch_size,
                                shuffle=False,
                                num_workers=num_workers,
                                use_generated=False),
        'test': get_dataloader(dataset=dataset,
                                split='test',
                                root=root,
                                batch_size=batch_size,
                                shuffle=False,
                                num_workers=num_workers,
                                use_generated=False)
    }

    return dataloaders