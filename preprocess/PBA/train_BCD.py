import torch
import torch.optim as optim

import os
import wandb
import argparse

from utils.seed import fix
from utils.dataloader import return_dataloaders
from utils.utils import GeneralizedCELoss
from utils.loops import train, valid, test
from utils.networks import return_resnet18
from utils.networks import NaiveMLP

def main():
    parser = argparse.ArgumentParser(description='Training BCD')
    parser.add_argument('-seed', required=True, type=int, help="random seed")
    parser.add_argument('-cuda', required=True, type=int, help="gpu")
    parser.add_argument('-device', required=True, type=str, help="device")
    parser.add_argument('-epochs', required=True, type=int, help="epochs")
    parser.add_argument('-lr', required=True, type=float, help="learning rate")
    parser.add_argument('-batch_size', required=True, type=int, help="batch size")
    parser.add_argument('-dataset', required=True, type=str, help="dataset")
    args = parser.parse_args()

    ###############################################
    # GPU Setting
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda)

    # Random seed
    seed = args.seed

    # WandB
    remote = True
    project_name = 'Deboost - mar 2'
    run_name = f'BCD {seed} | seed: {seed}'

    # Training details
    epochs = args.epochs
    learning_rate = args.lr
    batch_size = args.batch_size
    device = args.device

    # Save models
    save = True
    save_path = './BCD_models/'
    ###############################################  

    # Fix random seed
    fix(seed)

    if args.dataset == 'bffhq':
        # Random initialized Resnet-18 for bffhq(2 classes)
        model = return_resnet18(num_classes=2)
    elif args.dataset == 'cmnist':
        # Random initialized NaiveMLP for cmnist(10 classes)
        model = NaiveMLP()
    elif args.dataset == 'BAR':
        model = NaiveMLP()
    elif args.dataset == 'CATNDOG':
        model = NaiveMLP()

    # Training utils
    dataloaders = return_dataloaders(dataset=args.dataset,
                                    root='../../dataset/', 
                                    batch_size=batch_size,)
    loss_fn = GeneralizedCELoss()
    optimizer = optim.Adam(params = model.parameters(), lr=learning_rate)

    # WandB settings
    if remote:
        wandb.init(
            project=project_name,
            name=run_name,
            config={
                "random seed": seed,
                "learning_rate": learning_rate,
                "batch_size": batch_size, 
                "epochs": epochs,
                "note": ''
            }
        )
        wandb.define_metric("Train/*", step_metric="Batch step")
        wandb.define_metric("Valid/*", step_metric="Epoch step")
        wandb.define_metric("Accuracy/*", step_metric="Epoch step")

    for epoch in range(epochs):
        train(remote=True,
              model=model,
              epoch=epoch,
              device=device,
              dataloaders=dataloaders,
              optimizer=optimizer,
              loss_fn=loss_fn,
              loss_type='GCE',
              seed = seed)

        valid(remote=True,
              model=model,
              epoch=epoch,
              device=device,
              dataloaders=dataloaders,
              loss_fn=loss_fn,
              loss_type='GCE')

        test(remote=True,
             model=model,
             epoch=epoch,
             device=device,
             dataloaders=dataloaders,
             loss_fn=loss_fn,
             loss_type='GCE')
    
    if save: torch.save(model.state_dict(), f=f'{save_path}{args.dataset}/BCD{seed}.pth')

if __name__ == '__main__':
    main()