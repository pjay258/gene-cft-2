import torch
import wandb
from utils.seed import fix


def train_iter(X, y, model, loss_fn, optimizer, device):
    preds = model(X)
    loss = loss_fn(preds, y).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return preds, loss


def train(remote,
          model,
          epoch,
          device,
          dataloaders,
          optimizer,
          loss_fn,
          loss_type,
          seed):
    print("Epochs:", epoch)
    
    # Fix random seed
    fix(seed)

    # Train loops
    model.train()
    model = model.to(device)
    train_correct = 0
    train_correct = 0
    train_total = 0

    for batch_idx, (X, y, *_) in enumerate(dataloaders['train']):
        X, y = X.to(device), y.to(device)
        preds, loss = train_iter(X=X, y=y,
                                model=model,
                                loss_fn=loss_fn,
                                optimizer=optimizer,
                                device=device)

        _, pred_labels = torch.max(preds, dim=1)
        train_correct += (pred_labels == y).sum().item()
        train_total += y.size(0)

        if remote:
            if loss_type == 'GCE':
                wandb.log(
                    {
                    "Batch step": batch_idx+len(dataloaders['train'])*epoch,
                    "Train/Classification Loss(GCE)": loss.item(),
                    }
                )
            elif loss_type == 'CE':
                wandb.log(
                    {
                    "Batch step": batch_idx+len(dataloaders['train'])*epoch,
                    "Train/Classification Loss(CE)": loss.item(),
                    }
                )

    if remote:
        wandb.log(
            {
            "Epoch step": epoch,
            "Accuracy/Trainset": train_correct/train_total,
            }
        )

def valid(remote,
          model,
          epoch,
          device,
          dataloaders,
          loss_fn,
          loss_type):

    # Valid loops
    with torch.no_grad():
        model.eval()

        valid_correct = 0
        valid_total = 0
        valid_loss = 0
        for batch_idx, (X, y, *_) in enumerate(dataloaders['valid']):
            X, y = X.to(device), y.to(device)
            preds = model(X)
            loss = loss_fn(preds, y).mean()

            _, pred_labels = torch.max(preds, dim=1)
            valid_correct += (pred_labels == y).sum().item()
            valid_total += y.size(0)
            valid_loss += loss.item()

    if remote:
        wandb.log(
            {
            "Epoch step": epoch,
            "Accuracy/Validset": valid_correct/valid_total,
            }
        )
        if loss_type == 'GCE':
            wandb.log(
                {
                "Epoch step": epoch,
                "Valid/Validset Classification Loss(GCE)": valid_loss/valid_total,
                }
            )
        elif loss_type == 'CE':
            wandb.log(
                {
                "Epoch step": epoch,
                "Valid/Validset Classification Loss(CE)": valid_loss/valid_total,
                }
            )

def test(remote,
          model,
          epoch,
          device,
          dataloaders,
          loss_fn,
          loss_type):
    
    # Test loops
    with torch.no_grad():
        model.eval()

        test_correct = 0
        test_total = 0
        test_loss = 0
        for batch_idx, (X, y, *_) in enumerate(dataloaders['test']):
            X, y = X.to(device), y.to(device)
            preds = model(X)
            loss = loss_fn(preds, y).mean()

            _, pred_labels = torch.max(preds, dim=1)
            test_correct += (pred_labels == y).sum().item()
            test_total += y.size(0)
            test_loss += loss.item()

    if remote:
        wandb.log(
            {
            "Epoch step": epoch,
            "Accuracy/Testset": test_correct/test_total,
            }
        )
        if loss_type == 'GCE':
            wandb.log(
                {
                "Epoch step": epoch,
                "Valid/Testset Classification Loss(GCE)": test_loss/test_total,
                }
            )
        elif loss_type == 'CE':
            wandb.log(
                {
                "Epoch step": epoch,
                "Valid/Testset Classification Loss(CE)": test_loss/test_total,
                }
            )