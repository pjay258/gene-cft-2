import torch
import torch.nn as nn
import pandas as pd
from glob import glob
import argparse
import os

from utils.networks import return_resnet18
from utils.networks import NaiveMLP
from utils.dataloader import return_dataloaders


def main():
    parser = argparse.ArgumentParser(description='Training BCD')
    parser.add_argument('-cuda', required=True, type=int, help="gpu")
    parser.add_argument('-device', required=True, type=str, help="device")
    parser.add_argument('-batch_size', required=True, type=int, help="batch size")
    parser.add_argument('-dataset', required=True, type=str, help="dataset")
    parser.add_argument('-num_BCD', required=True, type=int, help="num BCD")
    parser.add_argument('-bias_prob_threshold', required=True, type=float, help="bias prob threshold")
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda)

    # Load BCD models and comprise PBA
    PBA = []
    for BCD_idx in range(args.num_BCD):
        if args.dataset == 'bffhq':
            BCD = return_resnet18(num_classes=2)
            BCD.load_state_dict(torch.load(f'./BCD_models/{args.dataset}/BCD{BCD_idx}.pth'))
        elif args.dataset == 'cmnist':
            BCD = NaiveMLP()
            BCD.load_state_dict(torch.load(f'./BCD_models/{args.dataset}/BCD{BCD_idx}.pth'))

        BCD.eval()
        BCD = BCD.to(args.device)
        PBA.append(BCD)

    # Detect bias aligned/conflict samples of trainset
    softmax = nn.Softmax(dim=1)
    
    dataloaders = return_dataloaders(dataset=args.dataset,
                                     root='../../dataset/',
                                     batch_size=args.batch_size,
                                     shuffle=False)

    # 각 배치마다 bias detect 해서 csv로 저장
    with torch.no_grad():
        for batch_idx, (X, y, path, *_) in enumerate(dataloaders['train']):
            X, y = X.to(args.device), y.to(args.device)
            
            gt_probs_list = []
            for BCD_idx in range(args.num_BCD):
                logits = PBA[BCD_idx](X)
                probs = softmax(logits)
                gt_probs = probs.gather(1, y.unsqueeze(1)).squeeze(1)
                gt_probs_list.append(gt_probs)

            gt_probs_stack = torch.stack(gt_probs_list, dim=0)

            # p_y가 bias_prob_threshold를 초과하도록 예측한 BCD가 절반 이상이면 biased(1)
            # 그렇지 않으면 conflict(0)
            bias_flag = (gt_probs_stack>args.bias_prob_threshold).sum(dim=0)>(args.num_BCD/2)
            int_bias_flag = bias_flag.long()
            list_bias_flag = int_bias_flag.tolist() # e.g. [1, 1, 1, 0, 1, ...] 0이면 conflict 

            batch_bias_table = {
                'file_path': path,
                'bias': list_bias_flag
            }
            df = pd.DataFrame(batch_bias_table)
            df.to_csv(f'./bias_tables/{args.dataset}/bias_table_{batch_idx}.csv', index=False)

    # 모든 batch csv를 하나의 csv로 합치기
    bias_tables = glob(f'./bias_tables/{args.dataset}/bias_table_*.csv')
    dfs = [pd.read_csv(f) for f in bias_tables]
    df = pd.concat(dfs, ignore_index=True)
    df.to_csv(f'./bias_tables/{args.dataset}/bias_table.csv', index=False)


if __name__ == '__main__':
    main()