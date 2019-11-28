import argparse
import os
import torch


data_dir = "../../data/dataset-x-ray"
train_dir = os.path.join(data_dir, 'train')
valid_dir = os.path.join(data_dir, 'valid')
test_dir = os.path.join(data_dir, 'test')
checkpoint_dir = os.path.join(data_dir, 'checkpoint')

train_label_dir = os.path.join(data_dir, 'train_label.csv')
valid_label_dir = os.path.join(data_dir, 'valid_label.csv')
test_label_dir = os.path.join(data_dir, 'test_label.csv')

parser = argparse.ArgumentParser(description='X-RAY classification')

parser.add_argument('--mode', type=str, default='train',
                    help='train, valid, test')
parser.add_argument('--use_cude', type=str, default='cpu')

parser.add_argument('--n_threads', type=int, default = 4)
parser.add_argument('--seed', type = int, default = 1)

parser.add_argument('--train_ratio', type=float, default = 0.9)

parser.add_argument('--train_dir', type=str, default = train_dir)
parser.add_argument('--valid_dir', type=str, default = valid_dir)
parser.add_argument('--test_dir', type=str, default = test_dir)
parser.add_argument('--train_label_dir', type=str, default = train_label_dir)
parser.add_argument('--valid_label_dir', type=str, default = valid_label_dir)
parser.add_argument('--test_label_dir', type=str, default = test_label_dir)

parser.add_argument('--img_kinds', type=str, default='abs,scatt',
                    help='abs,scatt,fftphs' )
parser.add_argument('--energy', type='str', default='e1,e2',
                    help='e1,e2,e3,e4')

parser.add_argument('--model', type=str, default = 'cnn')
parser.add_argument('--checkpoint_dir', type=str, default = checkpoint_dir)
parser.add_argument('--epoch_num', type=int, default = 0)
parser.add_argument('--resume_best', action='store_true',
                    help='Resume the last model of epoch from checkpoint')
parser.add_argument('--lr', type=float, default = 1e-4)
parser.add_argument('--batch_size', type=int, default=32)

args = parser.parse_args()

torch.manual_seed(args.seed)
