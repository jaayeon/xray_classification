import argparse
import os
import torch


data_dir = "../../data/dataset-x-ray"

checkpoint_dir = os.path.join(data_dir, 'checkpoint')

train_label_dir = os.path.join(data_dir, 'train_label.csv')
valid_label_dir = os.path.join(data_dir, 'valid_label.csv')
test_label_dir = os.path.join(data_dir, 'test_label.csv')

parser = argparse.ArgumentParser(description='X-RAY classification')

parser.add_argument('--mode', type=str, default='train',
                    help='train, valid, test')
parser.add_argument('--use_cuda', type=bool, default=True)
parser.add_argument('--device', type = str, default = 'cpu')
parser.add_argument('--resume_best', action='store_true',
                    help='Resume the last model of epoch from checkpoint')
parser.add_argument('--resume', type = bool, default = False)

parser.add_argument('--seed', type = int, default = 1)
parser.add_argument('--multi_gpu', type = bool, default = True)

parser.add_argument('--train_ratio', type=float, default = 0.7)
parser.add_argument('--valid_ratio', type=float, default = 0.2)
parser.add_argument('--test_ratio', type=float, default = 0.1)

parser.add_argument('--data_dir', type=str, default = data_dir)
parser.add_argument('--checkpoint_dir', type=str, default = checkpoint_dir)

parser.add_argument('--img_kinds', type=str, default='Abs,Scatt',
                    help='Abs, FFTPhs, Scatt' )
parser.add_argument('--energy', type=str, default='e1,e2',
                    help='e1,e2,e3,e4')

parser.add_argument('--model', type=str, default = 'vgg')
parser.add_argument('--n_epochs', type=int, default = 200)
parser.add_argument('--epoch_num', type = int, default = 0)
parser.add_argument('--start_epoch', type = int, default = 1)
parser.add_argument('--lr', type=float, default = 1e-4)
parser.add_argument('--batch_size', type=int, default=600)
parser.add_argument('--b1', type = float, default = 0.9,
                        help = 'Adam : decay of first order momentum of gradient')
parser.add_argument('--b2', type = float, default = 0.999, 
                        help = 'Adm : decay of second order momentum of gradient')

#입력받은 인자값을 args에 저장
args = parser.parse_args()

torch.manual_seed(args.seed)
