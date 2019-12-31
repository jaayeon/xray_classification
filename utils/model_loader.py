import torch
from utils.saver import save_checkpoint
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../model')))
from model import VGG
import glob

def load_model(opt, checkpoint_dir):
    checkpoint_list = glob.glob(os.path.join(checkpoint_dir, "*.pth"))
    checkpoint_list.sort()

    if opt.resume_best :
        loss_list = list(map(lambda x : float(os.path.basename(x).split('_')[4][:-4]), checkpoint_list))
        best_loss_idx = loss_list.index(min(loss_list))
        checkpoint_pth = checkpoint_list[best_loss_idx]
    else :
        checkpoint_pth = checkpoint_list[len(checkpoint_list)-1]

    net = VGG(opt)

    if os.path.isfile(checkpoint_pth):
        print("=> loading checkpoint '{}'".format(checkpoint_pth))
        checkpoint = torch.load(checkpoint_pth)

        n_epoch = checkpoint['epoch']
        net.load_state_dict(checkpoint['net'].state_dict())
        print("=> loaded checkpoint '{}'(epoch {})".format(checkpoint_pth, n_epoch))
    else :
        print("=> no checkpoint found at {}".format(checkpoint_pth))
        n_epoch = 0

    return n_epoch+1, net