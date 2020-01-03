#run_train, run_valid, run_test by opt's mode

import torch
from model import VGG
from utils.model_loader import load_model
from utils.saver import save_checkpoint
from utils.trainer import train
from utils.evaluator import evaluate
from data.data_loader import get_data_loader
import os
import torch.nn as nn
from options import args
from tqdm import tqdm


def run_train(opt, training_data_loader, validation_data_loader):
    if not os.path.exists(opt.checkpoint_dir):
        os.makedirs(opt.checkpoint_dir)

    log_file = os.path.join(opt.checkpoint_dir, 'vgg_log.csv')

    print('[Initialize networks for training]')

    net = VGG(opt)
    L2_criterion = nn.MSELoss()
    print(net)

    if opt.resume :
        opt.start_epoch, net = load_model(opt, opt.checkpoint_dir)
    else : 
        with open(log_file, mode = 'w') as f:
            f.write('epoch, train_loss, valid_loss\n')

    print('===> Setting GPU')
    print('CUDA Available', torch.cuda.is_available())

    if opt.use_cuda and torch.cuda.is_available():
        opt.use_cuda = True
        opt.device = 'cuda'
    else : 
        opt.use_cuda = False
        opt.device = 'cpu'

    if torch.cuda.device_count() > 1 and opt.multi_gpu : 
        print("Use" + str(torch.cuda.device_count()) + 'GPUs')
        net = nn.DataParallel(net)

    if opt.use_cuda :
        net = net.to(opt.device)
        L2_criterion = L2_criterion.to(opt.device)

    print("===> Setting Optimizer")
    optimizer = torch.optim.Adam(net.parameters(), lr = opt.lr, betas = (opt.b1, opt.b2))


    for epoch in range(opt.start_epoch, opt.n_epochs) :
        opt.epoch_num = epoch
        train_loss, train_acc = train(opt, net, optimizer, training_data_loader, loss_criterion = L2_criterion)
        valid_loss, valid_acc = evaluate(opt, net, validation_data_loader, loss_criterion = L2_criterion)

        with open(log_file, mode = 'a') as f:
            f.write("%d, %08f,%08f,%08f,%08f\n"%(
                epoch,
                train_loss,
                train_acc,
                valid_loss,
                valid_acc
            ))
        save_checkpoint(opt, net, epoch, valid_loss)



def test_model(opt, test_data_loader):
    print('===> Test')

    opt.resume_best = True
    _, net = load_model(opt, opt.checkpoint_dir)
    criterion = nn.MSELoss()

    if torch.cuda.device_count() > 1 and opt.multi_gpu :
        print('Use' + str(torch.cuda.device_count()) + 'GPUs')
        net = nn.DataParallel(net)

    if opt.use_cuda and torch.cuda.is_available():
        opt.use_cuda = True
        opt.device = 'cuda'
    else :
        opt.use_cuda = False
        opt.device = 'cpu'

    if opt.use_cuda :
        net = net.to(opt.device)
        criterion = criterion.to(opt.device)


    total_correct = 0.0
    total_acc = 0.0

    with torch.no_grad():
        for i, batch in enumerate(tqdm(test_data_loader),1):
            x, label = batch[0], batch[1]
            
            if opt.use_cuda :
                x = x.to(opt.device, dtype = torch.float)
                label = label.to(opt.device, dtype = torch.float)
            out = net(x)

            # print('output>>>', out)
            # print('\nlabel>>>', label)

            
            _, pred = torch.max(out.data,1)
            _, label = torch.max(label.data,1)

            total_correct += (pred == label).sum().item()

    total_acc = 100.*total_correct/len(test_data_loader.dataset)

    print('\nTEST RESULT ===> ACC : %d/%d (%3.4f%%)\n'%(total_correct, len(test_data_loader.dataset), total_acc))


        


if __name__ == "__main__":

    opt = args
    print(opt)
    data_dir = opt.data_dir
    print('data_dir is : {}'.format(data_dir))

    training_data_loader, validation_data_loader, test_data_loader = get_data_loader(opt)

    if opt.mode is 'train' :
        run_train(opt, training_data_loader, validation_data_loader )
    else : 
        test_model(opt, test_data_loader)
