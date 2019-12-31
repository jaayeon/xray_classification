
import torch
import time
from tqdm import tqdm

def evaluate(opt, model, data_loader, loss_criterion):
    print('===> Validation')

    start_time = time.time()

    total_loss = 0.0
    total_correct = 0.0
    total_acc = 0.0

    with torch.no_grad():
        for i, batch in enumerate(tqdm(data_loader), 1):
            x, label = batch[0], batch[1]

            if opt.use_cuda:
                x = x.to(opt.device, dtype = torch.float)
                label = label.to(opt.device, dtype = torch.float)

            out = model(x)
            # label = label.to(torch.float)

            loss = loss_criterion(out, label)
            total_loss += loss.item()

            _, pred = torch.max(out.data,1)
            _, label = torch.max(label.data,1)
            
            total_correct += (pred == label).sum().item()
    
    total_loss = total_loss/i 
    total_acc = 100.*total_correct/len(data_loader.dataset)

    print("\n***Validation %.2fs => Epoch[%d/%d] :: Loss : %.20f, ACC : %d/%d (%3.4f%%)\n"%
            (time.time()-start_time, opt.epoch_num, opt.n_epochs, total_loss, total_correct, len(data_loader.dataset), total_acc))


    return (total_loss, total_acc)