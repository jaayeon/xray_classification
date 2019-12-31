import torch
import time
from tqdm import tqdm

def train(opt, model, optimizer, data_loader, loss_criterion):
    print('===> Training')
    start_time = time.time()

    total_loss = 0.0
    total_acc = 0.0
    total_correct = 0.0

    for i, batch in enumerate(tqdm(data_loader),1):
    # for i, batch in enumerate(data_loader,1):
        x, label = batch[0], batch[1]
        # print("\n\n&&&&&&&&&&&&&&&&&&&&&&&&&&CHECKING THE SIZE OF IMG#####################\n\n")
        # print(x.shape)
        # print("\n\n&&&&&&&&&&&&&&&&&&&&&&&&&&CHECKING THE SIZE OF IMG#####################\n\n")
        if opt.use_cuda :
            x = x.to(opt.device, dtype = torch.float)
            label = label.to(opt.device, dtype = torch.float)
            optimizer.zero_grad()

        out = model(x)
        # print(label.type())
        # label = label.to(torch.float)
        # print(label.type())
        loss = loss_criterion(out, label)
        loss.backward()
        optimizer.step()

        total_loss +=loss.item()
        
        # pred = out.max(1, keepdim = True)[1]
        _, pred = torch.max(out.data,1)
        _, label = torch.max(label.data,1)
        # total_correct += pred.eq(out.view_as(pred)).sum().item()
        # print(out)
        # print(pred)
        # print(label)
        total_correct += (pred == label).sum().item()

    total_loss = total_loss/i
    total_acc = 100.*total_correct/len(data_loader.dataset)

    print("***\nTraining %.2fs => Epoch[%d/%d] :: Loss : %.20f, ACC : %d/%d (%3f%%)\n"%(time.time()-start_time, opt.epoch_num, opt.n_epochs, total_loss, total_correct, len(data_loader.dataset), total_acc))
    # print("\n***Training %.2fs => Epoch[%d/%d] :: Loss : %.5f, ACC : %d/%d (%3.4f%)\n"%(time.time()-start_time, opt.epoch_num, opt.n_epochs, total_loss, total_correct, len(data_loader.dataset), total_acc))


    return(total_loss, total_acc)