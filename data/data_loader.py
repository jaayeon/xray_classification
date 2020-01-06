import os
from torch.utils import data
from torch.utils.data import DataLoader
from data.data_processing import DatasetFromFolder

def get_data_loader(opt):


    dataset = DatasetFromFolder(opt)

    train_len = int(opt.train_ratio * len(dataset))
    valid_len = int(opt.valid_ratio * len(dataset))
    test_len = int(len(dataset) - train_len - valid_len) 

    train_dataset, valid_dataset, test_dataset = data.random_split(dataset,lengths = [train_len, valid_len, test_len] )
    train_data_loader = DataLoader(dataset=train_dataset,
                                    batch_size = opt.batch_size,
                                    shuffle = True)
    #num_workers : multi-threading

    print(test_dataset)

    valid_data_loader = DataLoader(dataset=valid_dataset,
                                    batch_size = opt.batch_size,
                                    shuffle = False)

    test_data_loader = DataLoader(dataset=test_dataset,
                                    batch_size = opt.batch_size,
                                    shuffle = False)                            

    return train_data_loader, valid_data_loader, test_data_loader








