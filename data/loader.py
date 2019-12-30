import os
import torch.utils import data
import torch.utils.data import DatatLoader
from data_processing import trainDataset, testDataset 


def get_train_valid_data_loader(opt):

    x_ray_data = trainDataset(opt)
    train_len = int(opt.train_ratio * len(x_ray_data))
    valid_len = len(x_ray_data) - train_len

    train_dataset, valid_dataset = data.random_split(x_ray_data, lengths = [train_len, valid_len] )
    train_data_loader = DatatLoader(dataset=train_dataset,
                                    batch_size = opt.batch_size,
                                    shuffle = True, num_workders=2)
    #num_workers : multi-threading

    valid_data_loader = DatatLoader(dataset=valid_dataset,
                                    batch_size = opt.batch_size,
                                    shuffle = False, num_workers = 2)

    return train_data_loader, valid_data_loader


def get_test_data_loader(opt):

    x_ray_data = testDataset(opt)

    test_data_loader = DataLoader(datasest=x_ray_data, 
                                batch_size = opt.batch_size,
                                shuffle = False, num_workers = 2)
    return test_data_loader

