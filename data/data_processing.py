import os, glob
import torch.utils.data as data
from torchvision.transforms import Compose, ToTensor, Resize, Normalize
import numpy as np
from PIL import Image, ImageFilter
import cv2
import pandas as pd


def stack_img(opt, img_path_id):

    #Abs, FFTPhs, Scatt
    kinds = opt.img_kinds.split(',')

    #img_path_id = D:/data/xray-classification/e1/E1_Flour_0_90
    #[:-2] : D:/data/xray-classification/e1/E1_Flour
    img_path = ('_').join(img_path_id.split('_')[:-2])
    #[-2:] : 0_90
    img_id = ('_').join(img_path_id.split('_')[-2:])

    IMG_st = np.array([])

    for kind in kinds : 
        #img_path_kind = D:/data/xray-classification/e1/E1_Flour/Abs
        img_path_kind = os.path.join(img_path, kind)
        #img_id = 0_90 
        #img = AbsImg_0_90.IMG
        img = kind + 'Img_' + img_id + '.IMG'
        IMG = open(os.path.join(img_path_kind, img), 'r')
        # IMG_arr = np.fromfile(IMG, dtype = np.double).reshape(71,71)
        IMG_arr = np.fromfile(IMG, dtype = np.double)
        
        IMG_st = np.append(IMG_st, IMG_arr)
    
    #IMG_st : (C,W,H)
    IMG_st = IMG_st.reshape(len(kinds), 71,71)

    return IMG_st


def label2class(label):     # one hot encoding (0-2 --> [., ., .])

    resvec = [0, 0, 0]
    #[Flour, Salt, Sugar]
    if label == 0:		cls = 0;    resvec[cls] = 1
    elif label == 1:	cls = 1;    resvec[cls] = 1
    elif label == 2:	cls = 2;    resvec[cls] = 1
    else : print('label error... not one of 0,1,2')

    return resvec



def name2class(name):

    resvec = [0,0,0]
    if 'Flour' in name:     
        cls = 0
        resvec[cls] = 1
    elif 'Salt' in name :       
        cls = 0
        resvec[cls] = 1
    elif 'Sugar' in name : 
        cls = 0
        resvec[cls] = 1

    else : print('img_path_id not include Flour,Salt,Sugar')

    return resvec



def get_list(opt):

    energy = opt.energy.split(',')

    data_dir = opt.data_dir

    all_list = []

    for edir in energy : 
        fpath = os.path.join(data_dir, edir)
        files = os.listdir(fpath)
        for f in files : 
            if f.endswith('.xlsx') : 
                fn = '_'.join(f.split('_')[0:2])
                fn = os.path.join(fpath, fn)
                #fn = D:/data/xray-classification/e1/E1_Flour

                table = pd.read_excel(os.path.join(fpath,f)).values.tolist()
                for i in range(len(table)):
                    #table = D:/data/xray-classification/e1/E1_Flour_0_90
                    table[i][0] = fn + '_' + str(table[i][0]) 

                all_list.extend(table)

    return all_list




def input_transform(opt):
    compose = Compose([ToTensor(), Normalize((0.485,0.456,0.406), (0.229,0.224,0.225))])
    # compose = Compose([ToTensor()])
    return compose



class DatasetFromFolder(data.Dataset) :
    def __init__(self, opt):
        super(DatasetFromFolder, self).__init__()
        
        #img_list = [name, label]
        self.all_list = get_list(opt)
        #self.img_list = [os.path.join(data_dir, x) for x in os.listdir(data_dir) if is_image_file(x)]
        #self.img_label_list = np.loadtxt("%s.csv"%(train_label_dir), delimiter=',')

        self.input_transform = input_transform(opt)

    def __getitem__(self, idx):
        print('loading train dataset', self.all_list[idx])
        
        #get list id & mk to img path

        #ex) img_path_id = D:/data/xray-classification/e1/E1_Flour_0_90
        img_path_id = self.all_list[idx][0]
        #ex) img_label = 0 or 1 or 2
        img_label = self.all_list[idx][1]


        img = stack_img(opt, img_path_id)
        img_class = label2class(opt, img_label)

        img = self.input_transform(img)

        if img_class != name2class(img_path_id) : 
            raise ValueError('img_id is different from img_label')
        
        return img, img_class, img_path_id

    def __len__(self):
        return len(self.all_list)


# class testDataset(data.Dataset):
#     def __init__(self, opt):
#         super(testDataset, self).__init__()

#         if opt.mode =='test':
#             img_dir = opt.test_dir
#         if opt.mode == 'valid':
#             img_dir = opt.valid_dir

#         self.input_transform = input_transform(opt)
#         self.img_list = [os.path.join(img_dir, x) for x in os.listdir(img_dir) if is_image_file(x)]

#     def __getitem__(self, idx):
#         img = cv2.imread(self.img_list[idx])
#         img_name = os.path.basename(self.img_list[idx])

#         img = self.input_transform(img)

#         return img, img_name

#     def __len__(self):
#         return len(self.img_list)