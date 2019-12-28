import os, glob
import torch.utils.data as data
from torchvision.transforms import Compose, ToTensor, Resize, Normalize
import numpy as np
from PIL import Image, ImageFilter
import cv2
import pandas as pd


def stack_img(opt, img_path, img_id):

    #Abs, FFTPhs, Scatt
    kinds = opt.img_kinds.split(',')

    for kind in kinds : 
        img_path = os.path.join(img_path, kind)
        img = kind + 'Img_' + img_id + '.IMG'
        # img_arr = np.fromfile(img, dtype = np.double).reshape(71,71)
        IMG = open(img, 'r')
        IMG_arr = np.fromfile(IMG, dtype = np.double).reshape(71,71)



    train_dir = opt.train_dir

    return



def get_list(opt):

    energy = opt.energy.split(',')

    train_dir = opt.train_dir

    all_list = []

    for edir in energy : 
        fpath = os.path.join(train_dir, edir)
        files = os.listdir(fpath)
        for f in files : 
            if f.endswith('.xlsx') : 
                fn = '_'.join(f.split('_')[0:2])
                fn = os.path.join(fpath, fn)

                table = pd.read_excel(os.path.join(fpath,f)).values.tolist()
                for i in range(len(table)):
                    table[i][0] = fn + '_' + str(table[i][0]) 

                all_list.extend(table)


    return all_list





def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.IMG'])

def input_transform(opt):
    compose = Compose([ToTensor(), Normalize((0.485,0.456,0.406), (0.229,0.224,0.225))])
    compose = Compose([ToTensor()])
    return compose


#train과 valid set 어떻게 구분하지,,,,어떻게 쌓지 으악 .. 매번 만들어? 종나 비효율적이긴 한거 같은데 매번 쌓아서 돌리면 학습 속도가 너무 느릴것 같아. 그냥 미리 만들어 놓고 하자. 
#train과 valid를 구분해서 저장해 놓는게 아니라 통째로 저장되어 있는 곳에서 randomsplit해서 load할때 다르게 하는건가봐!
class trainDataset(data.Dataset) :
    def __init__(self, opt):
        super(trainDataset, self).__init__()
        
        #img_list = [name, label]
        self.all_list = get_list(opt)
        #self.img_list = [os.path.join(train_dir, x) for x in os.listdir(train_dir) if is_image_file(x)]
        #self.img_label_list = np.loadtxt("%s.csv"%(train_label_dir), delimiter=',')

        self.input_trasform = input_transform(opt)

    def __getitem__(self, idx):
        print('loading train dataset', self.all_list[idx])
        #img = Image.open(self.img_list[idx])
        #img = cv2.imread(self.img_list[idx], cv2.GRAY_SCALE)
        img_path_id = self.all_list[idx][0]
        img_path = ('_').join(img_path_id.split('_')[:-2])
        img_id = ('_').join(img_path_id.split('_')[-2:])

        img = stack_img(opt, img_path, img_id)

        img_id = os.path.basename(self.img_list[idx])[:-4]
        img_label_id = img_label_list[idx,0]
        img_label = img_label_list[idx,1]

        if img_id != img_label_id : 
            raise ValueError('img_id is different from img_label_id')
        
        return img, img_label, img_id

    def __len__(self):
        return len(self.img_list)


class testDataset(data.Dataset):
    def __init__(self, opt):
        super(testDataset, self).__init__()

        if opt.mode =='test':
            img_dir = opt.test_dir
        if opt.mode == 'valid':
            img_dir = opt.valid_dir

        self.input_transform = input_transform(opt)
        self.img_list = [os.path.join(img_dir, x) for x in os.listdir(img_dir) if is_image_file(x)]

    def __getitem__(self, idx):
        img = cv2.imread(self.img_list[idx])
        img_name = os.path.basename(self.img_list[idx])

        img = self.input_transform(img)

        return img, img_name

    def __len__(self):
        return len(self.img_list)