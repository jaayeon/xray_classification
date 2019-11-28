import os, glob
import torch.utils.data as data
from torchvision.transforms import Compose, ToTensor, Resize, Normalize
import numpy as np
from PIL import Image, ImageFilter
import cv2

def stack_img(opt, idx, img_list):

    file_path = img_list[idx]

    img_arr = np.fromfile(file_path, dtype = np.double).reshape(71,71)
    
    kinds = opt.img_kinds.split(',')
    energy = opt.energy.split(',')

    train_dir = opt.train_dir




    return

def get_label_list(opt):
    energy = opt.energy.split(',')
    train_dir = opt.train_dir
    #csv file찾아서 원하는 e만 stack 해서 return

    return

def get_img_list(opt):


    return 


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.jpg', '.png'])

def input_transform(opt):
    #compose = Compose([ToTensor(), Normalize((0.485,0.456,0.406), (0.229,0.224,0.225))])
    compose = Compose([ToTensor()])
    return compose


#train과 valid set 어떻게 구분하지,,,,어떻게 쌓지 으악 .. 매번 만들어? 종나 비효율적이긴 한거 같은데 매번 쌓아서 돌리면 학습 속도가 너무 느릴것 같아. 그냥 미리 만들어 놓고 하자. 
#train과 valid를 구분해서 저장해 놓는게 아니라 통째로 저장되어 있는 곳에서 randomsplit해서 load할때 다르게 하는건가봐!
class trainDataset(data.Dataset) :
    def __init__(self, opt):
        super(trainDataset, self).__init__()

        train_dir = opt.train_dir
        train_label_dir = opt.train_label_dir

        self.img_list = get_img_list(opt)
        self.img_label_list = get_label_list(opt)
        #self.img_list = [os.path.join(train_dir, x) for x in os.listdir(train_dir) if is_image_file(x)]
        #self.img_label_list = np.loadtxt("%s.csv"%(train_label_dir), delimiter=',')

        self.input_trasform = input_transform(opt)

    def __getitem__(self, idx):
        print('loading', self.img_list[idx])
        #img = Image.open(self.img_list[idx])
        #img = cv2.imread(self.img_list[idx], cv2.GRAY_SCALE)
        img = stack_img(opt, idx, self.img_list)

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