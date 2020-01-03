import torch
import torchvision
import torch.nn as nn

class VGG(nn.Module):

    def __init__(self, opt):
        super(VGG, self).__init__()
        self.c = len(opt.img_kinds.split(','))
        self.conv = nn.Sequential(
            #c 71 71
            nn.Conv2d(self.c,64,3,padding=1),nn.LeakyReLU(0.2),
            nn.Conv2d(64,64,3,padding=1),nn.LeakyReLU(0.2),
            nn.MaxPool2d(2,2),
            #64 35 35
            nn.Conv2d(64,128,3,padding=1), nn.LeakyReLU(0.2),
            nn.Conv2d(128,128,3, padding=1), nn.LeakyReLU(0.2),
            nn.MaxPool2d(2,2),
            #128 17 17
            nn.Conv2d(128,256,3,padding=1), nn.LeakyReLU(0.2),
            nn.Conv2d(256,256,3,padding=1), nn.LeakyReLU(0.2),
            nn.Conv2d(256,256,3,padding=1), nn.LeakyReLU(0.2),
            nn.MaxPool2d(2,2),
            #256 8 8
            nn.Conv2d(256,512,3,padding=1), nn.LeakyReLU(0.2),
            nn.Conv2d(512,512,3,padding=1), nn.LeakyReLU(0.2),
            nn.Conv2d(512,512,3,padding=1), nn.LeakyReLU(0.2),
            nn.MaxPool2d(2,2),
            #512 4 4
            nn.Conv2d(512,512,3,padding=1), nn.LeakyReLU(0.2),
            nn.Conv2d(512,512,3,padding=1), nn.LeakyReLU(0.2),
            nn.Conv2d(512,512,3,padding=1), nn.LeakyReLU(0.2),
            nn.MaxPool2d(2,2),
        )
        #512 2 2

        self.avg_pool = nn.AvgPool2d(2)
        #512 1 1
        self.classifier = nn.Linear(512*2*2, 3)


    def forward(self,x):
        features = self.conv(x)
        # features = self.avg_pool(features)
        # x = x.view(-1,features.shape[0])
        x = features.view(-1,512*2*2)
        x = self.classifier(x)
        # x = self.softmax(x)
        return x

    