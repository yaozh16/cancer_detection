# -*- coding: utf-8 -*-


import math
import torch
from torch import nn
import os
from torch.utils.data import DataLoader
from torchvision import transforms,models
from CustomDataSet import MyDataset
import Common

class CustomVGG(nn.Module):
    def __init__(self, feature_cfg, D_out,batch_norm=False, init_weights=True):
        super(CustomVGG, self).__init__()
        self.features = self.make_layers(feature_cfg,batch_norm)
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, D_out),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def make_layers(self,cfg, batch_norm=False):
        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)

class ImageNet(torch.nn.Module):
    def __init__(self, Image_in, D_out):
        super(ImageNet, self).__init__()
        self.Image_in=Image_in
        self.D_out=D_out

        #self.net=models.vgg11(num_classes=D_out)
        self.net=models.vgg19(num_classes=D_out)

    def forward(self, image_data):
        y_pred= self.net(image_data)
        return y_pred

class DiagnosNet(torch.nn.Module):
    def __init__(self, Diagnos_in, D_out):
        super(DiagnosNet, self).__init__()
        self.Diagnos_in=Diagnos_in
        self.D_out=D_out
        self.H1=100
        self.fc = nn.Sequential(
            nn.Linear(self.Diagnos_in, self.H1),
            nn.ReLU(inplace=True) ,
            nn.Linear(self.H1, self.D_out),
        )

    def forward(self,diagnos_data):
        return self.fc(diagnos_data)

class CombineNet(torch.nn.Module):
    def __init__(self, Image_in,Diagnos_in, D_out):
        super(CombineNet, self).__init__()
        self.Image_in=Image_in
        self.Diagnos_in=Diagnos_in
        self.D_out=D_out

        self.H1_1=500
        self.H1_2=10


        self.H2=100
        self.H3=20
        self.imagenet=ImageNet(Image_in,self.H1_1)
        self.diagnosnet=DiagnosNet(Diagnos_in,self.H1_2)
        self.fc=nn.Sequential(
            nn.Linear(self.H1_1+self.H1_2,self.H2),
            nn.ReLU(inplace=True),
            nn.Linear(self.H2,self.H3),
            nn.ReLU(inplace=True) ,
            nn.Linear(self.H3, self.D_out),
        )

    def forward(self, image_data,diagnos_data):
        h1=self.imagenet(image_data)
        h2=self.diagnosnet(diagnos_data)
        combined = torch.cat((h1.view(h1.size(0), -1),
                              h2.view(h2.size(0), -1)), dim=1)
        y_pred = self.fc(combined)
        return y_pred


def accuracy(y_pred,target):
    _, predicted = torch.max(y_pred.data, 1)
    _, actual = torch.max(target.data, 1)
    return int((predicted == actual).sum())*1.0/target.size(0)
    #print(int(target.size(0)))
    #return (predicted == actual).sum()



def train():
    Common.checkDirectory("model")
    # 根据自己定义的那个MyDataset来创建数据集！注意是数据集！而不是loader迭代器
    train_data = MyDataset(datacsv='train.csv',rootpath=os.path.join("formated","train"), transform=transforms.ToTensor())
    valid_data = MyDataset(datacsv='valid.csv',rootpath=os.path.join("formated","train"), transform=transforms.ToTensor())

    train_loader = DataLoader(dataset=train_data, batch_size=64, shuffle=True)
    valid_loader = DataLoader(dataset=valid_data, batch_size=10)

    model=CombineNet(3,3,5)

    best_acc=0

    criterion = torch.nn.MSELoss(reduction='sum')
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
    for t in range(120):
        print("epoch {0}".format(t),flush=True)
        batch_index=0
        for img_data,diagnos_data, target in train_loader:

            # Forward pass: Compute predicted y by passing x to the model
            y_pred = model(img_data ,diagnos_data)

            # Compute and print loss
            loss = criterion(y_pred, target)
            print("\t[e {0}:b {1}]{2}[{3}]".format(t ,batch_index, loss.item(),accuracy(y_pred,target)),flush=True)

            # Zero gradients, perform a backward pass, and update the weights.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_index+=1
        accurate_count=0
        total_count=0
        for img_data,diagnos_data,target in valid_loader:
            y_pred = model(img_data ,diagnos_data)
            accurate_count+=accuracy(y_pred,target)*target.size(0)
            total_count+=target.size(0)
        acc=accurate_count*1.0/total_count
        print("[accuracy]{0}".format(accurate_count*1.0/total_count),flush=True)
        if(acc>best_acc):
            best_acc=acc
            torch.save(model.state_dict(), os.path.join("model","{0}_{1}.mdl".format(t,acc)))

def test():
    test_data = MyDataset(datacsv='train.csv',rootpath=os.path.join("formated","test"), transform=transforms.ToTensor())


if __name__ == "__main__":
    train()