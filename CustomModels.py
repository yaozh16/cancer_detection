# -*- coding: utf-8 -*-


import math
import torch
from torch import nn
import os
from torch.utils.data import DataLoader
from torchvision import transforms,models
from CustomDataSet import MyDataset
import Common
import numpy as np


class CombineNet(torch.nn.Module):
    def __init__(self, Image_in,Diagnos_in, D_out,custom_option,net_type=None):
        super(CombineNet, self).__init__()
        self.custom_option=custom_option
        self.Image_in=Image_in
        self.Diagnos_in=Diagnos_in
        self.D_out=D_out
        self.H1=60
        self.H2=20
        if(net_type==models.vgg11 or
            net_type==models.vgg13 or
            net_type==models.vgg19):
            self.imagenet=net_type(num_classes=self.H1)
        elif(net_type==models.resnet50):
            self.imagenet = net_type(pretrained=True)
            self.imagenet.fc=nn.Linear(2048,self.H1)
        elif(net_type==models.resnet34):
            self.imagenet = net_type(pretrained=True)
            self.imagenet.fc=nn.Linear(512,self.H1)
        else:
            assert False
        self.imagenet.fc=nn.Linear(2048,self.H1)
        self.diagnosnet=nn.Sequential(
            nn.Linear(self.Diagnos_in, 100),
            nn.ReLU(inplace=True) ,
            nn.Linear(100, self.H1),
        )
        self.fc=nn.Sequential(
            nn.Linear(self.H1*2,self.H2),
            nn.ReLU(inplace=True),
            nn.Linear(self.H2,self.D_out),
        )
        def combine_func(src_img,src_dia):
            if(self.custom_option=="IMG_ONLY"):
                return torch.cat((src_img.view(src_img.size(0), -1),
                                  src_img.view(src_img.size(0), -1)),
                                 dim=1)
            elif(self.custom_option=="DIA_ONLY"):
                return torch.cat((src_dia.view(src_dia.size(0), -1),
                                  src_dia.view(src_dia.size(0), -1)),
                                 dim=1)
            else:
                return torch.cat((src_img.view(src_img.size(0), -1),
                                  src_dia.view(src_dia.size(0), -1)),
                                 dim=1)
        self.combine_func=combine_func
    def forward(self, image_data,diagnos_data):
        h1=self.imagenet(image_data)
        h2=self.diagnosnet(diagnos_data)
        combined = self.combine_func(h1,h2)
        y_pred = self.fc(combined)
        return y_pred
    def save_to(self,path_prefix):
        torch.save(self.imagenet.state_dict(), "{0}_{1}.mdl".format(path_prefix,"image"))
        torch.save(self.diagnosnet.state_dict(), "{0}_{1}.mdl".format(path_prefix,"diagnos"))
        torch.save(self.fc.state_dict(), "{0}_{1}.mdl".format(path_prefix,"fc"))
    def load_from(self,path_prefix_img,path_prefix_diagnos,path_prefix_fc):
        self.imagenet.load_state_dict(torch.load("{0}_{1}.mdl".format(path_prefix_img,"image")))
        self.diagnosnet.load_state_dict(torch.load("{0}_{1}.mdl".format(path_prefix_diagnos,"diagnos")))
        self.fc.load_state_dict(torch.load("{0}_{1}.mdl".format(path_prefix_fc,"fc")))

def accuracy(y_pred,target):
    _, predicted = torch.max(y_pred.data, 1)
    _, actual = torch.max(target.data, 1)
    return int((predicted == actual).sum())*1.0/target.size(0)
    #print(int(target.size(0)))
    #return (predicted == actual).sum()

def train(type,title,net_type):
    model=CombineNet(3,3,5,type)
    save_path=os.path.join("model","{0}_{1}_{2}_{3}".format(title,type,2,"0.48623853211009177"))
    #model.load_from()
    Common.checkDirectory("model")
    # 根据自己定义的那个MyDataset来创建数据集！注意是数据集！而不是loader迭代器
    train_data = MyDataset(datacsv='train.csv',rootpath=os.path.join("formated","train"), transform=transforms.ToTensor())
    valid_data = MyDataset(datacsv='valid.csv',rootpath=os.path.join("formated","train"), transform=transforms.ToTensor())

    train_loader = DataLoader(dataset=train_data, batch_size=64, shuffle=True)
    valid_loader = DataLoader(dataset=valid_data, batch_size=10)


    best_acc=0

    criterion = torch.nn.MSELoss(reduction='sum')

    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4,momentum = 0.4)
    #optimizer = torch.optim.adam(model.parameters(), lr=1e-4 )
    for epo in range(120):
        print("epoch {0}".format(epo),flush=True)
        batch_index=0
        for img_data,diagnos_data, target in train_loader:

            # Forward pass: Compute predicted y by passing x to the model
            y_pred = model(img_data ,diagnos_data)

            # Compute and print loss
            loss = criterion(y_pred, target)
            print("\t[e {0}:b {1}]{2}[{3}]".format(epo ,batch_index, loss.item(),accuracy(y_pred,target)),flush=True)

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
            model.save_to(os.path.join("model","{0}_{1}_{2}_{3}".format(title,type,epo,acc)))

def test(epo,acc,title1,title2,title3,type1,type2,type3):
    test_data = MyDataset(datacsv='valid.csv',rootpath=os.path.join("formated","test"), transform=transforms.ToTensor())
    valid_loader = DataLoader(dataset=test_data, batch_size=1, shuffle=False)
    model = CombineNet(3, 3, 5, "IMG_ONLY")
    para_path1=os.path.join("model","{0}_{1}_{2}_{3}".format(title1,type1,epo,acc))
    para_path2=os.path.join("model","{0}_{1}_{2}_{3}".format(title2,type2,epo,acc))
    para_path3=os.path.join("model","{0}_{1}_{2}_{3}".format(title3,type3,epo,acc))
    model.load_from(para_path1,para_path2,para_path3)
    batch_index=0
    for img_data,diagnos_data, target in valid_loader:

            # Forward pass: Compute predicted y by passing x to the model
            y_pred = model(img_data ,diagnos_data)
            _, predicted = torch.max(y_pred.data, 1)
            id=test_data.datavalues[batch_index][0]
            print("{0}:{1}".format(id,predicted))
            batch_index+=1

if __name__ == "__main__":
    train("IMG_ONLY","resnet34",models.resnet34)