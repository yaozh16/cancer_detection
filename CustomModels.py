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
    def __init__(self, Image_in, Diagnos_in, D_out, train_option, net_type):
        super(CombineNet, self).__init__()
        self.train_option=train_option
        self.Image_in=Image_in
        self.Diagnos_in=Diagnos_in
        self.D_out=D_out
        self.H1=60
        self.H2=20
        print("train_option:{0}".format(train_option),flush=True)
        print("img net_type:{0}".format(net_type),flush=True)
        if(net_type=="vgg11"):
            self.imagenet=models.vgg11(num_classes=self.H1)
        elif(net_type=="vgg13"):
            self.imagenet=models.vgg13(num_classes=self.H1)
        elif(net_type=="resnet50"):
            self.imagenet = models.resnet50(pretrained=True)
            self.imagenet.fc=nn.Linear(2048,self.H1)
        elif(net_type=="resnet34"):
            self.imagenet = models.resnet34(pretrained=True)
            self.imagenet.fc=nn.Linear(512,self.H1)
        elif(net_type=="resnet101"):
            self.imagenet = models.resnet101(pretrained=False)
            self.imagenet.fc=nn.Linear(2048,self.H1)
        elif(net_type=="resnet152"):
            self.imagenet = models.resnet152(pretrained=True)
            self.imagenet.fc=nn.Linear(2048,self.H1)
        else:
            assert False

        self.diagnosnet=nn.Sequential(
            nn.Linear(self.Diagnos_in, 100),
            nn.ReLU(inplace=True) ,
            nn.Linear(100, self.H1),
        )
        self.fc=nn.Sequential(
            nn.Linear(self.H1*2,self.H2),
            nn.ReLU(inplace=True),
            nn.Linear(self.H2,self.D_out),
            nn.Sigmoid()
        )

        if (self.train_option == "IMG_ONLY"):
            for k, v in self.diagnosnet.named_parameters():
                v.requires_grad = False
            self.combine_func=lambda src_img,src_dia:torch.cat((src_img.view(src_img.size(0), -1),
                              src_img.view(src_img.size(0), -1)),
                             dim=1)
        elif (self.train_option == "DIA_ONLY"):
            for k, v in self.imagenet.named_parameters():
                v.requires_grad = False
            self.combine_func=lambda src_img,src_dia:torch.cat((src_dia.view(src_dia.size(0), -1),
                              src_dia.view(src_dia.size(0), -1)),
                             dim=1)
        else:
            for k, v in self.diagnosnet.named_parameters():
                v.requires_grad = False
            for k, v in self.imagenet.named_parameters():
                v.requires_grad = False
            self.combine_func=lambda src_img,src_dia:torch.cat((src_img.view(src_img.size(0), -1),
                              src_dia.view(src_dia.size(0), -1)),
                             dim=1)
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
        if(path_prefix_img!=None):
            self.imagenet.load_state_dict(torch.load("{0}_{1}.mdl".format(path_prefix_img,"image")))
        if(path_prefix_diagnos!=None):
            self.diagnosnet.load_state_dict(torch.load("{0}_{1}.mdl".format(path_prefix_diagnos,"diagnos")))
        if(path_prefix_fc!=None):
            self.fc.load_state_dict(torch.load("{0}_{1}.mdl".format(path_prefix_fc,"fc")))

def accuracy(y_pred,target):
    _, predicted = torch.max(y_pred.data, 1)
    _, actual = torch.max(target.data, 1)
    return int((predicted == actual).sum())*1.0/target.size(0)

def train(model,train_option, net_type):

    Common.checkDirectory("model")
    # 根据自己定义的那个MyDataset来创建数据集！注意是数据集！而不是loader迭代器
    train_data = MyDataset(datacsv='train.csv',rootpath=os.path.join("formated","train"), transform=transforms.ToTensor())
    valid_data = MyDataset(datacsv='valid.csv',rootpath=os.path.join("formated","train"), transform=transforms.ToTensor())

    train_loader = DataLoader(dataset=train_data, batch_size=58, shuffle=True)
    valid_loader = DataLoader(dataset=valid_data, batch_size=60)


    best_acc=0

    criterion = torch.nn.MSELoss(reduction='sum')

    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
                                lr=1e-4,
                                momentum = 0.2,
                               )
    def valid_round():
        accurate_count = 0
        total_count = 0
        for img_data, diagnos_data, target in valid_loader:
            y_pred = model(img_data, diagnos_data)
            accurate_count += accuracy(y_pred, target) * target.size(0)
            total_count += target.size(0)
            del img_data, diagnos_data, target
        acc = accurate_count * 1.0 / total_count
        print("[accuracy]{0}".format(accurate_count * 1.0 / total_count), flush=True)
        return acc
    valid_round()
    for epo in range(30):
        print("epoch {0}".format(epo),flush=True)
        batch_index=0
        for img_data,diagnos_data, target in train_loader:
            try:
                # Forward pass: Compute predicted y by passing x to the model
                y_pred = model(img_data ,diagnos_data)

                # Compute and print loss
                loss = criterion(y_pred, target)
                print("\t[e {0}:b {1}]{2}[{3}]".format(epo ,batch_index, loss.item(),accuracy(y_pred,target)),flush=True)

                # Zero gradients, perform a backward pass, and update the weights.
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            except Exception as e:
                print("[ERROR!]{0}:ignored".format(e),flush=True)
            del img_data,diagnos_data,target
            batch_index+=1
        acc=valid_round()
        if(acc>best_acc):
            best_acc=acc
            model.save_to(os.path.join("model","{0}_{1}_{2}_{3}".format(net_type, train_option, epo, acc)))
        elif(epo%3==0):
            model.save_to(os.path.join("model","{0}_{1}_{2}_{3}".format(net_type, train_option, epo, acc)))

def test(epo,acc,title1,title2,title3,type1,type2,type3):
    test_data = MyDataset(datacsv='valid.csv',rootpath=os.path.join("formated","test"), transform=transforms.ToTensor())
    valid_loader = DataLoader(dataset=test_data, batch_size=50, shuffle=False)
    model = CombineNet(3, 3, 5, "IMG_ONLY","BOTH")
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
    train_option="IMG_ONLY"
    net_type="resnet50"

    model = CombineNet(3, 3, 5, train_option=train_option, net_type=net_type)
    #img_mdl_path = None
    img_mdl_path = os.path.join("model", "{0}_{1}_{2}_{3}".format(net_type, "IMG_ONLY", 12, "0.6055045871559633"))
    #dia_mdl_path = os.path.join("model", "{0}_{1}_{2}_{3}".format(net_type, "IMG_ONLY", 12, "0.6055045871559633"))
    dia_mdl_path =None
    #fc_mdl_path = None
    fc_mdl_path = os.path.join("model", "{0}_{1}_{2}_{3}".format(net_type, "IMG_ONLY", 12, "0.6055045871559633"))
    model.load_from(img_mdl_path, dia_mdl_path, fc_mdl_path)
    train(model,train_option,net_type)