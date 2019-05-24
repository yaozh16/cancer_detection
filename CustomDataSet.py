#coding=utf-8

from PIL import Image
import torch
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms
import pandas as pd
import os
import cv2
import numpy as np
from preprocessor import cropValidBox

class MyDataset(Dataset):  # 创建自己的类：MyDataset,这个类是继承的torch.utils.data.Dataset
    def __init__(self, datacsv, rootpath,device="cpu" , transform=None, target_transform=None):  # 初始化一些需要传入的参数
        super(MyDataset, self).__init__()


        # id,img,age,HER2,P53, molecular_subtype(label)
        self.rootpath=rootpath
        data=pd.read_csv(os.path.join(rootpath,datacsv))
        self.datavalues=data.values
        self.datacolumns=data.columns.values
        self.transform = transform
        self.target_transform = target_transform

        self.labelcandidates=np.array([[0]*5 for i in range(5)])
        for i in range(5):
            self.labelcandidates[i,i]=1
        self.dtype = torch.float
        self.device = torch.device

    def __getitem__(self, index):
        # 这个方法是必须要有的，用于按照索引读取每个元素的具体内容
        img_path = os.path.join(self.rootpath,"images",self.datavalues[index][0],self.datavalues[index][1])
        diagnos_data=torch.from_numpy(np.array(self.datavalues[index][2:5],dtype=np.float32))
        label=np.array(self.labelcandidates[self.datavalues[index][-1]],dtype=np.float32)
        label=torch.from_numpy(label)

        img_data=Image.fromarray(cv2.imread(img_path))# 按照img_path读入图片并预处理
        if self.transform is not None:
            img_data = self.transform(img_data)  # 是否进行transform
        return img_data,diagnos_data, label    # return很关键，return回哪些内容，那么我们在训练时循环读取每个batch时，就能获得哪些内容

    def __len__(self):  # 这个函数也必须要写，它返回的是数据集的长度，也就是多少张图片，要和loader的长度作区分
        return self.datavalues.shape[0]



if __name__=="__main__":
    # 根据自己定义的那个MyDataset来创建数据集！注意是数据集！而不是loader迭代器
    train_data = MyDataset(datacsv='train.csv',rootpath=os.path.join("formated","train"), transform=transforms.ToTensor())
    valid_data = MyDataset(datacsv='valid.csv',rootpath=os.path.join("formated","train"), transform=transforms.ToTensor())

    train_loader = DataLoader(dataset=train_data, batch_size=64, shuffle=True)
    valid_loader = DataLoader(dataset=valid_data, batch_size=64)


    for img_data,diagnos_data, target in train_loader:
        print("bi:{0}/{1}".format(img_data.shape,diagnos_data.shape))
        print("ba:{0}".format(target.shape))