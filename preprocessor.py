#coding=utf-8

import os
import Common
import numpy as np
import cv2
import random
import glob
from matplotlib import pyplot as plt
import pandas as pd

def cropValidBox(img_path, display=False):
    img_r = cv2.imread(img_path)
    img = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)
    h, w = img.shape
    padding = 10
    timg = np.zeros((h + padding * 2, w + padding * 2), dtype=img.dtype)
    timg[padding:-padding, padding:-padding] = img
    img = timg
    img = img
    h, w = img.shape


    non_zero_row = np.sum(img == 0, axis=1)
    non_zero_col = np.sum(img == 0, axis=0)
    assert len(non_zero_row) == h
    assert len(non_zero_col) == w
    diffH_nz = np.array(non_zero_col[:-1] - non_zero_col[1:])
    diffV_nz = np.array(non_zero_row[:-1] - non_zero_row[1:])

    r1 = 1.0 / 2
    r2 = 1.0 / 2

    def plotRankBySum(arr_nz, n, i):
        plt.subplot(n, 1, i + 1)
        plt.plot(range(len(arr_nz)), arr_nz, label="arr_nz({0})".format(i))
        plt.legend()

    def getEdgeIndex(arr, reverse):
        cand = []
        thresh = 150
        max = np.max(np.abs(arr))
        while (len(cand) < 3 and (thresh >= max / 4 or len(cand) < 1)):
            thresh -= 10
            if (reverse):
                d = [e for e in arr.argsort() if arr[e] > thresh]
            else:
                d = [e for e in arr.argsort() if arr[e] < -thresh]
            cand = sorted(d, reverse=reverse)
        return cand[0]

    hs = (
        getEdgeIndex(diffH_nz[:int(r1 * (w - 1))], reverse=True),
        getEdgeIndex(diffH_nz[int(r2 * (w - 1)):], reverse=False) + int(r2 * (w - 1))
    )
    vs = (
        getEdgeIndex(diffV_nz[:int(r1 * (h - 1))], reverse=True),
        getEdgeIndex(diffV_nz[int(r2 * (h - 1)):], reverse=False) + int(r2 * (h - 1))
    )


    if display:
        plt.figure(figsize=(10, 10))
        plotRankBySum(diffH_nz, 2, 0)
        plotRankBySum(diffV_nz, 2, 1)

        imgc = img.copy()

        idx = vs[0]
        cv2.line(imgc, (0, idx), (w, idx), (255))
        idx = vs[-1]
        cv2.line(imgc, (0, idx), (w, idx), (255))
        idx = hs[0]
        cv2.line(imgc, (idx, 0), (idx, h), (255))
        idx = hs[-1]
        cv2.line(imgc, (idx, 0), (idx, h), (255))
        # cv2.line(img,(l,0),(l,h),(255))
        cv2.imshow("({1},{2}:{3},{4}):{0}".format(idx, vs[0], vs[-1], hs[0], hs[-1]), imgc)
        ret = cv2.waitKey(0)
        if ret == 115:
            plt.show("test")
        plt.close('all')
        cv2.destroyAllWindows()

    return img_r[max(0,vs[0]-padding*2):vs[-1]-padding*2,max(0,hs[0]-padding*2):hs[-1]-padding*2,:]
def format2size():
    all_paths=sorted(glob.glob(os.path.join("train","images","*","*.jpg")))
    for i,img_path in enumerate(all_paths):
        img=cropValidBox(img_path, False)
        img=cv2.resize(img,(224,224))
        dstpath=os.path.join("formated",img_path)
        Common.checkDirectory(os.path.split(dstpath)[0])
        cv2.imwrite(os.path.join("formated",img_path),img)

def splitTrainSet(p=0.7):
    random.seed(1)
    patients=pd.read_csv(os.path.join("train","feats.csv"))
    all_img_paths=sorted(glob.glob(os.path.join("train","images","*","*.jpg")))
    trainset=random.sample(all_img_paths,int(p*len(all_img_paths)))
    validset=[e for e in all_img_paths if not e in trainset]

    def splited_set2csv(path,splited_set):
        with open(path,"w") as f:
            f.write("id,img,age,HER2,P53,molecular_subtype\n")
            #print("id,img,age,HER2,P53,molecular_subtype")
            for img_path in splited_set:
                img_idx=os.path.split(img_path)[-1]
                id=os.path.split(os.path.split(img_path)[0])[-1]
                p=patients[patients["id"]==id]
                fstr="{0},{1},{2},{3},{4},{5}".format(p.id.values[0],img_idx,p.age.values[0]*3,p.HER2.values[0]*64,int(p.P53.values[0])*255,p.molecular_subtype.values[0])
                #print(fstr)
                f.write(fstr)
                f.write("\n")
            f.close()
    splited_set2csv(os.path.join("formated","train","train.csv"),trainset)
    splited_set2csv(os.path.join("formated","train","valid.csv"),validset)

if __name__=="__main__":
    splitTrainSet()
    format2size()