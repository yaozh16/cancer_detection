#coding=utf-8

import os
import Common
import numpy as np
import cv2
import random
import glob
from matplotlib import pyplot as plt
import pandas as pd
from tqdm import tqdm

def cropValidBox(img_r, display=False):
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
def format2size(src_dir="train",dst_dir="formated"):
    all_paths=sorted(glob.glob(os.path.join(src_dir,"images","*","*.jpg")))
    for i,img_path in enumerate(tqdm(all_paths)):
        imr_r=cv2.imread(img_path)
        img=cropValidBox(imr_r, False)
        img=cv2.resize(img,(224,224))
        dstpath=os.path.join(dst_dir,img_path)
        Common.checkDirectory(os.path.split(dstpath)[0])
        cv2.imwrite(os.path.join(dst_dir,img_path),img)

def splitSet(p=0.7, src_dir="train", dst_dir=os.path.join("formated", "train")):
    random.seed(1079)
    Common.checkDirectory(dst_dir)
    patients=pd.read_csv(os.path.join(src_dir,"feats.csv"))
    all_img_paths=sorted(glob.glob(os.path.join(src_dir,"images","*","*.jpg")))
    trainset=random.sample(all_img_paths,int(p*len(all_img_paths)))
    validset=[e for e in all_img_paths if not e in trainset]

    def splited_set2csv(path,splited_set):
        with open(path,"w") as f:
            f.write("id,img,age,HER2,P53,molecular_subtype\n")
            for img_path in tqdm(splited_set):
                img_idx=os.path.split(img_path)[-1]
                id=os.path.split(os.path.split(img_path)[0])[-1]
                p=patients[patients["id"]==id]
                if(p.values.shape[0]==0):
                    continue
                fstr="{0},{1},{2},{3},{4},{5}".format(p.id.values[0],img_idx,p.age.values[0]/80.0,p.HER2.values[0]/3.0,int(p.P53.values[0]),int(p.values[0][-1]))
                f.write(fstr)
                f.write("\n")
            f.close()
    splited_set2csv(os.path.join(dst_dir,"train.csv"),trainset)
    splited_set2csv(os.path.join(dst_dir,"valid.csv"),validset)

if __name__=="__main__":
    print("* split train\dev set\n")
    splitSet(p=0.8, src_dir=os.path.join("..","cancer_detection_dataset","train"), dst_dir=os.path.join("formated", "train"))
    print("* split test set\n")
    splitSet(p=0, src_dir=os.path.join("..","cancer_detection_dataset","test"), dst_dir=os.path.join("formated", "test"))
    print("* format2size train\n")
    format2size(src_dir=os.path.join("..","cancer_detection_dataset","train"),dst_dir="formated")
    print("* format2size test\n")
    format2size(src_dir=os.path.join("..","cancer_detection_dataset","test"),dst_dir="formated")
