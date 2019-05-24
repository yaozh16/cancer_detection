#coding=utf-8

import os
import pandas

device="cuda:0"


def checkDirectory(path):
    prefix,suffix=os.path.split(path)
    if(not os.path.exists(path)  and not prefix==""):
        checkDirectory(prefix)
    if(not os.path.exists(path)):
        os.mkdir(path)