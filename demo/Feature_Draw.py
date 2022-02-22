# -*- coding: utf-8 -*-
"""
Created on Wed Apr 21 21:19:26 2021

@author: Lucas
"""

import cv2
import numpy as np
import os

x=5

fl='./P4'
filelist=os.listdir(fl)
img=np.array([[1]*x for l in range (x)])

for file in filelist:
    a=file
    a=a.replace('file.txt','.jpg')
    imgpath=os.path.join(fl,a)
    print(a)
    filepath=os.path.join(fl,file)
    f=open(filepath,'r')
    j=0
    for line in f.readlines():
        a=line.split(' ')
        for i in range(x):
            img[j][i]=a[i]
        j+=1
    cv2.imwrite(imgpath,img)
   
    

