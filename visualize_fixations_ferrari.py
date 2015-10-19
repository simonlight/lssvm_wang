import json
import cv2
import os
import numpy as np
import itertools
from matplotlib import pyplot as plt
###CONFIG###
fixation_path = "/local/wangxin/Data/ferrari_gaze/gazes/"
pascal_voc_2012_train_images = "/local/wangxin/Data/VOCdevkit_trainset/VOC2012/JPEGImages/"
scale = 6
############


def color_map(color):
    """Change color name to RGB list. Note that the value is not correct"""
    if color == 'b':
        return [0,0,0]
    elif color == 'g':
        return [255,0,0]
    elif color == 'r':
        return [0,255,0]
    elif color == 'c':
        return [0,0,255]
    elif color == 'm':
        return [255,255,0]
    elif color == 'y':
        return [255,0,255]
    elif color == 'k':
        return [0,255,255]
    
for root,dirs,files in os.walk(fixation_path):
    for file in files:
        cls, year, id= file.split('_')
        cls = 'horse'
        year='2010'
        id='001856.ggg'
        id=id[:-4]
#         file = "2012_003108.json"
#         filename_root = file[:-5]#.json
        fixation = []
        f = open(root+cls+'_'+year+'_'+id+'.txt')
        for line in f:
            x,y = line.strip().split(',')
            fixation.append([int(x),int(y)])
        f.close()

        img = pascal_voc_2012_train_images+year+'_'+id+'.jpg'
        print img
        img = cv2.imread(img,0)
        rows = img.shape[0]
        cols = img.shape[1]
        out = np.zeros((rows,cols,3), dtype='uint8')
        out= np.dstack([img, img, img])
        for x,y in fixation:
            cv2.circle(out, (x,y),3,[255,255,0],1)
        
        for sy,sx in itertools.product(range(scale),repeat=2):
            start = (int(sx*0.1*cols), int(sy*0.1*rows))        
            end =(int(sx*0.1*cols)+int((11-scale) * cols/10),int(sy*0.1*rows)+int((11-scale) * rows/10))
            cv2.rectangle(out, start,end,(0,0,255)) 
            cv2.imshow('Matched Features', out)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
                 