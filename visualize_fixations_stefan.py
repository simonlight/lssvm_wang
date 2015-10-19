import json
import cv2
import os
import numpy as np
import itertools
from matplotlib import pyplot as plt
from spyderlib.widgets.internalshell import SysOutput
###CONFIG###
fixation_path = "/local/wangxin/Data/gaze_voc_actions_stefan/train_gazes/"
pascal_voc_2012_train_images = "/local/wangxin/Data/VOCdevkit_trainset/VOC2012/JPEGImages/"
gaze_path="/local/wangxin/Data/gaze_voc_actions_stefan/train_gazes/"
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
        file = "2011_005162.json"
        filename_root = file[:-5]#.json
        fixation = json.load(open(gaze_path+file)).values()
        img = pascal_voc_2012_train_images+filename_root+'.jpg'
        img = cv2.imread(img,0)
        rows = img.shape[0]
        cols = img.shape[1]
        out = np.zeros((rows,cols,3), dtype='uint8')
        out= np.dstack([img, img, img])
        colors = ['b','g','r','c','m','y','k']
        for cnt, subj in enumerate(fixation):
            for fix in subj:
                cv2.circle(out, (fix[0],fix[1]), 4, color_map(colors[cnt]), 1)
        
        for sy,sx in itertools.product(range(scale),repeat=2):
            start = (int(sx*0.1*cols), int(sy*0.1*rows))        
            end =(int(sx*0.1*cols)+int((11-scale) * cols/10),int(sy*0.1*rows)+int((11-scale) * rows/10))
            cv2.rectangle(out, start,end,(0,0,255)) 
            cv2.imshow('Matched Features', out)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
                 