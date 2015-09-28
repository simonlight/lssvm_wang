import json
import cv2
import os
import numpy as np
import itertools
from matplotlib import pyplot as plt

fixation_path = "/local/wangxin/Data/gaze_voc_actions_stefan/train_gazes"
pascal_voc_2012_train_images = "/local/wangxin/Data/VOCdevkit_trainset/VOC2012/JPEGImages/"
gaze_path="/local/wangxin/Data/gaze_voc_actions_stefan/train_gazes/"

for root,dirs,files in os.walk(fixation_path):
    for file in files:
        file = "2010_005042.xxxx"
        filename_root = file[:-5]#.json
        fixation = json.load(open(gaze_path+file)).values()
        img = pascal_voc_2012_train_images+filename_root+'.jpg'
        print img
        img = cv2.imread(img,0)
        rows = img.shape[0]
        cols = img.shape[1]
        out = np.zeros((rows,cols,3), dtype='uint8')
        out= np.dstack([img, img, img])
        for subj in fixation:
            for fix in subj:
                cv2.circle(out, (fix[0],fix[1]), 4, (255, 0, 0), 1)
        scale = 6
        for sy,sx in itertools.product(range(scale),repeat=2):
            print sx,sy
            start = (int(sx*0.1*cols), int(sy*0.1*rows))
            
            end =(int(sx*0.1*cols)+int((11-scale) * cols/10),int(sy*0.1*rows)+int((11-scale)   * rows/10))
            print start,end
            print rows,cols
            cv2.rectangle(out, start,end,(0,0,255)) 
            cv2.imshow('Matched Features', out)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
                 