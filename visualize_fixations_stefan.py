from matplotlib import pyplot as plt
from path_config import *
import json
import cv2
import os
import numpy as np
import xml.etree.cElementTree as ET

import itertools
###CONFIG###
fixation_path = "/local/wangxin/Data/gaze_voc_actions_stefan/train_gazes/"
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
    
def visualize_fixations(fixation_path):
    for root,dirs,files in os.walk(fixation_path):
        for file in files:
            file = "2012_002313.json"
            filename_root = file[:-5]#.json
            fixation = json.load(open(gaze_path+file)).values()
            img = VOC2012_TRAIN_IMAGES+filename_root+'.jpg'
            img = cv2.imread(img,0)
            rows = img.shape[0]
            cols = img.shape[1]
            out = np.zeros((rows,cols,3), dtype='uint8')
            out= np.dstack([img, img, img])
            colors = ['b','g','r','c','m','y','k']
            for cnt, subj in enumerate(fixation):
                for fix in subj:
                    cv2.circle(out, (fix[0],fix[1]), 1, color_map(colors[cnt]), 1)
            
            for sy,sx in itertools.product(range(scale),repeat=2):
                start = (int(sx*0.1*cols), int(sy*0.1*rows))        
                end =(int(sx*0.1*cols)+int((11-scale) * cols/10),int(sy*0.1*rows)+int((11-scale) * rows/10))
                cv2.rectangle(out, start,end,(0,0,255)) 
                cv2.imshow('Matched Features', out)
                cv2.waitKey(0)
                cv2.destroyAllWindows()


def slice_cnt(x,y,left, right, up, down):
    if x>=left and x<right and y>=up and y<down:
        return 1.0
    else:
        return 0.0

def IoU_gt_gaze_region(fixation_path):
    action_names=["jumping", "phoning", "playinginstrument", "reading", "ridingbike", "ridinghorse", "running", "takingphoto", "usingcomputer", "walking"]
    for root,dirs,files in os.walk(fixation_path):
        file_ratio = [0]*10
        file_num = [0]*10
        for file in files:
            class_index = -1
            filename_root = file[:-5]#.json
            fixations = json.load(open(gaze_path+file)).values()
            
            xmltree = ET.ElementTree(file=VOC2012_TRAIN_ANNOTATIONS+filename_root+'.xml')            
            for elem in xmltree.iterfind('object'):
                if len(list(elem.iter('name')))>1:
                    print "error of object"
                else:
                    for actions in elem.iter('actions'):
                        class_index = action_names.index([action.tag for action in actions if action.text is '1'][0])
                    for coor in elem.iter('bndbox'):
                        xmax = [coorelem.text for coorelem in coor if coorelem.tag == 'xmax'][0]
                        xmin = [coorelem.text for coorelem in coor if coorelem.tag == 'xmin'][0]
                        ymax = [coorelem.text for coorelem in coor if coorelem.tag == 'ymax'][0]
                        ymin = [coorelem.text for coorelem in coor if coorelem.tag == 'ymin'][0]
            ingaze = 0
            cnt=0
            for ob in fixations:
                for (point_x, point_y) in ob:
                    cnt+=1
                    ingaze+=slice_cnt(int(point_x),int(point_y), int(float(xmin)), int(float(xmax)), int(float(ymin)), int(float(ymax)))
            file_ratio[class_index] += ingaze/cnt
            # any bb receives no fixations?
            # None!
            if ingaze/cnt<=0.1:
                print filename_root, action_names[class_index]
            file_num[class_index] += 1
        print action_names
        print file_ratio
        print file_num
visualize_fixations(fixation_path)
# IoU_gt_gaze_region(fixation_path)