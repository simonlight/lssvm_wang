import json
import cv2
import os
import numpy as np
import itertools
from matplotlib import pyplot as plt
from path_config import *
import Image
import xml.etree.cElementTree as ET

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

def visualize_fixations(fixation_path):

    for root,dirs,files in os.walk(fixation_path):
        for file in files:
            cls, year, id= file.split('_')
            cls = 'horse'
            year='2009'
            id='003768.ggg'
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
                
def read_fixations(fixation_file, cls, eye_path):
    f = open(eye_path+cls + '_' + fixation_file)
    fixations = []
    for line in f: 
        x,y = line.strip().split(',')
        fixations.append([int(x),int(y)])
    f.close()
    return fixations                

def slice_cnt(x,y,left, right, up, down):
    if x>=left and x<right and y>=up and y<down:
        return 1.0
    else:
        return 0.0

def IoU_gt_gaze_region(eye_path=VOC2012_OBJECT_EYE_PATH, annotations = VOC2012_TRAIN_ANNOTATIONS):
    object_names=["dog", "cat", "motorbike", "boat", "aeroplane", "horse" ,"cow", "sofa", "diningtable", "bicycle"]
    no_gaze=0
    for root,dirs,files in os.walk(eye_path):
        file_ratio = [0]*10
        file_num = [0]*10
        for cnt, filename in enumerate(files):
            fixation_file = '_'.join(([filename.split('_')[1], filename.split('_')[2]]))
            im = fixation_file[:-4]+'.jpg'
            cls = filename.split('_')[0]
            class_index = object_names.index(cls)
            fixations = read_fixations(fixation_file, cls, eye_path)
            xmltree = ET.ElementTree(file=VOC2012_TRAIN_ANNOTATIONS+fixation_file[:-4]+'.xml')            
            
            #bb of objects of given class
            objects=[]
            for elem in xmltree.iterfind('object'):
                for name in elem.iter('name'):
                    
                    if name.text == cls:                            
                        for coor in elem.iter('bndbox'):
                            xmax = [coorelem.text for coorelem in coor if coorelem.tag == 'xmax'][0]
                            xmin = [coorelem.text for coorelem in coor if coorelem.tag == 'xmin'][0]
                            ymax = [coorelem.text for coorelem in coor if coorelem.tag == 'ymax'][0]
                            ymin = [coorelem.text for coorelem in coor if coorelem.tag == 'ymin'][0]
                           
                            objects.append([int(float(xmax)),int(float(xmin)),int(float(ymax)),int(float(ymin))])
            ingaze = 0
            cnt=0
            for (point_x, point_y) in fixations:
                cnt+=1
                for obj in objects:
                    if slice_cnt(int(point_x),int(point_y), obj[1], obj[0], obj[3], obj[2]) ==1.0:
                        ingaze+=1.0
                        break
            if ingaze/cnt<=0.1:
                no_gaze+=1
                print filename
                print fixations
            file_ratio[class_index] += ingaze/cnt
            file_num[class_index] += 1
        print no_gaze
        print object_names
        print file_ratio
        print file_num

if __name__ == "__main__":
    IoU_gt_gaze_region()
                 