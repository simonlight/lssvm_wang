root = "/local/wangxin/Data/gaze_voc_actions_stefan/"

eye_tracking_path = root+"samples/"
gaze_path = root+"train_gazes/"

action_subjects  = ["006","007","008","009","010","011","018","020"] 

pascal_voc_2012_train_images = "/local/wangxin/Data/VOCdevkit_trainset/VOC2012/JPEGImages/"


train_list = root+"action_train_image_list"

import collections
import os
from PIL import Image

def valide_subjects(train_list, eye_tracking_path):
    f = open(train_list)
    train_images = [line.strip() for line in f]
    subj_fixationcnt = collections.defaultdict(lambda:0)
    for root, dirs, files in os.walk(eye_tracking_path):
        for file in files:
            subj, year, id = file.split('_')
            filename = '_'.join([year,id])
            if subj in action_subjects and filename[:-4] in train_images:
                sub_fixationcnt[subj]+=1
    print subj_fixationcnt   


import json
import math
import numpy as np
from xml.dom import minidom
valide_subjs = ["006","007","008","009","010","011","018"]

def mapping(image_res_x, image_res_y, x_screen, y_screen, screen_res_x=1280.0, screen_res_y=1024.0):
    sx=image_res_x/screen_res_x;
    sy=image_res_y/screen_res_y;
    s=max(sx,sy);
    dx=max(image_res_x*(1/sx-1/sy)/2,0);
    dy=max(image_res_y*(1/sy-1/sx)/2,0);
    x_stimulus=s*(x_screen-dx);
    y_stimulus=s*(y_screen-dy);
    return x_stimulus, y_stimulus

def valide_fixations(train_list, eye_tracking_path, valide_subjs):
    tl = open(train_list)
    train_images = [line.strip() for line in tl]
    tl.close()
    
    
    for root, dirs, files in os.walk(eye_tracking_path):
        for file in files:
            subj, year, id = file.split('_')
            if not subj in valide_subjs:
                continue
            filename = '_'.join([year,id])[:-4]
            if filename in train_images:
                json_path = gaze_path+filename[:-4]+'.json'
                if os.path.exists(json_path):
                    fixations_file=open(json_path,'r')
                    old_fixations = json.load(fixations_file)
                    fixations_file.close()
                else: 
                    old_fixations = collections.defaultdict(lambda:[])
                
                image_path = (pascal_voc_2012_train_images + file[4:])[:-4] 
                image_res_x, image_res_y= Image.open(image_path).size
                
                
                et = open(eye_tracking_path+file,'r')
                et.readline()
                new_fixations = collections.defaultdict(lambda:[])
                
                for line in et:
                    time, pupil_diameter, pupil_area, x_screen, y_screen, event = line.strip().split('\t')
                    if event == 'F':
                        x_stimulus, y_stimulus = mapping(image_res_x, image_res_y, int(float(x_screen)), int(float(y_screen)))
                        new_fixations[str(subj).strip()].append([int(x_stimulus),int(y_stimulus)])
                et.close()
#                 print fixations
                new_fixations.update(old_fixations) 
                
                gaze_json = open(json_path,'w')
                json.dump(new_fixations,gaze_json)
                gaze_json.close()
                
pascal_voc_2012_annotations = "/local/wangxin/Data/VOCdevkit_trainset/VOC2012/Annotations/"
action_names=["jumping", "phoning", "playinginstrument", "reading", "ridingbike", "ridinghorse", "running", "takingphoto", "usingcomputer", "walking"]
#scales = [1,4,9,16,25,36,49,64]
scales = [100]
slice = 10.0
def slice_cnt(x,y,left, right, up, down):
    if x>=left and x<right and y>=up and y<down:
        return 1.0
    else:
        return 0.0

def calculate_gaze_ratio(train_list, gaze_path):
    tl = open(train_list)
    train_images = [line.strip() for line in tl]
    #print train_images #['2010_006088.jpg', '2010_006089.jpg']
    tl.close()
    c=0
    for im in train_images:
        c+=1
        print c
        fixation_file = open(gaze_path+im[:-4]+'.json')
        fixations = json.load(fixation_file)
        fixation_file.close()
        total_fixations = sum([len(observers) for observers in fixations.values()])
        
        xmldoc = minidom.parse(pascal_voc_2012_annotations+im.strip()[:-4]+'.xml')
        itemlist = xmldoc.getElementsByTagName("actions")
        for action in enumerate(action_names):
            if int(itemlist[0].getElementsByTagName(action[1])[0].childNodes[0].nodeValue) ==1:
                action_category = action[1]
                continue
       
        image_res_x, image_res_y= Image.open(pascal_voc_2012_train_images+im).size
        integrate_image = np.zeros((10,10))
        for x_inc in range(0,10):
            for y_inc in range(0,10):
                left = (x_inc) * image_res_x/10.0
                right = (x_inc+1) * image_res_x/10.0
                up = (y_inc) * image_res_y/10.0
                down = (y_inc+1) * image_res_y/10.0
                for ob in fixations.values():
                    for (point_x, point_y) in ob:
                         integrate_image[x_inc][y_inc]+=slice_cnt(point_x, point_y, left, right, up, down)
        

        for scale in scales:
            block_num = int(math.sqrt(scale))
            check=0
            for i_x in range(block_num):
                for i_y in range(block_num):
                    ratio = np.sum(integrate_image[i_x:11-block_num+i_x, i_y:11-block_num+i_y])/total_fixations
                    folder = root+"ETLoss_ratio/"+action_category+'/'+str(scale)+'/'
                    #if not os.path.exists(folder):
                        #os.makedirs(folder)
                    #if scale == 1:
                        #etloss_filename = folder + im[:-4]+'.txt'
                    #else:
                        #etloss_filename = folder + im[:-4]+'_'+str(i_x)+'_'+str(i_y)+'.txt'
                    #loss_file = open(etloss_filename,'w')
                    #loss_file.write(str(ratio))
                    #loss_file.close()
                    check+=ratio
            print check
            #
calculate_gaze_ratio(train_list, gaze_path)
    
