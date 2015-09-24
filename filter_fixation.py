root = "/local/wangxin/data/gaze_voc_actions_stefan/"

eye_tracking_path = root+"samples/"
gaze_path = root+"train_gazes/"

action_subjects  = ["006","007","008","009","010","011","018","020"] 

pascal_voc_2012_train_images = "/local/wangxin/data/VOCdevkit_trainset/VOC2012/JPEGImages/"


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
                        try:
                            new_fixations[str(subj).strip()].append([int(x_stimulus),int(y_stimulus)])
                        except KeyError:
                            print new_fixations[filename[:-4]]
                et.close()
#                 print fixations
                new_fixations.update(old_fixations) 
                
                gaze_json = open(json_path,'w')
                json.dump(new_fixations,gaze_json)
                gaze_json.close()
valide_fixations(train_list, eye_tracking_path, valide_subjs)
