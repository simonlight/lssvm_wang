action_names=["jumping", "phoning", "playinginstrument", "reading", "ridingbike", "ridinghorse", "running", "takingphoto", "usingcomputer", "walking"]

scales = [1,4,9,16,25,36,49,64]
#scales = [36]
slice = 10.0

from setting import *
from PIL import Image
from xml.dom import minidom
import collections
import os
import json
import math
import numpy as np


def valide_subjects(train_list, eye_tracking_path,subjs = VOC2012_ACTION_ACTION_SUBJS):
    """The subjects obtain 100% correctness are valid, otherwise, they are abandoned"""
    f = open(train_list)
    train_images = [line.strip() for line in f]
    subj_fixationcnt = collections.defaultdict(lambda:0)
    for root, dirs, files in os.walk(eye_tracking_path):
        for file in files:
            subj, year, id = file.split('_')
            filename = '_'.join([year,id])
            if subj in subjs and filename[:-4] in train_images:
                sub_fixationcnt[subj]+=1
    print subj_fixationcnt   


def mapping(image_res_x, image_res_y, x_screen, y_screen, screen_res_x=1280.0, screen_res_y=1024.0):
    """Mapping function, given with Stefan's still image dataset"""
    sx=image_res_x/screen_res_x;
    sy=image_res_y/screen_res_y;
    s=max(sx,sy);
    dx=max(image_res_x*(1/sx-1/sy)/2,0);
    dy=max(image_res_y*(1/sy-1/sx)/2,0);
    x_stimulus=s*(x_screen-dx);
    y_stimulus=s*(y_screen-dy);
    return x_stimulus, y_stimulus

def valide_fixations(train_list, eye_tracking_path, valide_subjs, eye_tracking_json_path):
    """Write gazes into .json"""
    tl = open(train_list)
    train_images = [line.strip() for line in tl]
    tl.close()   
    os.mkdir(eye_tracking_json_path)
                    
    for root, dirs, files in os.walk(eye_tracking_path):
        for file in files:
            subj, year, id = file.split('_')
            if not subj in valide_subjs:
                continue
            filename = '_'.join([year,id])[:-4]
            if filename in train_images:
                json_path = eye_tracking_json_path+filename[:-4]+'.json'
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
                new_fixations.update(old_fixations) 
                
                gaze_json = open(json_path,'w')
                json.dump(new_fixations,gaze_json)
                gaze_json.close()
                
def slice_cnt(x,y,left, right, up, down):
    if x>=left and x<right and y>=up and y<down:
        return 1.0
    else:
        return 0.0

def calculate_gaze_ratio(train_list, gaze_path, annotations = VOC2012_TRAIN_ANNOTATIONS):
    tl = open(train_list)
    train_images = [line.strip() for line in tl]
    tl.close()
    for c,im in enumerate(train_images):
        print c
        fixation_file = open(gaze_path+im[:-4]+'.json')
                
        fixations = json.load(fixation_file)
        fixation_file.close()
        total_fixations = sum([len(observers) for observers in fixations.values()])
        
        xmldoc = minidom.parse(annotations + im.strip()[:-4]+'.xml')
                 
        itemlist = xmldoc.getElementsByTagName("actions")
        for action in enumerate(action_names):
            if int(itemlist[0].getElementsByTagName(action[1])[0].childNodes[0].nodeValue) ==1:
                action_category = action[1]
                continue
       
        image_res_x, image_res_y= Image.open(pascal_voc_2012_train_images+im).size
        

        integrate_image = np.zeros((10,10))
        for d1_inc in range(0,10):
            for d2_inc in range(0,10):
                left = (d2_inc) * image_res_x/10.0
                right = (d2_inc+1) * image_res_x/10.0
                up = (d1_inc) * image_res_y/10.0
                down = (d1_inc+1) * image_res_y/10.0
                for ob in fixations.values():
                    for (point_x, point_y) in ob:
                         integrate_image[d1_inc][d2_inc]+=slice_cnt(point_x, point_y, left, right, up, down)
        for scale in scales:
            block_num = int(math.sqrt(scale))
            check=0
            for i_x in range(block_num):
                for i_y in range(block_num):
                    ratio = np.sum(integrate_image[i_x:11-block_num+i_x, i_y:11-block_num+i_y])/total_fixations
                    folder = +str(scale)+'/'
                    if not os.path.exists(folder):
                        os.makedirs(folder)
                    if scale == 1:
                        etloss_filename = folder + im[:-4]+'.txt'
                    else:
                        etloss_filename = folder + im[:-4]+'_'+str(i_x)+'_'+str(i_y)+'.txt'
                    loss_file = open(etloss_filename,'w')
                    loss_file.write(str(ratio))
                    loss_file.close()
                    check+=ratio
    
if __name__ == "__main__":
    valide_fixations(VOC2012_ACTION_TRAIN_LIST, VOC2012_ACTION_EYE_PATH, VOC2012_ACTION_VALIDE_SUBJS, VOC2012_ACTION_EYE_CONTEXT_JSON_PATH)
#     calculate_gaze_ratio(VOC2012_ACTION_TRAIN_LIST, gaze_path)