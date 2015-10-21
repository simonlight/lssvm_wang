


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
            file = "2012_000156.json"
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
                
                
                xmax,xmin,ymax,ymin,cls = ground_truth_bb(VOC2012_TRAIN_ANNOTATIONS+filename_root)
                cv2.rectangle(out, (xmin,ymin), (xmax,ymax),(0,255,255)) 
                
                
                cv2.imshow('Matched Features', out)
                cv2.waitKey(0)
                cv2.destroyAllWindows()


def slice_cnt(x,y,left, right, up, down):
    if x>=left and x<right and y>=up and y<down:
        return 1.0
    else:
        return 0.0

# the gaze ratio of the ground_truth bounding box
def IoU_gt_gaze_region(fixation_path):
    action_names=["jumping", "phoning", "playinginstrument", "reading", "ridingbike", "ridinghorse", "running", "takingphoto", "usingcomputer", "walking"]
    for root,dirs,files in os.walk(fixation_path):
        file_ratio = [0]*10
        file_num = [0]*10
        for file in files:
            class_index = -1
            filename_root = file[:-5]#.json
            fixations = json.load(open(gaze_path+file)).values()
            
            xmax,xmin,ymax,ymin,cls = ground_truth_bb(VOC2012_TRAIN_ANNOTATIONS+filename_root)

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

#gaze ratio in the sliding window with respect to the IoU of bb&ground

def ground_truth_bb(filerootname):
    action_names=["jumping", "phoning", "playinginstrument", "reading", "ridingbike", "ridinghorse", "running", "takingphoto", "usingcomputer", "walking"]
    xmltree = ET.ElementTree(file=filerootname+'.xml')            
    for elem in xmltree.iterfind('object'):
        if len(list(elem.iter('name')))>1:
            print "error of object"
        else:
            for actions in elem.iter('actions'):
                class_index = action_names.index([action.tag for action in actions if action.text is '1'][0])
                cls = action_names[class_index]
            for coor in elem.iter('bndbox'):
                xmax = int(float([coorelem.text for coorelem in coor if coorelem.tag == 'xmax'][0]))
                xmin = int(float([coorelem.text for coorelem in coor if coorelem.tag == 'xmin'][0]))
                ymax = int(float([coorelem.text for coorelem in coor if coorelem.tag == 'ymax'][0]))
                ymin = int(float([coorelem.text for coorelem in coor if coorelem.tag == 'ymin'][0]))
    return xmin,ymin,xmax,ymax, cls

def correlation_IoU_gaze_ratio(fixation_path):
    action_names=["jumping", "phoning", "playinginstrument", "reading", "ridingbike", "ridinghorse", "running", "takingphoto", "usingcomputer", "walking"]
    IoU_list=[]
    ratio_list=[]
    for root,dirs,files in os.walk(fixation_path):

        cnt = 0
        for file in files:
            cnt +=1
            print cnt
            class_index = -1
            filename_root = file[:-5]#-5 because of .json
            fixations = json.load(open(gaze_path+file)).values()
        
            xmin,ymin,xmax,ymax, cls = ground_truth_bb(VOC2012_TRAIN_ANNOTATIONS+filename_root)
            image_path = (VOC2012_TRAIN_IMAGES + filename_root)+'.jpg'
            image_res_x, image_res_y= Image.open(image_path).size 
#             print image_res_x, image_res_y
            for sy,sx in itertools.product(range(scale),repeat=2):
                hxmin, hymin = (int(sx*0.1*image_res_x), int(sy*0.1*image_res_y))   
                hxmax, hymax =(int(sx*0.1*image_res_x)+int((11-scale) * image_res_x/10),int(sy*0.1*image_res_y)+int((11-scale) * image_res_y/10))
                IoG = metric_calculate.getIoG(hxmin, hymin, hxmax, hymax, xmin, ymin, xmax, ymax)
#                 print sy,sx
                ratio_file = VOC2012_ACTION_ETLOSS_ACTION+cls+'/'+str(scale*scale)+'/'+filename_root+'_'+str(sy)+'_'+str(sx)+'.txt'
                ratio_f = open(ratio_file)
                ratio = float(ratio_f.readline().strip())
                ratio_f.close()

                if IoG>0.8 and ratio<0.1:
                    print filename_root, cls
#                 print ratio
#                 print ratio
                # any bb receives no fixations?
                # None!
#                 print IoG,ratio
                IoU_list.append(IoG)
                ratio_list.append(ratio)
        return IoU_list, ratio_list

if __name__ == "__main__":
    from matplotlib import pyplot as plt
    from path_config import *
    import json
    import cv2
    import os
    import numpy as np
    import xml.etree.cElementTree as ET
    import metric_calculate
    import Image
    import itertools
    ###CONFIG###
    fixation_path = "/local/wangxin/Data/gaze_voc_actions_stefan/train_gazes/"
    gaze_path="/local/wangxin/Data/gaze_voc_actions_stefan/train_gazes/"
    scale = 6
    ############
#     visualize_fixations(fixation_path)
    IoU, ratio = correlation_IoU_gaze_ratio(fixation_path)
#     IoU=[1,2]
#     ratio=[3,4]
    plt.scatter(IoU,ratio,c='r', s=1)
    plt.show()
    # IoU_gt_gaze_region(fixation_path)