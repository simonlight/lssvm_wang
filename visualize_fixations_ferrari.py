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

def read_fixations(fixation_file, cls, eye_path):
    f = open(eye_path+cls + '_' + fixation_file)
    fixations = []
    for line in f: 
        x,y = line.strip().split(',')
        fixations.append([int(x),int(y)])
    f.close()
    return fixations                

def slice_cnt(x,y,left, right, up, down):
    if x>left and x<=right and y>up and y<=down:
        return 1.0
    else:
        return 0.0

def ground_truth_bb_all(filerootname):
    xmltree = ET.ElementTree(file=filerootname+'.xml')            
            
    #bb of objects of given class
    bbs=[]
    for category in VOC2012_OBJECT_CATEGORIES:        
        for elem in xmltree.iterfind('object'):
            for name in elem.iter('name'):
                
                if name.text == category:                            
                    for coor in elem.iter('bndbox'):
                        xmax = [coorelem.text for coorelem in coor if coorelem.tag == 'xmax'][0]
                        xmin = [coorelem.text for coorelem in coor if coorelem.tag == 'xmin'][0]
                        ymax = [coorelem.text for coorelem in coor if coorelem.tag == 'ymax'][0]
                        ymin = [coorelem.text for coorelem in coor if coorelem.tag == 'ymin'][0]
                       
                        bbs.append([int(float(xmin)),int(float(ymin)),int(float(xmax)),int(float(ymax))])
    return bbs

def ground_truth_bb(filerootname, category):
    xmltree = ET.ElementTree(file=filerootname+'.xml')            
            
    #bb of objects of given class
    bbs=[]
    for elem in xmltree.iterfind('object'):
        for name in elem.iter('name'):
            
            if name.text == category:                            
                for coor in elem.iter('bndbox'):
                    xmax = [coorelem.text for coorelem in coor if coorelem.tag == 'xmax'][0]
                    xmin = [coorelem.text for coorelem in coor if coorelem.tag == 'xmin'][0]
                    ymax = [coorelem.text for coorelem in coor if coorelem.tag == 'ymax'][0]
                    ymin = [coorelem.text for coorelem in coor if coorelem.tag == 'ymin'][0]
                   
                    bbs.append([int(float(xmin)),int(float(ymin)),int(float(xmax)),int(float(ymax))])
    return bbs

def visualize_fixations(fixation_path):

    for root,dirs,files in os.walk(fixation_path):
        for file in files:
            cls, year, id= file.split('_')
            cls = 'aeroplane'
            year='2011'
            id='002222.ggg'
            id=id[:-4]
            filename_root= '_'.join([year,id])
    #         file = "2012_003108.json"
    #         filename_root = file[:-5]#.json
            fixation = []
            f = open(root+cls+'_'+year+'_'+id+'.txt')
            for line in f:
                x,y = line.strip().split(',')
                fixation.append([int(x),int(y)])
            f.close()
    
            img = VOC2012_TRAIN_IMAGES+year+'_'+id+'.jpg'
            img = cv2.imread(img,0)
            rows = img.shape[0]
            cols = img.shape[1]
            
            for sy,sx in itertools.product(range(scale),repeat=2):
                out = np.zeros((rows,cols,3), dtype='uint8')
                out= np.dstack([img, img, img])
                for x,y in fixation:
                    cv2.circle(out, (x,y),3,[255,255,0],1)
                start = (int(sx*0.1*cols), int(sy*0.1*rows))        
                end =(int(sx*0.1*cols)+int((11-scale) * cols/10),int(sy*0.1*rows)+int((11-scale) * rows/10))
                cv2.rectangle(out, start,end,(0,0,255)) 
                
                bbs = ground_truth_bb(VOC2012_TRAIN_ANNOTATIONS+filename_root, cls)
                for xmin,ymin,xmax,ymax in bbs:
                    cv2.rectangle(out, (xmin,ymin), (xmax,ymax),(0,255,255))
                
                if scale ==1:
                    gaze_ratio_file = open(VOC2012_OBJECT_ETLOSS_ACTION+cls+'/'+str(scale*scale)+'/'+cls+'_'+filename_root+'.txt')
                else:
                    gaze_ratio_file = open(VOC2012_OBJECT_ETLOSS_ACTION+cls+'/'+str(scale*scale)+'/'+cls+'_'+filename_root+'_'+str(sy)+'_'+str(sx)+'.txt')
                

                gaze_ratio=gaze_ratio_file.readline().strip()
                gaze_ratio_file.close()
                
                cv2.putText(out, str(gaze_ratio[:4])+','+str(sy*6+sx),\
                            (int(0.5*(start[0]+end[0])),int(0.5*(start[1]+end[1]))),\
                            cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,0,255),2
                            )
                cv2.imshow(cls+' '+filename_root, out)
                
                k = cv2.waitKey(0)
                #space to next image
                cv2.destroyAllWindows()
                if k == 1048608:
                    break
                else:
                    continue

                
def IoU_gt_gaze_region(fixations):
    object_names=["dog", "cat", "motorbike", "boat", "aeroplane", "horse" ,"cow", "sofa", "diningtable", "bicycle"]
    no_gaze=0
    for root,dirs,files in os.walk(VOC2012_OBJECT_EYE_PATH):
        file_ratio = [0]*10
        file_num = [0]*10
        for cnt, filename in enumerate(files):
            fixation_file = '_'.join(([filename.split('_')[1], filename.split('_')[2]]))
            im = fixation_file[:-4]+'.jpg'
            cls = filename.split('_')[0]
            class_index = object_names.index(cls)
            fixations = read_fixations(fixation_file, cls, VOC2012_OBJECT_EYE_PATH)
            
            bbs = ground_truth_bb(VOC2012_TRAIN_ANNOTATIONS+fixation_file[:-4], cls)
            
            ingaze = 0
            cnt=0
            for (point_x, point_y) in fixations:
                cnt+=1
                for bb in bbs:
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

def correlation_IoU_gaze_ratio(fixation_path):
    object_names=["dog", "cat", "motorbike", "boat", "aeroplane", "horse" ,"cow", "sofa", "diningtable", "bicycle"]
    IoU_list=[]
    ratio_list=[]
    for root,dirs,files in os.walk(fixation_path):
        file_ratio = [0]*10
        file_num = [0]*10
        for cnt, filename in enumerate(files):
            print cnt
            fixation_file = '_'.join(([filename.split('_')[1], filename.split('_')[2]]))
            filename_root = fixation_file[:-4]
            im = filename_root+'.jpg'
            cls = filename.split('_')[0]
            class_index = object_names.index(cls)
            fixations = read_fixations(fixation_file, cls, VOC2012_OBJECT_EYE_PATH)
            
            bbs = ground_truth_bb(VOC2012_TRAIN_ANNOTATIONS+fixation_file[:-4], cls)
            image_path = (VOC2012_TRAIN_IMAGES + filename_root)+'.jpg'
            image_res_x, image_res_y= Image.open(image_path).size 
#             print image_res_x, image_res_y
            
            for sy,sx in itertools.product(range(scale),repeat=2):
                hxmin, hymin = (int(sx*0.1*image_res_x), int(sy*0.1*image_res_y))   
                hxmax, hymax =(int(sx*0.1*image_res_x)+int((11-scale) * image_res_x/10),int(sy*0.1*image_res_y)+int((11-scale) * image_res_y/10))
                topIoG = metric_calculate.getTopIoG(hxmin, hymin, hxmax, hymax, bbs)
#                 print sy,sx
                ratio_file = VOC2012_OBJECT_ETLOSS_ACTION+cls+'/'+str(scale*scale)+'/'+cls+'_'+filename_root+'_'+str(sy)+'_'+str(sx)+'.txt'
                ratio_f = open(ratio_file)
                ratio = float(ratio_f.readline().strip())
                ratio_f.close()

#                     if IoG>0.8 and ratio<0.1:
#                         print filename_root, cls
#                 print ratio
#                 print ratio
                # any bb receives no fixations?
                # None!
#                 print IoG,ratio
                IoU_list.append(topIoG)
                ratio_list.append(ratio)
        return IoU_list, ratio_list

def metric_file_analyse(metric_folder, typ, scale, print_mode):
#     categories=["jumping", "phoning", "playinginstrument", "reading", "ridingbike", "ridinghorse", "running", "takingphoto", "usingcomputer", "walking"]
    tradeoff_cv = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
    epsilon_cv = [0.001]
    lambda_cv = ['1.0E-4']
    for epsilon in epsilon_cv:
        for lambd in lambda_cv: 
            for category in VOC2012_OBJECT_CATEGORIES:  
#             for category in ["boat"]:
                for tradeoff in tradeoff_cv:
#                         best_cv = get_best_cv(cls, scale)
                    if typ=='':
                        my_typ = typ
                    else:
                        my_typ = typ+'_'

                    f= open(VOC2012_OBJECT_METRIC_ROOT+metric_folder+"metric_"+my_typ+str(tradeoff)+'_'+str(scale)+"_"+str(epsilon)+"_"+str(lambd)+"_"+category+".txt")
                    totalIoU = 0.0
                    cnt=0
                    tp = 0
                    tn =0
                    fp=0
                    fn=0
                    positive=0
                    negative=0
                    fixation_ratio=0
            #         object = False
                    for line in f:
                        yp, yi, hp, filename_root = line.strip().split(',')
                        if yi=='1':
                            positive+=1
                        elif yi=='0':
                            negative+=1
                        
                        if yi=='1':
#                         if True:
                            tp+=1
                            cnt+=1
#                                 print filename_root
                            gaze_file_root = '_'.join([category, filename_root,])
                            xml_filename_root = filename_root
                            bbs = ground_truth_bb(VOC2012_TRAIN_ANNOTATIONS+xml_filename_root, category)
#                             bbs = ground_truth_bb_all(VOC2012_TRAIN_ANNOTATIONS+xml_filename_root)
                            grid_1, grid_2 = metric_calculate.h2GridCoor(hp, scale)
                            
                            ratio_file = VOC2012_OBJECT_ETLOSS_ACTION+category+'/'+str(metric_calculate.convert_scale(scale))+'/'+gaze_file_root+'_'+str(grid_1)+'_'+str(grid_2)+'.txt'
                            try:
                                ratio_f = open(ratio_file)
                                ratio = float(ratio_f.readline().strip())
                                ratio_f.close()
                                fixation_ratio+=ratio
                            except IOError:
#                                 print filename_root
                                pass

#                                 print hp,fixation_ratio,hp,filename_root
                            im = Image.open(VOC2012_TRAIN_IMAGES+xml_filename_root+'.jpg')
                            width, height = im.size
                            #0, 0, 250, 187.5
                            hxmin, hymin, hxmax, hymax = metric_calculate.h2Coor(width, height, hp, scale)
                            IoU = metric_calculate.getTopIoU(hxmin, hymin, hxmax, hymax, bbs)
                            totalIoU += IoU
#                                 print filename_root,hp
                        elif yi=='0' and yp=='0':
                            tn+=1
                        elif yi=='0' and yp=='1':
                            fp+=1
                        elif yi=='1' and yp=='0':
                            fn+=1
                        
                    if print_mode == "print":
                        print totalIoU/cnt

                     
                    else:
                        print "content:%s, category:%s, tradeoff:%.1f, scale:%d, epsilon:%f, lambda:%s, averageIoU:%f, TP:%f, TN:%f, FP:%f, FN:%f, acc:%f, fixation ratio:%f\n"%\
                       (metric_folder, category, tradeoff, scale, epsilon, lambd, totalIoU/cnt, tp, tn, fp, fn, tp+tn,fixation_ratio/cnt)
                    
                print category+"**********"
if __name__ == "__main__":
    import json
    import cv2
    import os
    import numpy as np
    import itertools
    from matplotlib import pyplot as plt
    from path_config import *
    import Image
    import xml.etree.cElementTree as ET
    import metric_calculate
    
    ###CONFIG###
    ############
    print "90 positive test iou"
    metric_file_analyse(metric_folder = "C1e-4_e1e-3_scale_90_cv_gamma_ferrari/", typ="test", scale = 90, print_mode= "print")
#     visualize_fixations(VOC2012_OBJECT_EYE_PATH)
#     IoU, ratio = correlation_IoU_gaze_ratio(VOC2012_OBJECT_EYE_PATH)
# #     IoU=[1,2]
# #     ratio=[3,4]
#     plt.scatter(IoU,ratio,c='r', s=1)
#     plt.show() 