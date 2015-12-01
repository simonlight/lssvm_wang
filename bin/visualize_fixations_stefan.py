


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
            file = "2011_006217.json"
            
            
            for sy,sx in itertools.product(range(scale),repeat=2):
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

                start = (int(sx*0.1*cols), int(sy*0.1*rows))        
                end =(int(sx*0.1*cols)+int((11-scale) * cols/10),int(sy*0.1*rows)+int((11-scale) * rows/10))
                cv2.rectangle(out, start,end,(0,0,255)) 
                
                xmin,ymin,xmax,ymax,cls = ground_truth_bb(VOC2012_TRAIN_ANNOTATIONS+filename_root)
                cv2.rectangle(out, (xmin,ymin), (xmax,ymax),(0,255,255)) 
                
                gaze_ratio_file = open(VOC2012_ACTION_ETLOSS_ACTION+cls+'/'+str(scale*scale)+'/'+filename_root+'_'+str(sy)+'_'+str(sx)+'.txt')
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

def metric_file_analyse(metric_folder, typ):
    categories=["jumping", "phoning", "playinginstrument", "reading", "ridingbike", "ridinghorse", "running", "takingphoto", "usingcomputer", "walking"]
    scale_cv=[50]
    tradeoff_cv = [0.0,0.5]
    epsilon_cv = [0.01]
    lambda_cv = ['1.0E-4']
    for scale in scale_cv:        
        for epsilon in epsilon_cv:
            for lambd in lambda_cv: 
                for category in categories:  
                    for tradeoff in tradeoff_cv:
#                         best_cv = get_best_cv(cls, scale)
                        if typ=='':
                            my_typ = typ
                        else:
                            my_typ = typ+'_'

                        f= open(VOC2012_ACTION_METRIC_ROOT+metric_folder+"metric_"+my_typ+str(tradeoff)+'_'+str(scale)+"_"+str(epsilon)+"_"+str(lambd)+"_"+category+".txt")
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
                            yp, yi, hp, image_path = line.strip().split(',')
                            if yi=='1':
                                positive+=1
                            elif yi=='0':
                                negative+=1
                            
                            if yi=='1' and yp=='1':
                                tp+=1
                                cnt+=1
                                filename_root,_ = image_path.split("/")[-1].split('.')
                                xmin,ymin,xmax,ymax, _ = ground_truth_bb(VOC2012_TRAIN_ANNOTATIONS+filename_root)
                                grid_1, grid_2 = metric_calculate.h2GridCoor(hp, scale)
                                
                                ratio_file = VOC2012_ACTION_ETLOSS_ACTION+category+'/'+str(metric_calculate.convert_scale(scale))+'/'+filename_root+'_'+str(grid_1)+'_'+str(grid_2)+'.txt'
                                ratio_f = open(ratio_file)
                                ratio = float(ratio_f.readline().strip())
                                ratio_f.close()
                                
                                fixation_ratio+=ratio
#                                 print hp,fixation_ratio,hp,filename_root
                                im = Image.open(VOC2012_TRAIN_IMAGES+filename_root+'.jpg')
                                width, height = im.size
                                #0, 0, 250, 187.5
                                hxmin, hymin, hxmax, hymax = metric_calculate.h2Coor(width, height, hp, scale)
                                IoU = metric_calculate.getIoU(hxmin, hymin, hxmax, hymax, xmin, ymin, xmax, ymax)
                                totalIoU += IoU
#                                 print filename_root,hp
                            elif yi=='0' and yp=='0':
                                tn+=1
                            elif yi=='0' and yp=='1':
                                fp+=1
                            elif yi=='1' and yp=='0':
                                fn+=1
                            
                                
                        print "content:%s, category:%s, tradeoff:%.1f, scale:%d, epsilon:%f, lambda:%s, averageIoU:%f, TP:%f, TN:%f, FP:%f, FN:%f, acc:%f, fixation ratio:%f\n"%\
                              (metric_folder, category, tradeoff, scale, epsilon, lambd, totalIoU/cnt, tp, tn, fp, fn, tp+tn, fixation_ratio/cnt)
    

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
    metric_file_analyse(metric_folder = "C1e-4_e1e-2_stefan_train/", typ="train")
    visualize_fixations(fixation_path)
#     IoU, ratio = correlation_IoU_gaze_ratio(fixation_path)
# #     IoU=[1,2]
# #     ratio=[3,4]
#     plt.scatter(IoU,ratio,c='r', s=1)
#     plt.show()
    # IoU_gt_gaze_region(fixation_path)