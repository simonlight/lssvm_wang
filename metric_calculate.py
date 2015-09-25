import xml.etree.cElementTree as ET
from PIL import Image
import math
def convert_scale(scale):
    return ((100 - scale)/10+1)**2

def get_best_cv(cls, scale):
    #cat 100 0.6 0.92
    best_res_root  = "/home/wangxin/Data/ferrari_data/reduit_allbb/best_cv_res/"
    method = "std_et.txt"
    f = open(best_res_root+method)
    for line in f:
        c, s, best_cv, _ = line.strip().split(' ')
        if c.strip()== cls and s.strip() == str(scale):
            return best_cv
    
def scale2area(width,height,scale):
    return width*scale/100.0, height * scale / 100.0

#from latent region id to region coordiantes
def h2coor(width, height, h, scale):
    hx = int(h)/int(math.sqrt(convert_scale(scale)))
    hy = int(h)%int(math.sqrt(convert_scale(scale)))
    
    hxmin = 0.1*hy*width
    hymin = 0.1*hx*height
    area  = scale2area(width, height, scale)
    hxmax = hxmin + area[0]
    hymax = hymin + area[1]
    return hxmin, hymin, hxmax, hymax

def getIoU(hxmin, hymin, hxmax, hymax, xmin, ymin, xmax, ymax):
    left = max(xmin,hxmin)+0.0
    right = min(xmax,hxmax)
    upper = max(ymin,hymin)
    bottom = min(ymax,hymax)
    if right-left>0 and bottom-upper >0:
        I = (right-left)*(bottom-upper)
        U = (hxmax - hxmin)*(hymax - hymin)+(xmax - xmin)*(ymax - ymin) - I
        return I/U
    else:
        return 0.0
    
# classes=['cat', 'dog', 'bicycle', 'motorbike', 'boat', 'aeroplane', 'horse', 'cow', 'sofa', 'diningtable']
classes=["jumping", "phoning", "playinginstrument", "reading", "ridingbike", "ridinghorse", "running", "takingphoto", "usingcomputer", "walking"]

#classes = ['boat']
scales=[50]
# scales=[70]
root="/home/wangxin/results/gaze_voc_actions_stefan/std_et/std_et/"
for cls in classes:
    
    for scale in scales:        
        best_cv = get_best_cv(cls, scale)
        f= open(root+"metric_"+str(scale)+"_"+cls+"_"+str(best_cv)+"_pos_neg.txt")
        cnt = 0
        totalIoU = 0.0
#         object = False
        for line in f:
            yp, h, image_path = line.strip().split(',')
            c, year, imp = image_path.split("/")[-1].split('_')
            im_path_code = year+'_'+imp
            tree = ET.ElementTree(file = '/home/wangxin/Data/VOCdevkit_trainset/VOC2012/Annotations/'+im_path_code[:-4]+'.xml')
            im = Image.open(image_path)
            width, height = im.size
            hxmin, hymin, hxmax, hymax = h2coor(width, height, h, scale)
            predIoU = 0.0
            for elem in tree.iter(tag='object'):
                if elem[0].text == 'person': 
                    object = True
                    if elem[1].tag == "bndbox":
                        xmax = int(elem[1][0].text)
                        xmin = int(elem[1][1].text)
                        ymax = int(elem[1][2].text)
                        ymin = int(elem[1][3].text)
                        IoU = getIoU(hxmin, hymin, hxmax, hymax, xmin, ymin, xmax, ymax)
#                         print hxmin, hymin, hxmax, hymax, xmin, ymin, xmax, ymax
                        if IoU > predIoU:
                            predIoU = IoU
            if object == True:
                cnt += 1.0
            totalIoU += predIoU   
#         print "for class "+cls+ " scale:"+str(scale) + "the IoU percentage is: " + str(totalIoU/cnt)
        print str(totalIoU/cnt)