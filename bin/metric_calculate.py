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
    
def scale2area(width, height, scale):
    return width*scale/100.0, height * scale / 100.0

#from latent region id to region coordiantes
def h2Coor(width, height, h, scale):
    grid_1, grid_2 = h2GridCoor(h, scale)
    hxmin = 0.1*grid_2*width
    hymin = 0.1*grid_1*height
    area  = scale2area(width, height, scale)
    hxmax = hxmin + area[0]
    hymax = hymin + area[1]
    return hxmin, hymin, hxmax, hymax

def h2GridCoor(h, scale):
    grid_1 = int(h)/int(math.sqrt(convert_scale(scale)))
    grid_2 = int(h)%int(math.sqrt(convert_scale(scale)))
    return grid_1, grid_2

def getIoU(hxmin, hymin, hxmax, hymax, xmin, ymin, xmax, ymax):
    left = max(xmin,hxmin)+0.0
    right = min(xmax,hxmax)
    upper = max(ymin,hymin)
    bottom = min(ymax,hymax)
    I = max(0,(right-left))*max(0,(bottom-upper))
    U = (hxmax - hxmin)*(hymax - hymin)+(xmax - xmin)*(ymax - ymin) - I
    return I/U

def getTopIoU(hxmin, hymin, hxmax, hymax, bbs):  
    topIoU=0
    for xmax,xmin,ymax,ymin in bbs:
        IoU = getIoU(hxmin, hymin, hxmax, hymax, xmax,xmin,ymax,ymin)
        if IoU>topIoU:
            topIoU = IoU
    return topIoU

def getIoG(hxmin, hymin, hxmax, hymax, xmin, ymin, xmax, ymax):
    left = max(xmin,hxmin)+0.0
    right = min(xmax,hxmax)
    upper = max(ymin,hymin)
    bottom = min(ymax,hymax)
    I = max(0,(right-left))*max(0,(bottom-upper))
    G = (xmax-xmin)*(ymax-ymin)
    return I/G

def getTopIoG(hxmin, hymin, hxmax, hymax, bbs):
    topIoG=0
    for xmax,xmin,ymax,ymin in bbs:
        IoG = getIoG(hxmin, hymin, hxmax, hymax, xmax,xmin,ymax,ymin)
        if IoG>topIoG:
            topIoG = IoG
    return topIoG

if __name__ == '__main__':
    import xml.etree.cElementTree as ET
    from PIL import Image
    