#How many images are left according to ferrari. They claim 6270+91 images. But we do have 6452 images
classes=['cat', 'dog', 'bicycle', 'motorbike', 'boat', 'aeroplane', 'horse', 'cow', 'sofa', 'diningtable']
ppp=0
for c in classes:
    for line in open("/media/workspace/work_data/VOC2012/VOC2012/ImageSets/Main/"+c+"_trainval.txt"):
        name,res = line.split()
        if res == '1':
            ppp+=1
print ppp#6452