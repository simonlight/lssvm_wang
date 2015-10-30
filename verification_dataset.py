# Verify if every file in the file list is in the file folder: OF COURSE?
import os.path
pascal_voc_2012_action_train_listpath = "/local/wangxin/Data/VOCdevkit_trainset/VOC2012/ImageSets/Action/"
pascal_voc_2012_train_images = "/local/wangxin/Data/VOCdevkit_trainset/VOC2012/JPEGImages/"

pascal_voc_2012_action_test_listpath = "/local/wangxin/Data/VOCdevkit_testset/VOC2012/ImageSets/Action/"
pascal_voc_2012_test_images = "/local/wangxin/Data/VOCdevkit_testset/VOC2012/JPEGImages/"

train_list_files = ["train.txt", "val.txt", "trainval.txt"]
test_list_files = ["test.txt"]

def verify_num(list_path, list_files, image_files):
    for l in list_files:
        f = open(list_path + l)
        for line in enumerate(f):
            if not os.path.exists(image_files+line[1].strip()+".jpg"):
                print "no file %s" % line[1].strip()+".jpg"
        print "%s: %d files are verified"%(l,line[0]+1)

verify_num(pascal_voc_2012_action_train_listpath, train_list_files, pascal_voc_2012_train_images)
verify_num(pascal_voc_2012_action_test_listpath, test_list_files, pascal_voc_2012_test_images)

# queren