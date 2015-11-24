import os
import itertools
import path_config
import re
#generate example_files for training and validation set
def contains_object(res):
    if res in ["-1", "0"]:
        return '0'
    elif res == "1":
        return "1"

def contains_class(im, cls):
    if im.split("_")[0]==cls:
        return '1'
    else:
        return '0'

# # eye-trackered images we actually have
# def action_image_list(root):
#     #there are some images are duplicated in content with different names
#     list_files = [files for root, dirs, files in os.walk(root)][0]
#     return list_files
# 
# def full_action_image_list(example_list):
#     f = open(example_list)
#     return [item.strip() for item in f.readlines()]


def single_name_label(refpath):
    name_label = {}
    with open(refpath) as t:
        for line in t:
            name, _, label = line.split()
            name = name.strip()
            label = label.strip()
            name_label[name]=label
    return name_label

def full_name_label(refpath):
    name_label = {}
    with open(refpath) as t:
        for line in t:
            name, _, label = line.split()
            name = name.strip()
            label = label.strip()
            #if the image contains an action, it's labeled 1
            if name in name_label.keys() and name_label[name] == "1":
                continue
            name_label[name]=label
                
    return name_label

#param: val|train|test
#refpath: [train_file, val_file, trainval_file ]path
#cls: class name 
#file_typ: "train", "val", "trainval"
#scale: 30-100
#vm: validated images. Some images in ferrari's data are not in the original exp_type
#exp_type: experiment type: 'fuul' 'reduit' 'ground'
def generate(example_list_dir, refpath, cls, file_typ, scale):
    
    if not os.path.exists(example_list_dir):
        os.makedirs(example_list_dir)
    
    f = open('_'.join([example_list_dir+cls, file_typ, 'scale', str(scale), 'matconvnet_m_2048_layer_20.txt']),'w')
    print '_'.join([example_list_dir+cls, file_typ, 'scale', str(scale), 'matconvnet_m_2048_layer_20.txt'])
    
    name_label = full_name_label(refpath)
    #image quantity
    f.write(str(len(name_label))+"\n")
    
    if scale != 100:
        suffix = ['_'+str(i)+'_'+str(j) for i,j in itertools.product(range((100-scale)/10+1),range((100-scale)/10+1))]
    else:
        suffix = ['']
    
    #different images
        
    for filename_root in name_label.keys():
        content = '/home/wangxin/Data/full_stefan_gaze/'+'action_'+file_typ+'_images/'+filename_root
#             print content
        content += ' ' + contains_object(name_label[filename_root])
        #content += ' ' + contains_class(im, cls)
        content += ' ' + str(int(11 - 0.1 * scale) ** 2)
        for suf in suffix:
            content += ' ' + '/home/wangxin/Data/full_stefan_gaze/'+'m2048_trainval_features/'+str(scale)+'/'+filename_root +suf+'.txt'
        f.write(content+'\n')    
    else:
        pass


if __name__ =="__main__":
    from path_config import *
    #test/train files root
    file_root = "/local/wangxin/Data/VOCdevkit_trainset/VOC2012/ImageSets/Action/"
    
#     validated_images = full_action_image_list("/local/wangxin/Data/VOCdevkit_trainset/VOC2012/ImageSets/Action/train.txt")
    for scale in range(100,29, -10):
        print scale
        for cls in path_config.VOC2012_ACTION_CATEGORIES:
            example_list_dir = "/local/wangxin/Data/full_stefan_gaze/example_files/"+str(scale)+"/"
            training_file = file_root + cls + '_train.txt'
            val_file = file_root + cls + '_val.txt'
            trainval_file = file_root + cls + '_trainval.txt'
            generate(example_list_dir, training_file, cls, "train", scale)
            generate(example_list_dir, val_file, cls, "val", scale)
            generate(example_list_dir, trainval_file, cls, "trainval",scale)
