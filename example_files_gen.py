import os
import itertools

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
# eye-trackered images we actually have
def etImages(root = "/home/wangxin/Data/ferrari_data/POETdataset/POETdataset/PascalImages"):
    #there are some images are duplicated in content with different names
    list_files = [files for root, dirs, files in os.walk(root)][0]
    return list_files

#param: val|train|test
#refpath: [train_file, val_file, trainval_file ]path
#cls: class name 
#file_typ: "train", "val", "trainval"
#scale: 30-100
#vm: validated images. Some images in ferrari's data are not in the original exp_type
#exp_type: experiment type: 'fuul' 'reduit' 'ground'
def generate(refpath, cls, file_typ, scale, vm):
    example_list_dir = "/home/wangxin/Data/ferrari_data/POETdataset/POETdataset"+"/example_files_pos_val/"+str(scale)+"/"
    if not os.path.exists(example_list_dir):
        os.makedirs(example_list_dir)
    f = open('_'.join([example_list_dir+cls, file_typ, 'scale', str(scale), 'matconvnet_m_2048_layer_20.txt']),'w')
#     if file_typ == "train":
#         f.write("3103\n")
#     elif file_typ == "val":
#         f.write("3167\n")
    if scale !=100:
        suffix = ['_'+str(i)+'_'+str(j) for i,j in itertools.product(range((100-scale)/10+1),range((100-scale)/10+1))]
    else:
        suffix = ['']
    
    ref_files={}
    
    with open(refpath) as t:
        for line in t:
            name, res = line.split()
            name = name.strip()+'.jpg'
            res = res.strip()
            ref_files[name]=res
            """print cls+'_'+name+'.jpg'
            print vm[0]
            """
    cnt=0
    for im in vm:
        original_name='_'.join([im.split('_')[1],im.split('_')[2]])
        if original_name in ref_files.keys():
            if contains_object(ref_files[original_name.split('.')[0]+'.jpg']) == '1':
                cnt+=1
        else:
            pass
    f.write(str(cnt)+'\n')
    for im in vm:
        original_name='_'.join([im.split('_')[1],im.split('_')[2]])
        if original_name in ref_files.keys():
            content = '/home/wangxin/Data/ferrari_data/POETdataset/POETdataset/PascalImages/'+im
            #print content
            content += ' ' + contains_object(ref_files[original_name.split('.')[0]+'.jpg'])
            if contains_object(ref_files[original_name.split('.')[0]+'.jpg']) == '1':
                #content += ' ' + contains_class(im, cls)
                content += ' ' + str(int(11 - 0.1 * scale) ** 2)
                for suf in suffix:
                    content += ' ' + '/home/wangxin/Data/ferrari_data//POETdataset/POETdataset'+'/matconvnet_m_2048_features/'+str(scale)+'/'+original_name.split('.')[0] +suf+'.txt'
                f.write(content+'\n')    
        else:
            pass
#test/train files root
file_root = "/home/wangxin/Data/VOC2012/VOC2012/ImageSets/Main/"

classes = ['cat', 'dog', 'bicycle', 'motorbike', 'boat', 'aeroplane', 'horse', 'cow', 'sofa', 'diningtable']
validated_images = ['_'.join([name.split('_')[1],name.split('_')[2]])for name in etImages()]#6452
print len(validated_images)
for i in range(100, 29, -10):
    print i
    for cls in classes:
        training_file = file_root + cls + '_train.txt'
        val_file = file_root + cls + '_val.txt'
        #trainval_file = file_root + cls + '_trainval.txt'
        #generate(training_file, cls, "train", i, etImages())
        generate(val_file, cls, "val", i, etImages())
        #generate(trainval_file, cls, "trainval", i, etImages(),'full')
