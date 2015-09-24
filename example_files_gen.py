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
def etImages(root ):
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
    example_list_dir = "/local/wangxin/Data/gaze_voc_actions_stefan/examples_files/"+str(scale)+"/"
    if not os.path.exists(example_list_dir):
        os.makedirs(example_list_dir)
    f = open('_'.join([example_list_dir+cls, file_typ, 'scale', str(scale), 'matconvnet_m_2048_layer_20.txt']),'w')
    print '_'.join([example_list_dir+cls, file_typ, 'scale', str(scale), 'matconvnet_m_2048_layer_20.txt'])
#     if file_typ == "train":
#         f.write("3103\n")
#     elif file_typ == "val":
#         f.write("3167\n")
    if scale != 100:
        suffix = ['_'+str(i)+'_'+str(j) for i,j in itertools.product(range((100-scale)/10+1),range((100-scale)/10+1))]
    else:
        suffix = ['']
    
    ref_files={}
    
    with open(refpath) as t:
        for line in t:
            name, _,res = line.split()
            name = name.strip()+'.jpg'
            res = res.strip()
            ref_files[name]=res
            """print cls+'_'+name+'.jpg'
            print vm[0]
            """
    cnt=0
    for im in vm:
        original_name='_'.join([im.split('_')[0],im.split('_')[1]])
        if original_name in ref_files.keys():
#             if contains_object(ref_files[original_name.split('.')[0]+'.jpg']) == '1':
            cnt+=1
        else:
            pass
    f.write(str(cnt)+'\n')
    for im in vm:
        original_name='_'.join([im.split('_')[0],im.split('_')[1]])
        if original_name in ref_files.keys():
            content = '/home/wangxin/Data/gaze_voc_actions_stefan/'+'action_'+file_typ+'_images/'+im
#             print content
            content += ' ' + contains_object(ref_files[original_name.split('.')[0]+'.jpg'])
            #content += ' ' + contains_class(im, cls)
            content += ' ' + str(int(11 - 0.1 * scale) ** 2)
            for suf in suffix:
                content += ' ' + '/home/wangxin/Data/gaze_voc_actions_stefan/'+'m2048_trainval_features/'+str(scale)+'/'+original_name.split('.')[0] +suf+'.txt'
            f.write(content+'\n')    
        else:
            pass
#test/train files root
file_root = "/local/wangxin/Data/VOCdevkit_trainset/VOC2012/ImageSets/Action/"

#classes = ['cat', 'dog', 'bicycle', 'motorbike', 'boat', 'aeroplane', 'horse', 'cow', 'sofa', 'diningtable']
classes = ['jumping', 'phoning', 'playinginstrument', 'reading', 'ridingbike', 'ridinghorse', 'running', 'takingphoto', 'usingcomputer', 'walking']
validated_images = etImages("/local/wangxin/Data/gaze_voc_actions_stefan/action_train_images/")
print len(validated_images)
for i in range(100,29, -10):
    print i
    for cls in classes:
        training_file = file_root + cls + '_train.txt'
        val_file = file_root + cls + '_val.txt'
#         trainval_file = file_root + cls + '_trainval.txt'
        generate(training_file, cls, "train", i, validated_images)
        generate(val_file, cls, "val", i, validated_images)
#         generate(trainval_file, cls, "trainval", i, validated_images,'full')
