import os

for root, dirs, files in os.walk("/home/wangxin/Data/gaze_voc_actions_stefan/example_files/"):
    for file in files:
        print root+'/'+file
        f = open(root+'/'+file)
        if not os.path.exists(root+'/'+'temp/'):
            os.mkdir(root+'/'+'temp/')
        new_f = open(root+'/'+'temp/'+file,"w")
        num = f.readline().strip()
        new_f.write(num+'\n')
        for line in f:
            components = line.strip().split(' ')
            jpg_path=components[0]
            jpg_path = jpg_path.replace("/home/wangxin/Data/gaze_voc_actions_stefan/action_train_images/",'' )
            cls = components[1]
            h= components[2] 
            new_line = ' '.join([jpg_path, cls, h])
            features =  components[3:]
            for fea in features:
                new_line = ' '.join([new_line, fea]) 
            new_f.write(new_line+'\n')
        f.close()
        new_f.close()