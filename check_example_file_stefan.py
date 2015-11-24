if __name__ == "__main__":
#     my_file = open("/local/wangxin/Data/gaze_voc_actions_stefan/example_files/50/walking_val_scale_50_matconvnet_m_2048_layer_20.txt")
    my_file = open("/local/wangxin/Data/ferrari_gaze/example_files/50/aeroplane_val_scale_50_matconvnet_m_2048_layer_20.txt")

    VOC_file = open("/local/wangxin/Data/VOCdevkit_trainset/VOC2012/ImageSets/Main/aeroplane_val.txt")

    positive_examples=[]
    my_file.readline()
    negative=0
    positive=0
    for line in my_file:
        line_elements = line.strip().split(' ')
        if line_elements[1]=='0':
            negative+=1
        else:
            positive+=1
            if line_elements[0].split('/')[-1][:-4][10:] in positive_examples:
                print line_elements[0].split('/')[-1][:-4][10:]
            positive_examples.append(line_elements[0].split('/')[-1][:-4][10:])
    print positive,negative
    
    voc_positive_examples={}
    for line in VOC_file:
        line_elements = line.strip().split(' ')
        if line_elements[-1]=='1':
            if line_elements[2] == '1':
                voc_positive_examples[line_elements[0]]=1
            else:
                voc_positive_examples[line_elements[0]]=0
    voc_positive_examples_list=[k for k,v in voc_positive_examples.items() if v!=0]
    
    print "my ex list:"
    print len(positive_examples),positive_examples
    
    print "voc ex list:"
    print len(voc_positive_examples_list),voc_positive_examples_list
    
    #check my_e in voc_ex
    print "check my_e in voc_ex"
    for my_e in positive_examples:
        if not my_e in voc_positive_examples_list:
            print my_e
    print "check voc_ex in_my_e"
     
    for voc_e in voc_positive_examples_list:
        if not voc_e in positive_examples:
            print voc_e
            
