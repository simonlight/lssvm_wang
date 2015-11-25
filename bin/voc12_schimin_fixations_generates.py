  
def single_act_image(action_train_listpath, train_images, annotation_path, single_action_path):
    f = open (action_train_listpath)
    f2 = open(single_action_path,'w')
    several_action=0
    multi_action = 0
    single_action = 0
    single_action_stat=[0]*10
    for image_name in f:
        xmldoc = minidom.parse(annotation_path+image_name.strip()+'.xml')
        itemlist = xmldoc.getElementsByTagName("actions")
        if len(itemlist) > 1:
            several_action +=1
            continue
        action_number= 0
        for action in enumerate(action_names):
            if int(itemlist[0].getElementsByTagName(action[1])[0].childNodes[0].nodeValue) !=0:
                action_number += int(itemlist[0].getElementsByTagName(action[1])[0].childNodes[0].nodeValue)
                last_action = action[1]
        if action_number == 1:
            single_action+=1
            single_action_stat[action_names.index(last_action)]+=1
            f2.write(image_name.strip()+'\n')

        else:
            multi_action+=1
    print "In total %d, %d images have several actions, %d images have multi-actions, %d images have single action"%(several_action+multi_action+single_action,several_action, multi_action, single_action)
    print """In single action, we have :\n 
jumping \t phoning \t playinginstrument \t reading \t ridingbike \t ridinghorse \t running \t takingphoto \t usingcomputer \t walking \n
    """
    
    for i in single_action_stat:
        print "%d \t"%i,
    print "\n In total, single actions:%d"%sum(single_action_stat)

if __name__ == "__main__":
    """
    data format:
     - <timestamp> represents the time stamp of the gaze sample (microseconds)
     - <pupil_diameter> and <pupil_area> specify the measured diameter and area, respectivey, of the detected pupil (pixels and squared
       pixels, respectively)
     - <x_screen> <y_screen> specifies the gaze coordinates (point of regard) in screen space
     - <event>  specifies the type of eye movement event assigned by SMI's detection algorithm to this sample, with 'F' meaning fixation, 
       'S' meaning saccade and 'B' meaning blink.
    timestamp    pupil_diameter    pupil_area    x_screen    y_screen    event
    2992518        36.62            1053        469.67        367.37        F
    2994516        36.58            1051        469.18        366.81        F
    2996519        36.64            1054.5        468.35        367.12        F
    2998518        36.64            1054.5        468.47        366.65        F"""


    """        <jumping>0</jumping>
                <other>0</other>
                <phoning>0</phoning>
                <playinginstrument>0</playinginstrument>
                <reading>0</reading>
                <ridingbike>1</ridingbike>
                <ridinghorse>0</ridinghorse>
                <running>0</running>
                <takingphoto>0</takingphoto>
                <usingcomputer>0</usingcomputer>
                <walking>0</walking>
    """
    
    from xml.dom import minidom
#     fixation_path = "/local/wangxin/Data/gaze_voc_actions_stefan/samples"
    pascal_voc_2012_action_train_listpath = "/local/wangxin/Data/VOCdevkit_trainset/VOC2012/ImageSets/Action/trainval.txt"
    pascal_voc_2012_train_images = "/local/wangxin/Data/VOCdevkit_trainset/VOC2012/JPEGImages/"
    pascal_voc_2012_annotations = "/local/wangxin/Data/VOCdevkit_trainset/VOC2012/Annotations/"
    action_names=["jumping", "phoning", "playinginstrument", "reading", "ridingbike", "ridinghorse", "running", "takingphoto", "usingcomputer", "walking"]
    single_action_path= "/local/wangxin/Data/gaze_voc_actions_stefan/action_train_image_list"
    
#     single_act_image(pascal_voc_2012_action_train_listpath, pascal_voc_2012_train_images, pascal_voc_2012_annotations, single_action_path)
