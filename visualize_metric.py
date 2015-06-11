import collections
metric = open("/home/wang/metric.txt")
def cast_name(name):
    if "reduit_singlebb" in name:
        return name.replace("reduit_singlebb","std")
    elif "reduit_allbb" in name:
        return name.replace("reduit_allbb","eye")
    elif "ground" in name:
        return name.replace("ground","gt")

def read_res(data_dict):
    res = collections.defaultdict(lambda : collections.defaultdict(lambda : collections.defaultdict(lambda : None)))
    for line in data_dict:
        exp_type, scale, cls, iou = line.split(" ")
        res[scale][exp_type][cls]=iou
    return res

import numpy as np
classes=['cat', 'dog', 'bicycle', 'motorbike', 'boat', 'aeroplane', 'horse', 'cow', 'sofa', 'diningtable']
import matplotlib.pyplot as plt
import csv
import numpy as np
res = read_res(metric)
x = np.array(range(10))
my_xticks = classes
for scale in res:
    et=[0]*10
    gd=[0]*10
    std=[0]*10
    for exp_type in res[scale]:
        if exp_type == "ground":
            for cls in res[scale][exp_type]:
                gd[classes.index(cls)]=eval(res[scale][exp_type][cls].strip())
        if exp_type == "reduit_allbb":
            for cls in res[scale][exp_type]:
                et[classes.index(cls)]=eval(res[scale][exp_type][cls].strip())
        if exp_type == "reduit_singlebb":
            for cls in res[scale][exp_type]:
                std[classes.index(cls)]=eval(res[scale][exp_type][cls].strip())
    print gd
    print std
    plt.xticks(x, my_xticks, rotation=-30)
    plt.xlabel("class name")
    plt.ylabel("average IoU")
    plt.plot(x,gd, color='r', label= cast_name("ground"))
    plt.plot(x,et, color='g',label= cast_name("reduit_allbb"))
    plt.plot(x,std, color='b', label= cast_name("reduit_singlebb"))
    title="scale:"+str()
    plt.legend()
    plt.grid()
    plt.title("scale:"+scale+" avg(std)="+str(sum(std)/len(std))[:5]+" avg(et)="+str(sum(et)/len(et))[:5]+ " avg(gd)="+str(sum(gd)/len(gd))[:5])
    #plt.show()
    plt.savefig("/home/wang/"+scale+"_iou")
    plt.clf()
    
    with open('/home/wang/'+str(scale)+'.csv', 'wb') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        spamwriter.writerow(['scale='+scale,'mean']+classes)
        spamwriter.writerow(['std']+[str(sum(std)/len(std))[:5]]+[str(c)[:5] for c in std])
        spamwriter.writerow(['ground']+[str(sum(gd)/len(gd))[:5]]+[str(c)[:5] for c in gd])
        spamwriter.writerow(['et']+[str(sum(et)/len(et))[:5]]+[str(c)[:5] for c in et])
        
