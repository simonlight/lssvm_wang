def res_file_2_dict(ap_results):
    res = collections.defaultdict(lambda : collections.defaultdict(lambda : collections.defaultdict(lambda : None)))
    for line in ap_results:
        category, tradeoff, scale, lbd, epsilon, ap_train, ap_val, ap_test = [i.split(":")[1] for i in line.strip().split()]
        res[scale][tradeoff][category]= [float(ap_train), float(ap_val), float(ap_test)]
    return res
    

def plot_res(res, res_typ):
#     for scale in ap_res.keys():
    for scale in ['50']:

        result_name = scale
        x_axis = res[scale].keys()
        y_train = [0]*11
        y_val = [0]*11
        y_test = [0]*11
        for tradeoff in np.arange(0,1.1,0.1):
            y_ap_all = res[scale][str(tradeoff)]
#             print tradeoff, y_ap_all
#             print scale, tradeoff, np.sum(y_ap_all.values(), axis=0) / len(y_ap_all.values())
            print np.sum(y_ap_all.values(), axis=0)
            ap_train, ap_val, ap_test = np.sum(y_ap_all.values(), axis=0) / len(y_ap_all.values())
            y_train[int(tradeoff*10)] = ap_train
            y_val[int(tradeoff*10)] = ap_val
            y_test[int(tradeoff*10)] = ap_test
        
        x = np.arange(0,1.1,0.1)
        
        plt.figure(figsize=(8,4))
        plt.plot(x,y_train,label="train "+res_typ,color="red",linewidth=2)
        plt.plot(x,y_val,label="validation "+res_typ,color="blue",linewidth=2)
        plt.plot(x,y_test,label="test "+res_typ,color="green",linewidth=2)
        plt.xlabel("Tradeoff")
        plt.ylabel(res_typ)
        plt.title(res_typ+" of scale:%s"%scale)
        plt.ylim(min(min(y_train), min(y_val),min(y_test)),max(max(y_train), max(y_val),max(y_test)))
        plt.axvline(x=y_val.index(max(y_val))/10.0, color= 'black', linestyle='dashed')
        plt.legend(loc='best',fancybox=True,framealpha=0.5)
        plt.show()

if __name__ == "__main__":
    
    import collections
    import matplotlib.pyplot as plt
    import csv
    import numpy as np
    import os
    import visualize_fixations_ferrari as vff
#     ap_results = open("/local/wangxin/results/ferrari_gaze/std_et/java_std_et/ap_summary.txt")
#     ap_res = res_file_2_dict(ap_results)
#     plot_res(ap_res, "AP")
    
    detection_folder = "/local/wangxin/results/ferrari_gaze/std_et/java_std_et/metric/"
    detection_res, gr_res = vff.metric_file_analyse(detection_folder)
    print detection_res
    plot_res(detection_res, "detection")
    plot_res(gr_res, "gaze ratio")

        
            
#     classes=['cat', 'dog', 'bicycle', 'motorbike', 'boat', 'aeroplane', 'horse', 'cow', 'sofa', 'diningtable']
#     my_xticks = classes
#     for scale in res:
#         et=[0]*10
#         gd=[0]*10
#         std=[0]*10
#         for exp_type in res[scale]:
#             if exp_type == "ground":
#                 for cls in res[scale][exp_type]:
#                     gd[classes.index(cls)]=eval(res[scale][exp_type][cls].strip())
#             if exp_type == "reduit_allbb":
#                 for cls in res[scale][exp_type]:
#                     et[classes.index(cls)]=eval(res[scale][exp_type][cls].strip())
#             if exp_type == "reduit_singlebb":
#                 for cls in res[scale][exp_type]:
#                     std[classes.index(cls)]=eval(res[scale][exp_type][cls].strip())
#         print gd
#         print std
#         plt.xticks(x, my_xticks, rotation=-30)
#         plt.xlabel("class name")
#         plt.ylabel("average IoU")
#         plt.plot(x,gd, color='r', label= cast_name("ground"))
#         plt.plot(x,et, color='g',label= cast_name("reduit_allbb"))
#         plt.plot(x,std, color='b', label= cast_name("reduit_singlebb"))
#         title="scale:"+str()
#         plt.legend()
#         plt.grid()
#         plt.title("scale:"+scale+" avg(std)="+str(sum(std)/len(std))[:5]+" avg(et)="+str(sum(et)/len(et))[:5]+ " avg(gd)="+str(sum(gd)/len(gd))[:5])
#         #plt.show()
#         plt.savefig("/home/wang/"+scale+"_iou")
#         plt.clf()
#         
#         with open('/home/wang/'+str(scale)+'.csv', 'wb') as csvfile:
#             spamwriter = csv.writer(csvfile, delimiter=',',
#                                     quotechar='|', quoting=csv.QUOTE_MINIMAL)
#             spamwriter.writerow(['scale='+scale,'mean']+classes)
#             spamwriter.writerow(['std']+[str(sum(std)/len(std))[:5]]+[str(c)[:5] for c in std])
#             spamwriter.writerow(['ground']+[str(sum(gd)/len(gd))[:5]]+[str(c)[:5] for c in gd])
#             spamwriter.writerow(['et']+[str(sum(et)/len(et))[:5]]+[str(c)[:5] for c in et])
#             
