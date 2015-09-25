import collections


def statistic(p):
    res_dict= collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(lambda: None)))
    for line in open(p):        
        cls, gamma, scale, _, testap, trainap = line.strip().split(' ')
        res_dict[cls][scale][gamma]=[trainap, testap]
    print "class num:"+str(len(res_dict))+"\n"
    
    print "scale num:"+str(len(res_dict))+"\n"
    for (k,v) in res_dict.items():
        print k, len(res_dict[k])
        for (k2,v2) in res_dict[k].items():
            print "\t",k2, len(res_dict[k][k2])



def best_res(p):
    res_dict= collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(lambda: None)))
    for line in open(p):
        cls, gamma, scale, _, testap, trainap = line.strip().split(' ')
        res_dict[cls][scale][gamma]=[testap, trainap]
    print "class num:"+str(len(res_dict))+"\n"
    
    print "scale num:"+str(len(res_dict))+"\n"
    for (k,v) in res_dict.items():
#         for (k2,v2) in res_dict[k].items():
        for k2 in ["100","90","80","70","60","50"]:
#             print "\t",k2
            max_train=0
            max_test=0
            max_gamma=0
            for (k3,v3) in res_dict[k][k2].items():
                if res_dict[k][k2][k3][0]>max_test:
                    max_test = res_dict[k][k2][k3][0]
                    max_gamma=k3
#                     max_train = res_dict[k][k2][k3][1]
            print k, k2,max_gamma, max_test
#             print max_test
#             print k
            
if __name__=='__main__':
    std_lssvm = "/home/wangxin/results/gaze_voc_actions_stefan/stdlssvm/res_lssvm.txt"
    et_lssvm = "/home/wangxin/results/gaze_voc_actions_stefan/std_et/std_et_no_prediction.txt"
    #statistic(std_et_res_path)  
#     print "###############"
#     best_res(std_lssvm)
    best_res(et_lssvm)
    
#     best_res(neg_pos_res_path)  