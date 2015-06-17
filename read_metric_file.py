def get_best_cv(cls, scale):
    #cat 100 0.6 0.92
    best_res_root  = "/home/wangxin/Data/ferrari_data/reduit_allbb/best_cv_res/"
    method = "std_et_no_prediction.txt"
    f = open(best_res_root+method)
    for line in f:
        c, s, best_cv, _ = line.strip().split(' ')
        if c.strip()== cls and s.strip() == str(scale):
            return best_cv

root="/home/wangxin/Data/ferrari_data/reduit_allbb/metric_pos_only/results_mori/"
# classes=['cat', 'dog', 'bicycle', 'motorbike', 'boat', 'aeroplane', 'horse', 'cow', 'sofa', 'diningtable']
# classes=['cat', 'dog', 'boat', 'aeroplane', 'horse', 'cow', 'sofa', 'diningtable']
classes = ['bicycle', 'motorbike']
scales=[100,90,80,70,60,50]
for cls in classes:
    for scale in scales:        
        best_cv = get_best_cv(cls, scale)
        
        cnt = 0
        correct = 0.0
        false = 0.0
        f_metric= open(root+"metric_"+str(scale)+"_"+cls+"_"+str(best_cv)+"_pos_neg.txt")
        for line in f_metric:
            cnt+=1
            yp, h, image_path = line.strip().split(',')
            if yp == '1':
                correct += 1
            if yp == '0':
                false+=1
        print cls, scale, cnt, correct, correct/cnt
        f_metric.close()