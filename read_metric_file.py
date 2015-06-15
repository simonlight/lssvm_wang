def get_best_cv(cls, scale):
    #cat 100 0.6 0.92
    best_res_root  = "/home/wangxin/Data/ferrari_data/reduit_allbb/best_cv_res/"
    method = "res_neg_pos_new_prediction.txt"
    f = open(best_res_root+method)
    for line in f:
        c, s, best_cv, _ = line.strip().split(' ')
        if c.strip()== cls and s.strip() == str(scale):
            return best_cv

root="/home/wangxin/Data/ferrari_data/reduit_allbb/results_neg_pos_new_prediction/"
# classes=['cat', 'dog', 'bicycle', 'motorbike', 'boat', 'aeroplane', 'horse', 'cow', 'sofa', 'diningtable']
classes=['cat', 'dog', 'boat', 'aeroplane', 'horse', 'cow', 'sofa', 'diningtable']
#classes = ['boat']
scales=[100,90,80,70,60,50]
for cls in classes:
    for scale in scales:        
        best_cv = get_best_cv(cls, scale)
        f_metric= open(root+"metric_"+str(scale)+"_"+cls+"_"+str(best_cv)+"_pos_neg.txt")
        total_metric_line = 0
        for l in f_metric:
            total_metric_line+=1
        f_metric.close()
        
        f_val = open("/home/wangxin/Data/ferrari_data/POETdataset/POETdataset/example_files_pos_val/"+str(scale)+"/"+cls+"_val_scale_"+str(scale)+"_matconvnet_m_2048_layer_20.txt")
        total_val_lines = 0
        for l in f_val:
            total_val_lines+=1
        f_val.close()
        
        offset = total_metric_line-total_val_lines
        
        cnt = 0
        correct = 0.0
        f_metric= open(root+"metric_"+str(scale)+"_"+cls+"_"+str(best_cv)+"_pos_neg.txt")
        for i in range(offset):
            f_metric.readline()
        for line in f_metric:
            cnt+=1
            yp, h, image_path = line.strip().split(',')
            if yp == '1':
                correct += 1
        print cls, scale, cnt, correct/cnt