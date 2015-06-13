root="/home/wangxin/Data/ferrari_data/reduit_allbb/results_neg_pos_no_prediction/"
# classes=['cat', 'dog', 'bicycle', 'motorbike', 'boat', 'aeroplane', 'horse', 'cow', 'sofa', 'diningtable']
classes=['cat', 'dog', 'boat', 'aeroplane', 'horse', 'cow', 'sofa', 'diningtable']
#classes = ['boat']
scales=[100,90,80,70,60,50]
for cls in classes:
    
    for scale in scales:        
        best_cv = get_best_cv(cls, scale)
        f= open(root+"metric_"+str(scale)+"_"+cls+"_"+str(best_cv)+"_pos_neg.txt")
        cnt = 0
        correct = 0.0
        for line in f:
            cnt+=1
            yp, h, image_path = line.strip().split(',')
            if yp == '1':
                correct += 1
        print correct/cnt