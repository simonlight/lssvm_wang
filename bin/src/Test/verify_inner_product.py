import numpy as np

def normalize(v):
    norm=np.linalg.norm(v)
    if norm==0: 
       return v
    return v/norm

def get_w(w_path):
    w_pos_file=open(w_path)
    w_pos=w_pos_file.readline()
    w_pos_file.close()
    w_pos_dim = np.array([float(d) for d in w_pos.strip().split('\t')])
    return w_pos_dim

if __name__=="__main__":
    
    w_pos = get_w("/home/wangxin/w/w+test.txt")
    w_neg = get_w("/home/wangxin/w/w-test.txt")
    for i in range(6):
        for j in range(6):
            feature_path = open("/local/wangxin/Data/ferrari_gaze/matconvnet_m_2048_features/50/"+"2008_006448_"+str(i)+"_"+str(j)+".txt")
            feature = np.zeros(2049)
            for cnt,line in enumerate(feature_path):
                feature[cnt] = float(line.strip())
            feature = normalize(feature)
            feature[2048]=1
            print "%d_%d"%(i,j)
            print np.dot(w_pos, feature)
            print np.dot(w_neg, feature)
                
            print "****"