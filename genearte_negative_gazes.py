#gaze points for negative class
import os
path = "/home/wangxin/Data/ferrari_data/reduit_allbb/gazes/"
classes=['cat', 'dog', 'bicycle', 'motorbike', 'boat', 'aeroplane', 'horse', 'cow', 'sofa', 'diningtable']
#classes=['cat','dog']
#path = "/home/wangxin/Dropbox/app/lssvm_wang/src/latent/lssvm/multiclass"

cnt=0
for root, _, files in os.walk(path):
	for iter_cls in classes:
		for f_name in files:
			cls,year,number = f_name.split("_")
			root_index = '_'.join([year,number])
			for c in classes:
				if c!=iter_cls and os.path.exists(path+'_'.join([c,root_index])):
					f = open('/home/wangxin/Data/ferrari_data/reduit_allbb/gaze_negative/'+iter_cls+'/'+f_name,'a')
					f_negative = open(path+'_'.join([c,root_index]),'r')
					for line in f_negative:
						f.write(line)
					f.close()
					f_negative.close()
			if not os.path.exists('/home/wangxin/Data/ferrari_data/reduit_allbb/gaze_negative/'+iter_cls+'/'+f_name):
				cnt+=1
				f = open('/home/wangxin/Data/ferrari_data/reduit_allbb/gaze_negative/'+iter_cls+'/'+f_name,'a')
				f.write('-1,-1')
				f.close()
print cnt
				
