
def valid_ferrari_image(ferrari_no_duplicate_image, typ):
	feature_root = "/home/wangxin/Data/ferrari_gaze/matconvnet_m_2048_features/" 
	for scale in [100,90,80,70,60,50,40,30]:
		for category in path_config.VOC2012_OBJECT_CATEGORIES:
			category_trainval_images = []
			train_num = 3039
			val_num = 3092
			trainval_num = 6131
			
			#e.g.: This image 2008_003607 contains cat sofa, and dog. In ferrari's dataset, cat and dog can't appear in the same time, so this image is annotated 
			#with sofa's gazes, while no gazes are associated with dog and cat. When dog is positive, it's impossible to get the 
			#gaze ratio of dog, so it's not consistent with our method.  

			if category in ["dog", "cat"] :
				category_trainval_images = [im for im in ferrari_no_duplicate_image if not im in ["2008_003607", "2010_005614", "2009_004073", "2008_005882"]] 
				train_num -= 3
				val_num -= 1
				trainval_num -= 4
			elif category in ["bicycle", "motorbike"] :
				category_trainval_images = [im for im in ferrari_no_duplicate_image if not im in ["2008_002948"]]
				train_num -= 1
				val_num -= 0
				trainval_num -= 1
			elif category in ["sofa", "diningtable"] :
				category_trainval_images = [im for im in ferrari_no_duplicate_image if not im in ["2010_003822"]]
				train_num -=1
				val_num -= 0
				trainval_num -= 1
			else:
				category_trainval_images = ferrari_no_duplicate_image
			
			example_file_folder = "/local/wangxin/Data/ferrari_gaze/example_files/"+str(scale)+'/'
			example_file_path = example_file_folder+category+'_'+typ+'_scale_'+str(scale)+'_matconvnet_m_2048_layer_20.txt'
			if not os.path.exists(example_file_folder):
				os.mkdir(example_file_folder)
			example_file=open(example_file_path,'w')
			if typ=="train":
				example_file.write(str(train_num)+"\n")
			elif typ=="val":
				example_file.write(str(val_num)+"\n")
			elif typ=="trainval":
				example_file.write(str(trainval_num)+"\n")
				
			voc_file_path = path_config.VOC2012_TRAIN_LIST + "Main/"+category + '_'+typ+'.txt'
			voc_file = open(voc_file_path)
			c=0
			for line in voc_file:
				file_piece= line.strip().split(' ')
				file_name_root = file_piece[0]
				label =  file_piece[-1]
				if label =='-1':
					label='0'
				if file_name_root in category_trainval_images:
					c+=1
					if scale == 100:
						example_line = ' '.join([file_name_root, label, str(metric_calculate.convert_scale(scale)), feature_root+str(scale)+'/'+file_name_root+'.txt'])
				 		example_file.write(example_line+'\n')
					else:
						feature_path=[]
						for grid_1 in range(int(math.sqrt(metric_calculate.convert_scale(scale)))):
							for grid_2 in range(int(math.sqrt(metric_calculate.convert_scale(scale)))):
								feature_path.append(feature_root+\
												str(scale)+"/"+file_name_root+'_'+str(grid_1)+'_'+str(grid_2)+'.txt')
				 		example_line = [file_name_root, label, str(metric_calculate.convert_scale(scale))]
				 		example_line.extend(feature_path)
				 		example_file.write(' '.join(example_line)+'\n')
			print c
			example_file.close()
			voc_file.close()


def image_verification():
	no_duplicate_image_names = path_config.VOC2012_OBJECT_ROOT+"matconvnet_m_2048_features/100/"				
	duplicate_image_names = path_config.VOC2012_OBJECT_ROOT+"gazes/"
	ferrari_no_duplicate_image= []
	ferrari_duplicate_image = []
	for root, dirs, files in os.walk(duplicate_image_names):
		for file in files:
			ferrari_duplicate_image.append(file[:-4])
	
	for root, dirs, files in os.walk(no_duplicate_image_names):
		for file in files:
			ferrari_no_duplicate_image.append(file[:-4])
	miss_train=[]
	miss_val=[]
	for category in path_config.VOC2012_OBJECT_CATEGORIES:
		VOC_train_list = "/local/wangxin/Data/VOCdevkit_trainset/VOC2012/ImageSets/Main/"+category+"_train.txt"
		VOC_val_list = "/local/wangxin/Data/VOCdevkit_trainset/VOC2012/ImageSets/Main/"+category+"_val.txt"
		voc_positive_train = []
		voc_positive_val = []
		
		for line in open(VOC_train_list):
			file_piece = line.strip().split(' ')
			file_name = file_piece[0]
			label = file_piece[-1]
			if label == '1' and file_name in ferrari_no_duplicate_image:
				voc_positive_train.append(file_name)
		for line in open(VOC_val_list):
			file_piece = line.strip().split(' ')
			file_name = file_piece[0]
			label = file_piece[-1]
			if label == '1' and file_name in ferrari_no_duplicate_image:
				voc_positive_val.append(file_name)
		for vpr in voc_positive_train:
			if not '_'.join([category,vpr]) in ferrari_duplicate_image:
				miss_train.append(vpr)
		for vpr in voc_positive_val:
			if not '_'.join([category,vpr]) in ferrari_duplicate_image:
				miss_val.append(vpr)
	print "train:%d "%len(set(miss_train)) + str(set(miss_train))
	print "val:%d "%len(set(miss_val))+ str(set(miss_val))
if __name__=="__main__":
	import path_config
	import metric_calculate
	import os
	import math
	no_duplicate_image_names = path_config.VOC2012_OBJECT_ROOT+"matconvnet_m_2048_features/100/"				
	ferrari_no_duplicate_image = []
	for root, dirs, files in os.walk(no_duplicate_image_names):
		for file in files:
			ferrari_no_duplicate_image.append(file[:-4])
	valid_ferrari_image(ferrari_no_duplicate_image, "trainval")
# 	valid_ferrari_image(ferrari_no_duplicate_image, "val")
# 	valid_ferrari_image(ferrari_no_duplicate_image, "train")
	
				
			