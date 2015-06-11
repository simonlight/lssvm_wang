#If all 0 the accuracy?
filename = "/media/workspace/work_data/ferrari_data/example_files/100/aeroplane_val_scale_100_matconvnet_m_2048_layer_20.txt"
f= open(filename)
f.readline()
c=0.0
for line in f:
	_, yes,_,_ = line.split();
	c+=int(yes)
print c/3092
