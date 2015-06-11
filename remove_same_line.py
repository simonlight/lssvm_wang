#If all 0 the accuracy?
filename="/local/wangxin/Data/ferrari_data/res_std.txt"
new_filename="/local/wangxin/Data/ferrari_data/res_std_new.txt"
f_new=open(new_filename,'w')
f=open(filename)
deja_vu=[]
for line in f:
	if line not in deja_vu:
		f_new.write(line)
		deja_vu.append(line)
	

f.close()
f_new.close()