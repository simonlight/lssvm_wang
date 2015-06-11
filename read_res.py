import collections

std=open("/local/wangxin/Data/ferrari_data/res_std_new.txt")
et=open("/local/wangxin/Data/ferrari_data/res_et_new.txt")

res_std = collections.defaultdict(lambda : collections.defaultdict(lambda : None))
res_et=collections.defaultdict(lambda : collections.defaultdict(lambda : None))

for line in std:
	cls, scale,_,acc,_,ap=line.split()
	res_std[cls][scale.split(":")[1]]=ap.split(":")[1]
	#res_std[cls][scale.split(":")[1]]=acc.split(":")[1]

for line in et:
	cls, scale,_,acc,_,ap=line.split()
	res_et[cls][scale.split(":")[1]]=ap.split(":")[1]
	#res_et[cls][scale.split(":")[1]]=acc.split(":")[1]
stdsum=0
"""for k,v in res_std.items():
	print k,
	v_max=0
	k_max=0
	for k_,v_ in v.items():
		if v_>v_max:
			v_max=v_
			k_max=k_
	print k_max,float(v_max)
	stdsum+=float(v_max)

print '\n'
etsum=0
for k,v in res_et.items():
	print k,
	v_max=0
	k_max=0
	for k_,v_ in v.items():
		if v_>v_max:
			v_max=v_
			k_max=k_
	print k_max,v_max
	etsum+=float(v_max)"""
	

	
etsum=0
for k,v in res_std.items():
	print k+'\t',
	class_avg=0.0
	v_max=0
	k_max=0
	min_c=10000
	max_c=0
	c=0
	for k_,v_ in v.items():
		c+=1
		etsum+=float(v_)
		class_avg+=float(v_)
		if float(v_)< min_c:
			min_c =float(v_)
		if float(v_)> max_c:
			max_c =float(v_)
			
	print  str(class_avg/c)+'\t',
	print str(max_c)+'\t',
	print str(min_c)+'\n',
print stdsum/61
print etsum/61