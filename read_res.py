import collections

# std=open("/local/wangxin/Data/ferrari_data/res_std_new.txt")
# et=open("/local/wangxin/Data/ferrari_data/res_et_new.txt")

std=open("/home/wangxin/results/gaze_voc_actions_stefan/stdlssvm/res_lssvm.txt")
et=open("/home/wangxin/results/gaze_voc_actions_stefan/std_et/std_et.txt")

def read_res(result_file):
	#Organize the dict like    category/lambda/scale/tradeoff/testap
	res= collections.defaultdict(lambda : collections.defaultdict(lambda : collections.defaultdict(lambda : collections.defaultdict(lambda : None))))

	for line in result_file:
		cls, tradeoff,scale,lbd,epsilon,aptest,aptrain = line.strip().split(' ')
		res[cls][lbd][scale][tradeoff]=aptest
	return res

std_res = read_res(std)
et_res = read_res(et)

def read_dict(res):
	cv_res = collections.defaultdict(lambda :[0,None])
	for k_cls in res.keys():
		for k_lbd in res[k_cls].keys():
			for k_scale in res[k_cls][k_lbd].keys():
				for k_tradeoff in res[k_cls][k_lbd][k_scale].keys():
					for v_aptest in res[k_cls][k_lbd][k_scale][k_tradeoff]:
						if v_aptest > cv_res[k_cls][0]:
							cv_res[k_cls][0] = v_aptest
							cv_res[k_cls][1] = [k_cls,k_lbd,k_scale,k_tradeoff]

cv_std_res = read_dict(std_res)
print cv_std_res
# stdsum=0
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
	etsum+=float(v_max)
"""
	

	
# etsum=0
# for k,v in res_std.items():
# 	print k+'\t',
# 	class_avg=0.0
# 	v_max=0
# 	k_max=0
# 	min_c=10000
# 	max_c=0
# 	c=0
# 	for k_,v_ in v.items():
# 		c+=1
# 		etsum+=float(v_)
# 		class_avg+=float(v_)
# 		if float(v_)< min_c:
# 			min_c =float(v_)
# 		if float(v_)> max_c:
# 			max_c =float(v_)
# 			
# 	print  str(class_avg/c)+'\t',
# 	print str(max_c)+'\t',
# 	print str(min_c)+'\n',
# print stdsum/61
# print etsum/61