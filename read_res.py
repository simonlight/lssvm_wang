import collections

# std=open("/local/wangxin/Data/ferrari_data/res_std_new.txt")
# et=open("/local/wangxin/Data/ferrari_data/res_et_new.txt")

std=open("/home/wangxin/results/gaze_voc_actions_stefan/stdlssvm/C1e-3.txt")
et=open("/home/wangxin/results/gaze_voc_actions_stefan/std_et/Allgamma_1e-3C.txt")

def read_res(result_file):
	#Organize the dict like    category/lambda/scale/tradeoff/testap
	res= collections.defaultdict(lambda : collections.defaultdict(lambda : collections.defaultdict(lambda : collections.defaultdict(lambda : None))))

	for line in result_file:
		cls, tradeoff,scale,lbd,epsilon,aptest,aptrain = line.strip().split(' ')
		res[cls][lbd][scale][tradeoff]=float(aptest)
	return res

std_res = read_res(std)
et_res = read_res(et)

def read_dict(res):
	cv_res = collections.defaultdict(lambda :[0,None])
	for k_cls in res.keys():
		for k_lbd in res[k_cls].keys():
			for k_scale in res[k_cls][k_lbd].keys():
				for k_tradeoff in res[k_cls][k_lbd][k_scale].keys():
					v_aptest = res[k_cls][k_lbd][k_scale][k_tradeoff]
					if v_aptest > cv_res[k_cls][0]:
						cv_res[k_cls][0] = v_aptest
						cv_res[k_cls][1] = [k_lbd,k_scale,k_tradeoff]
	return cv_res

cv_std_res = read_dict(std_res)
cv_et_res = read_dict(et_res)

print cv_std_res
print cv_et_res
