#cls_arr=("dog" "cat" "motorbike" "boat" "aeroplane" "horse" "cow" "sofa" "diningtable" "bicycle")
#cls_arr=("dog" "cat" "motorbike" "boat" "aeroplane")
#cls_arr=("cow" "sofa" "diningtable" "bicycle")
#cls_arr=("jumping" "phoning" "playinginstrument" "reading" "ridingbike" "ridinghorse" "running" "takingphoto" "usingcomputer" "walking")
#cls_arr=("takingphoto" "usingcomputer" "walking")
#scale_arr=("90" "80" "70" "60" "50" "40" "30")
#cls_arr=("phoning")
cls_arr=("dog")

scale_arr=("90")
#k='oarsub -p "host='"'"'big17'"'"' or host='"'"'big18'"'"' or host='"'"'big19'"'"' or host='"'"'big20'"'"' " -l "nodes=1/core=1,walltime=500:0:0" --notify "mail:xin.wang@lip6.fr" "/home/wangxin/lib/jdk1.8.0_25/bin/java -classpath /home/wangxin/mosek/7/tools/platform/linux64x86/bin/mosek.jar:/home/wangxin/lib/commons-cli-1.2.jar:/home/wangxin/lib/jkernelmachines.jar:/home/wangxin/code/lssvm_wang/src:. 
#data/uiuc/mac/LSSVMMulticlassTestET'

k='oarsub -p "host='"'"'big20'"'"' " -l "nodes=1/core=1,walltime=500:0:0" --notify "mail:xin.wang@lip6.fr" "/home/wangxin/lib/jdk1.8.0_25/bin/java -classpath /home/wangxin/mosek/7/tools/platform/linux64x86/bin/mosek.jar:/home/wangxin/lib/commons-cli-1.2.jar:/home/wangxin/lib/jkernelmachines.jar:/home/wangxin/code/lssvm_wang/src:. 
data/uiuc/mac/LSSVMMulticlassTestET'

#k='oarsub -p "host='"'"'big$1'"'"' " -l "nodes=1/core=1,walltime=500:0:0" --notify "mail:xin.wang@lip6.fr" "/home/wangxin/lib/jdk1.8.0_25/bin/java -classpath /home/wangxin/mosek/7/tools/platform/linux64x86/bin/mosek.jar:/home/wangxin/lib/commons-cli-1.2.jar:/home/wangxin/lib/jkernelmachines.jar:/home/wangxin/code/lssvm_wang/src:. 
#data/uiuc/mac/LSSVMMulticlassTestET'


end='"'
space=' '
for scale in ${scale_arr[@]}
do
	for cls in ${cls_arr[@]}
    do
        eval $k$space$cls$space$scale$end
    done
done
