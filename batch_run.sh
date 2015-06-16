#cls_arr=("dog" "cat" "motorbike" "boat" "aeroplane" "horse" "cow" "sofa" "diningtable")

#cls_arr=("dog" "cat" "bicycle" "motorbike" "boat")
cls_arr=("aeroplane" "horse" "cow" "sofa" "diningtable")
#cls_arr=("horse" "cow" "sofa" "diningtable" "aeroplane")
#scale_arr=("100" "90" "80")
#cls_arr=("motorbike" "boat")
#cls_arr=("dog" "cat")
#scale_arr=("50")
#scale_arr=("80" "70" "60" "50")
scale_arr=("100" "90" "80" "70" "60" "50")
#cls_arr=("bicycle")
#scale_arr=("100")
k='oarsub -p "host='"'"'big15'"'"' " -l "nodes=1/core=6,walltime=500:0:0" --notify "mail:xin.wang@lip6.fr" "/home/wangxin/lib/jdk1.8.0_25/bin/java -classpath /home/wangxin/mosek/7/tools/platform/linux64x86/bin/mosek.jar:/home/wangxin/lib/commons-cli-1.2.jar:/home/wangxin/lib/jkernelmachines.jar:/home/wangxin/code/lssvm_wang/src:. data/uiuc/mac/LSSVMMulticlassTestET'
end='"'
space=' '
for scale in ${scale_arr[@]}
do
	for cls in ${cls_arr[@]}
    do
        eval $k$space$cls$space$scale$end
    done
done
