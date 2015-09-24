root_path="/local/wangxin/data/VOCdevkit_testset/VOC2012/JPEGImages/"
target_path="/local/wangxin/data/gaze_voc_actions_stefan/action_test_images/"
while read -r file_name;
do 
	cp $root_path$file_name".jpg" $target_path$file_name".jpg"
done < action_test_image_list
