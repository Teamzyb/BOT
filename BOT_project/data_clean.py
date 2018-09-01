# -*- coding: UTF-8 -*-
import os
import shutil

def make_testdata(train_dir,test_dir):
	if not os.path.isdir(test_dir):
		os.makedirs(test_dir)

	train_image = os.path.join(train_dir,'image')
	test_image = os.path.join(test_dir,'image')

	for i , train_image_sub in enumerate(os.listdir(train_image)):
		if i%10 == 0 :
			print('准备移动文件')
			image_dir = os.path.join(train_image,train_image_sub)
			new_image_dir = os.path.join(test_image,train_image_sub)
			print(image_dir)
			shutil.move(image_dir,new_image_dir)
			print('移动文件完成')
		print(train_image_sub,i)

	print('==================================')

	train_label = os.path.join(train_dir,'label')
	test_label = os.path.join(test_dir,'label')

	for i , train_label_sub in enumerate(os.listdir(train_label)):
		if i%10 == 0 :
			print('准备移动文件')
			label_dir = os.path.join(train_label,train_label_sub)
			new_label_dir = os.path.join(test_label,train_label_sub)
			print(label_dir)
			shutil.move(label_dir,new_label_dir)
		print(train_label_sub,i)



if __name__ == '__main__':
	make_testdata('./data/train','./data/test')