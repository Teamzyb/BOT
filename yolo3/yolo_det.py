import cv2
from camera import *

'''
给图片加框和名字
'''
def Image_rectangle(img,c1,c2,name):
	c1 = tuple(c1.int())
	c2 = tuple(c2.int())
	color = (0, 0, 255)
	cv2.rectangle(img, c1, c2, color,1)    # 加框
	t_size = cv2.getTextSize(str(name), cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
	c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
	cv2.rectangle(img, c1, c2, color, -1) # -1填充作为文字框底色
	cv2.putText(img, str(name), (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225, 255, 255], 1)
	return img

if __name__ == '__main__':
	model_yolo3 = Init_yolo("yolov3.weights","cfg/yolov3.cfg")
	img = cv2.imread('./test.jpg')
	outputs, haveperson = yolo3(model_yolo3, img, biggest_per=False)

	if haveperson:
		for i,output in enumerate(outputs):
			print(i,output)
			img = Image_rectangle(img,output[1:3],output[3:5],str(i))
	else :
		print("Yolo3 can not detect person")

	cv2.imwrite('bot4-result.jpg',img)
	cv2.imshow('result',img)
	cv2.waitKey(0)

