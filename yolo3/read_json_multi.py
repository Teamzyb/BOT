import cv2
import json

# 单张图片可以read_json.py
data = "../BOT_project/data/train/image"


# image = cv2.imread("test.jpg")
#
# with open("test.json",'r') as load_f:
#     load_dict = json.load(load_f)
#     object = load_dict["annotation"][0]["object"]
#     print(object)
#     print(len(object))
#     for i in range(3):
#         for o in object:
#             minx = o['minx']
#             miny = o['miny']
#             maxx = o['maxx']
#             maxy = o['maxy']
#             gender = o['gender']
#             staff = o['staff']
#             customer = o['customer']
#             stand = o['stand']
#             sit = o['sit']
#             play_with_phone = o['play_with_phone']
#             cv2.rectangle(image, (minx, miny), (maxx, maxy), (255, 0,
#                                                               0), 2)
#             cv2.putText(image, "gender: " + str(gender) + ", " +
#                         "staff: " + str(staff) + ", " + "customer: " + str(customer) + ", " +
#                         "play_with_phone: " + str(play_with_phone),
#                         (minx, miny), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
#     cv2.imshow("frame", image)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()


