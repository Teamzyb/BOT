# -*- coding: utf-8 -*-
from util import *
from darknet import Darknet
from preprocess import   letterbox_image
# import models
# import data_manager
# import transforms as T
# from dataset_loader import Load_person
# from dataset_loader import ImageDataset
from torch.utils.data import DataLoader
from PIL import Image
import  os
import torch.backends.cudnn as cudnn
# import multiprocessing as mp

color_dict = {'Noface':(144, 128, 112),   # 新客灰色
              'Unknow': (0, 0, 0),      # 未知黑色
              '0001': (0, 0, 255),      # 红色
              '0002': (60, 20, 220),  # 猩红
              '0003': (84, 46, 8),    # 藏青色
              '0004': (204, 209, 72),
              '0005': (204, 209, 72),  # 蓝绿色
              'rNoface': (144, 128, 112),  # 新客灰色
              'rUnknow': (0, 0, 0),  # 未知黑色
              'r0001': (0, 0, 255),  # 红色
              'r0002': (60, 20, 220),  # 猩红
              'r0003': (84, 46, 8),  # 湛青色
              'r0004': (204, 209, 72),
              'r0005': (204, 209, 72),
              'tNoface': (144, 128, 112),  # 新客灰色
              'tUnknow': (0, 0, 0),  # 未知黑色
              't0001': (0, 0, 255),  # 红色
              't0002': (60, 20, 220),  # 猩红
              't0003': (84, 46, 8),  # 湛青色
              't0004': (204, 209, 72),
              't0005': (204, 209, 72),
              '6': (255, 0, 0),  #
              }
# source = "rtsp://admin:ncslab666@192.168.1.18/Streaming/Channels/1"
# class VideoCamera(object):
#     def __init__(self,num):
#         self.video = cv2.VideoCapture(0)
#     def __del__(self):
#         self.video.release()
#     def get_frame(self):
#         success, image = self.video.read()
#         ret, jpeg = cv2.imencode('.jpg', image)
#         return jpeg.tobytes()

'''
加载yolo3模型权重
加载reid（resnet50m-market1501）模型权重
'''
def Init_model(yolo_dir,yolo_cfg,reid_dir,reid_cfg):
    CUDA = torch.cuda.is_available()
    print("Loading Yolo3 network.....")
    model_yolo3 = Darknet(yolo_cfg)
    model_yolo3.load_weights(yolo_dir)
    print("Yolo3 Network successfully loaded")
    print("Initializing model: {}".format(reid_cfg))
    model_reid = models.init_model(name=reid_cfg, num_classes=751, loss={'xent'}, use_gpu=CUDA)
    if reid_dir:
        print("Loading checkpoint from '{}'".format(reid_dir))
        checkpoint = torch.load(reid_dir)
        model_reid.load_state_dict(checkpoint['state_dict'])
        print("Reid Network successfully loaded")
    if CUDA:
        model_reid = nn.DataParallel(model_reid).cuda()
        model_yolo3.cuda()
    return model_yolo3,model_reid

def Init_yolo(yolo_dir,yolo_cfg):
    CUDA = torch.cuda.is_available()
    print("Loading Yolo3 network.....")
    model_yolo3 = Darknet(yolo_cfg)
    model_yolo3.load_weights(yolo_dir)
    print("Yolo3 Network successfully loaded")
    if CUDA:
        model_yolo3.cuda()
    return model_yolo3

'''
保存裁剪的图片到对应人的名称文件夹之下
'''
def save_pic(detper,perlib,name='0001',camera_id= 'c1',num= 0):
    perlib = os.path.join(perlib, '%s' % str(name))
    if not os.path.isdir(perlib):  # Create the log directory if it doesn't exist
        os.makedirs(perlib)
    pic_outfile = os.path.join(perlib, '%s_%s_0%s.jpg' % (str(name), str(camera_id), str(num)))
    cv2.imwrite(pic_outfile, detper)
    return 0

def save_feature(gf,name,gfs,g_pid):
	gfs.append(gf)
	g_pid.extend([int(name)])
	gfs_cat = torch.cat(gfs, 0)
	g_pid_arr = np.asarray(g_pid)
	return gfs, g_pid, gfs_cat, g_pid_arr


'''
裁剪行人，返回视野中最大的行人的坐标
'''
def yolo3(model_yolo3,orig_im,confidence=0.5,reso =416,nms_thresh = 0.4, biggest_per = True):
    model_yolo3.eval()
    CUDA = torch.cuda.is_available()
    model_yolo3.net_info["height"] = reso
    inp_dim = int(model_yolo3.net_info["height"])
    assert inp_dim % 32 == 0
    assert inp_dim > 32
    img, orig_im, dim = prep_image(orig_im, inp_dim)
    im_dim = torch.FloatTensor(dim).repeat(1,2)
    if CUDA:
        im_dim = im_dim.cuda()
        img = img.cuda()
    with torch.no_grad():
        output = model_yolo3(Variable(img), CUDA)
    output = write_results(output, float(confidence), 80, nms = True, nms_conf = float(nms_thresh))
    if type(output)==int:return output,False
    im_dim = im_dim.repeat(output.size(0), 1)
    scaling_factor = torch.min(inp_dim/im_dim,1)[0].view(-1,1)

    output[:,[1,3]] -= (inp_dim - scaling_factor*im_dim[:,0].view(-1,1))/2
    output[:,[2,4]] -= (inp_dim - scaling_factor*im_dim[:,1].view(-1,1))/2
    output[:,1:5] /= scaling_factor
    for i in range(output.shape[0]):
        output[i, [1,3]] = torch.clamp(output[i, [1,3]], 0.0, im_dim[i,0])
        output[i, [2,4]] = torch.clamp(output[i, [2,4]], 0.0, im_dim[i,1])
    x = []
    for i in range(output.shape[0]):
        if output[i][-1]==0:
            x.append(list(output[i]))
    output=torch.from_numpy(np.array(x))

    if biggest_per ==True:
        area_list =list(map(lambda x: area(x), output))
        output = output[np.argmax(area_list)]
        output = output[np.newaxis, :]
    return output,True

'''
计算裁剪面积
'''
def area(x):
    if x[-1] == 0:
        size = int((x[3]-x[1])*(x[4]-x[2]))
    else:
        size = 0
    return size

'''
预处理图片
'''
def prep_image(img, inp_dim):
    """
    Prepare image for inputting to the neural network.

    Returns a Variable
    """
    orig_im = img
    dim = orig_im.shape[1], orig_im.shape[0]
    img = (letterbox_image(orig_im, (inp_dim, inp_dim)))
    img_ = img[:,:,::-1].transpose((2,0,1)).copy()
    img_ = torch.from_numpy(img_).float().div(255.0).unsqueeze(0)
    return img_, orig_im, dim



'''
将视野中最大的人裁剪出来
'''
def cut(x, img):
    c1 = tuple(x[1:3].int())
    c2 = tuple(x[3:5].int())
    cls = int(x[-1])
    if cls == 0:
        cut_person_opencv = img[c1[1]:c2[1], c1[0]:c2[0]]
        return cut_person_opencv


'''
保存gf与pid到本地，不断存取与保存
恶心的代码，不要用
'''
def save_gf_pid(perlib, gf, name='0001'):
    # TODO
    namelist, gf_list = [], []
    # gf = torch.cat(gf, 0)
    gf_dir = os.path.join(perlib, 'GF.cvs')
    pid_dir = os.path.join(perlib, 'PID.cvs')
    # 只有0个或1个cvs时，直接保存
    if not (os.path.exists(gf_dir) and os.path.exists(pid_dir)):
        print('不存在')
        namelist.append([int(name)])
        gf_list.append(gf)
        # gf_list = gf_list[np.newaxis, :]
        # gf_list= torch.cat(gf_list, 0)
    # 已有2cvs时，加载原数据，添加并保存
    else:
        gf_list = torch.from_numpy(np.loadtxt(gf_dir, delimiter=","))
        namelist = np.loadtxt(pid_dir, delimiter=",")
        namelist = list([namelist]) if not namelist.shape else list(namelist)
        gf_list.append(gf)
        namelist.append(int(name))
    # gf_list = np.asarray(gf_list)
    gf_list = torch.cat(gf_list, 0)
    namelist = np.asarray(namelist)
    np.savetxt(pid_dir, namelist, fmt="%f", delimiter=",")
    np.savetxt(gf_dir, gf_list, fmt="%f", delimiter=",")


''''
匹配算法
'''
def reidRecognition(qf,gf,g_pid,threshold,threshold_save):
    if type(gf)==list:
        print('无行人库')
        return False,'Unknow',False,-1

    m, n = qf.size(0), gf.size(0)
    distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
              torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    distmat.addmm_(1, -2, qf, gf.t())
    # dismat_list.append(distmat)
    distmat = distmat.numpy()
    if  np.min(distmat) < threshold:
        position = np.argmin(distmat)
        # name = '000'+str(g_pid[position])
        name =str('%04d' %g_pid[position])
        print('REID识别成功:%s %d'%(name,np.min(distmat)))
        if np.min(distmat)<threshold_save:
            return True,name,True,np.min(distmat)
        else: return True, name ,False,np.min(distmat)

    else :
        print('REID库无此人')
        return False,'Unknow',False,np.min(distmat)
'''
给图片加框和名字
'''
def Image_rectangle_text(img,c1,c2,name):
    c1 = tuple(c1.int())
    c2 = tuple(c2.int())
    color = color_dict[str(name)]
    cv2.rectangle(img, c1, c2, color,1)    # 加框
    t_size = cv2.getTextSize(str(name), cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
    c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
    cv2.rectangle(img, c1, c2, color, -1) # -1填充作为文字框底色
    cv2.putText(img, str(name), (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225, 255, 255], 1);
    return img

# '''
# 在图片右上角加入时间统计
# '''
# def write_information(img,information):
#     c3=(900,0)
#     c4=(1280,100) #右上角位置
#     cv2.rectangle(img, c3, c4, (0,0,0), -1) # -1填充作为文字框底色
#     for i in range(len(information)):
#         info = str(information[i][0])+ " : "+str(information[i][1])+' s'
#         info_size = cv2.getTextSize(info, cv2.FONT_HERSHEY_COMPLEX, 1, 1)[0]
#         pos_x = c3[0]
#         pos_y = c3[1]+(info_size[1] + 4)*(i+1)
#         cv2.putText(img, info, (pos_x, pos_y), cv2.FONT_HERSHEY_COMPLEX, 1, (204, 209, 72), 1)
#
#     return img


'''
在图片右上角加入时间统计
'''
def write_information(img,time_dict):
    c3=(900,0)
    c4=(1280,85) #右上角位置
    cv2.rectangle(img, c3, c4, (0,0,0), -1) # -1填充作为文字框底色
    i = 0
    for key in time_dict:
        i+=1
        info = str(key)+ " : "+str('%.2f'%time_dict[key]) + ' s'
        info_size = cv2.getTextSize(info, cv2.FONT_HERSHEY_COMPLEX, 1, 1)[0]
        pos_x = c3[0]
        pos_y = c3[1]+(info_size[1] + 4)*i
        cv2.putText(img, info, (pos_x, pos_y), cv2.FONT_HERSHEY_COMPLEX, 1, (204, 209, 72), 1)

    return img

'''
给人物累加滞留时间
'''
def add_time(time_dict, name, add):
    if name =='Unknow':
        return time_dict
    if not name in time_dict:
        time_dict[name] = 0
    time_dict[name] += add

    return time_dict


# 判断两个矩形是否相交
# box=(xA,yA,xB,yB)
def mat_inter(box1, box2):
    x01, y01, x02, y02 = box1
    x11, y11, x12, y12 = box2

    lx = abs((x01 + x02) / 2 - (x11 + x12) / 2)
    ly = abs((y01 + y02) / 2 - (y11 + y12) / 2)
    sax = abs(x01 - x02)
    sbx = abs(x11 - x12)
    say = abs(y01 - y02)
    sby = abs(y11 - y12)
    if lx <= (sax + sbx) / 2 and ly <= (say + sby) / 2:
        return True
    else:
        return False


def solve_coincide(box1, box2):
    # box=(xA,yA,xB,yB)
    # 计算两个矩形框的重合度
    if mat_inter(box1, box2) == True:
        x01, y01, x02, y02 = box1
        x11, y11, x12, y12 = box2
        col = min(x02, x12) - max(x01, x11)
        row = min(y02, y12) - max(y01, y11)
        intersection = col * row
        area1 = (x02 - x01) * (y02 - y01)
        area2 = (x12 - x11) * (y12 - y11)
        coincide = intersection / (area1 + area2 - intersection)
        return coincide
    else:
        return False

def frame_nn(label_last,output_now,output_last):
    rem,coincide2 = -1,0
    for i in range(output_last.shape[0]):
        coincide =  solve_coincide((int(output_now[1]),int(output_now[2]),int(output_now[3]),int(output_now[4])),(int(output_last[i][1]),int(output_last[i][2]),int(output_last[i][3]),int(output_last[i][4])))
        if coincide and (coincide>coincide2):
            rem,coincide2 = i,coincide
    if rem == -1:
        return False
    else:
        return label_last[rem]

# """
# 读取多个视频
# 2018.07.25
# """
# def load_muli_video():
#     cap1 = cv2.VideoCapture('./Test/video_in/cap1_18-06-20-110113.avi')
#     cap2 = cv2.VideoCapture('./Test/video_in/cap2_18-06-20-110109.avi')
#     cap3 = cv2.VideoCapture('./Test/video_in/cap3_18-06-20-110109.avi')
#     return cap1, cap2, cap3
#
# def load_gallery_feature(args):
#     torch.manual_seed(args.seed)
#     os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_devices
#     use_gpu = torch.cuda.is_available()
#     if args.use_cpu: use_gpu = False
#
#     print("==========\nArgs:{}\n==========".format(args))
#
#     if use_gpu:
#         print("Currently using GPU {}".format(args.gpu_devices))
#         cudnn.benchmark = True
#         torch.cuda.manual_seed_all(args.seed)
#     else:
#         print("Currently using CPU (GPU is highly recommended)")
#     print("Initializing dataset {}".format(args.dataset))
#     dataset = data_manager.init_dataset(
#         root=args.root, name=args.dataset, split_id=args.split_id,
#         cuhk03_labeled=args.cuhk03_labeled, cuhk03_classic_split=args.cuhk03_classic_split,
#     )
#     transform_test = T.Compose([
#         T.Resize((args.height, args.width)),
#         T.ToTensor(),
#         T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#     ])
#
#     pin_memory = True if use_gpu else False
#     # galleryloader = DataLoader(
#     #     ImageDataset(dataset.gallery, transform=transform_test),
#     #     batch_size=args.test_batch, shuffle=False, num_workers=args.workers,
#     #     pin_memory=pin_memory, drop_last=False,
#     # )
#     print(dataset.gallery)
#     galleryloader = DataLoader(
#         ImageDataset(dataset.gallery, transform=transform_test),
#         batch_size=1, shuffle=False, num_workers=0,
#         pin_memory=pin_memory, drop_last=False,
#     )
#     print("Initializing model: {}".format(args.arch))
#     # num_classes随训练集不同而改变：[market：751; duke:702;cuhk03：]
#     model = models.init_model(name=args.arch, num_classes=751, loss={'xent'}, use_gpu=use_gpu)
#     print("Model size: {:.5f}M".format(sum(p.numel() for p in model.parameters()) / 1000000.0))
#     if args.resume:
#         print("Loading checkpoint from '{}'".format(args.resume))
#         checkpoint = torch.load(args.resume)
#         model.load_state_dict(checkpoint['state_dict'])
#     if use_gpu:
#         model = nn.DataParallel(model).cuda()
#     model.eval()
#     with torch.no_grad():
#         gf, g_pids, g_camids = [], [], []
#         for batch_idx, (imgs, pids, camids) in enumerate(galleryloader):
#             if use_gpu: imgs = imgs.cuda()
#             print(imgs.shape)
#             features = model(imgs)
#             features = features.data.cpu()
#             print(features)
#             gf.append(features)
#             g_pids.extend(pids)
#             g_camids.extend(camids)
#         print(type(gf))
#         print(gf)
#         gf = torch.cat(gf, 0)
#         print(gf.shape)
#         print(g_pids)
#         g_pids = np.asarray(g_pids)
#         # g_camids = np.asarray(g_camids)
#         # np.savetxt('saved_emb/lab/gf1_res.cvs', gf, fmt="%f", delimiter=",")
#         # np.savetxt('saved_emb/lab/g_pids.cvs', g_pids, fmt="%f", delimiter=",")
#         # np.savetxt('saved_emb/lab/g_camids.cvs', g_camids, fmt="%f", delimiter=",")
#         print(g_pids)
#         print("Extracted features for gallery set, obtained {}-by-{} matrix".format(gf.size(0), gf.size(1)))
#         print("success save gallery feature,g_pids and g_camids")
#         return gf, g_pids
#
# ''''
# 匹配算法
# '''
# def reidRecognition_test(qf,gf,g_pid,threshold,threshold_save):
#     '''
#
#     :param qf: 当前行人特征
#     :param gf: 特征库
#     :param g_pid:目标人物
#     :param threshold:判断是否能确认的阈值
#     :param threshold_save:判断是否保存的阈值
#     :return:
#     '''
#     m, n = qf.size(0), gf.size(0)
#     distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
#               torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
#     distmat.addmm_(1, -2, qf, gf.t())
#     # dismat_list.append(distmat)
#     distmat = distmat.numpy()
#     if  np.min(distmat) < threshold:
#         position = np.argmin(distmat)
#         # name = '000'+str(g_pid[position])
#         name =str('%04d' %g_pid[position])
#         print('REID识别成功:%s %d'%(name,np.min(distmat)))
#         # if np.min(distmat)<threshold_save:
#         #     return True,name,True,np.min(distmat)
#         # else: return True, name ,False,np.min(distmat)
#         return True, name, False, np.min(distmat)
#     else :
#         print('REID库无此人')
#         return False,'Unknow',False,np.min(distmat)
#
#
#
#
#
#
#
#
# '''
# 给图片加框
# 2018.07.15
# '''
# def Image_rectangle_text_1(img,c1,c2,name):
#     c1 = tuple(c1.int())
#     c2 = tuple(c2.int())
#     color = color_dict[str(name)]
#     cv2.rectangle(img, c1, c2, color,1)    # 加框
#     # t_size = cv2.getTextSize(str(name), cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
#     # c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
#     # cv2.rectangle(img, c1, c2, color, -1) # -1填充作为文字框底色
#     #cv2.putText(img, str(name), (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225, 255, 255], 1);
#     return img
#
# '''
# 给图片加框
# 2018.07.15
# '''
# def Image_rectangle_text_2(img,c1,c2,name):
#     c1 = tuple(c1.int())
#     c2 = tuple(c2.int())
#     color = (255, 0, 0)
#
#     t_size = cv2.getTextSize(str(name), cv2.FONT_HERSHEY_SIMPLEX, 1, 1)[0]
#     c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
#     # cv2.rectangle(img, c1, c2, color, -1) # -1填充作为文字框底色
#     cv2.putText(img, 'test', (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_SIMPLEX, 2, [0, 255, 255], 2);
#     return img