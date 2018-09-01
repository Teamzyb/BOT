.JSON文件读取方法
本项目的文件如下：
{
 "annotation": [
  {
   "filename": "scene_1_00002.jpg",
   "image_id": "id_2",
   "width": 1920,
   "height": 1080,
   "depth": 3,
   "object": [
    {
     "name": "person",
     "minx": 495,
     "miny": 176,
     "maxx": 873,
     "maxy": 830,
     "gender": 1,
     "staff": 1,
     "customer": 0,
     "stand": 1,
     "sit": 0,
     "play_with_phone": 0
    },
    {
     "name": "person",
     "minx": 145,
     "miny": 96,
     "maxx": 255,
     "maxy": 312,
     "gender": 1,
     "staff": 1,
     "customer": 0,
     "stand": 1,
     "sit": 0,
     "play_with_phone": 0
    }
   ]
  }
 ]
}

可以看到这是一个字典，anno对一了一个list。list里又包含了一个字典这个字典里包含多个key值
所以使用如下的读取方法
with open("test.json",'r') as load_f:
    load_dict = json.load(load_f)
    object = load_dict["annotation"][0]["object"]
可以把object里的值给读取出来



