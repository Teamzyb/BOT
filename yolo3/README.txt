.JSON�ļ���ȡ����
����Ŀ���ļ����£�
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

���Կ�������һ���ֵ䣬anno��һ��һ��list��list���ְ�����һ���ֵ�����ֵ���������keyֵ
����ʹ�����µĶ�ȡ����
with open("test.json",'r') as load_f:
    load_dict = json.load(load_f)
    object = load_dict["annotation"][0]["object"]
���԰�object���ֵ����ȡ����



