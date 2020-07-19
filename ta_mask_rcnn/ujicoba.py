import os
import json
import pandas as pd

from glob import glob

lot = {}
coord = {}
base_path = 'datasets/CNREXT/'
images = glob(base_path + '*/*/*/*/*')
csvs = glob(base_path + '*.csv')
txts = glob(base_path + 'labels/camera*.txt')

for i in range(len(txts)):
    txt = txts[i]
    txt_data = open(txt, 'r').read()
    txt_data = txt_data.split('\n')

    for item in txt_data:
        try:
            path, label = item.split(' ')
            weather, date, camera, name = path.split('/')
            name = name.split('.jpg')[0]
            _, name_date, name_time, _, slotId = name.split('_')
            name_time = name_time.split('.')
            name_time = ''.join(name_time)
            true_name = name_date + '_' + name_time + '.jpg'
            true_path = base_path + 'image/' + weather + '/' + date + '/' + camera + '/' + true_name

            if true_path not in lot:
                lot[true_path] = []
            temp = {
                'camera': str(int(camera.split('camera')[-1])-1),
                'slotid': str(slotId), 
                'label': str(label)
            }
            lot[true_path].append(temp)
        except:
            pass

for i in range(len(csvs)):
    csv = csvs[i]
    csv_data = pd.read_csv(csv)

    coord[str(i)] = {}

    for j in range(len(csv_data)):
        slotId = str(csv_data.iloc[j]['SlotId'])
        x = int(csv_data.iloc[j]['X'])
        y = int(csv_data.iloc[j]['Y'])
        w = int(csv_data.iloc[j]['W'])
        h = int(csv_data.iloc[j]['H'])
        coord[str(i)][slotId] = {
            "x": x, 
            "y": y, 
            "w": w, 
            "h": h,
            'x1': int(x * 1000/2592),
            'y1': int(y * 750/1944),
            'x2': int((x + w) * 1000/2592),
            'y2': int((y + h) * 750/1944)
        }

# exit()

import mrcnn.model as modellib
from samples.coco import coco
from cv2 import cv2
from tqdm import tqdm
from sklearn.metrics import classification_report

class InferenceConfig(coco.CocoConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()
config.display()
model = modellib.MaskRCNN(mode = "inference", model_dir = 'logs/', config = config)
model.load_weights('mask_rcnn_coco_resnet50.h5', by_name = True)
class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 
               'bus', 'train', 'truck', 'boat', 'traffic light', 
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 
               'kite', 'baseball bat', 'baseball glove', 'skateboard', 
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 
               'teddy bear', 'hair drier', 'toothbrush']



for key, value in tqdm(lot.items()):
    true = []
    detected = []
    image = cv2.imread(key)
    # image = cv2.resize(image, (2592, 1944))
    result = model.detect([image])
    result = result[0]


    for item in value:
        coordinate = coord[item['camera']][item['slotid']]
        x1 = coordinate['x1']
        y1 = coordinate['y1']
        x2 = coordinate['x2']
        y2 = coordinate['y2']
        x = coordinate['x']
        y = coordinate['y']
        label = item['label']

        cv2.rectangle(image, (x1, y1), (x2, y2), (255,255,255), 2)
        marker = 0

        for i in range(len(result['rois'])):
            if class_names[result['class_ids'][i]] == 'car':
                y1_roi, x1_roi, y2_roi, x2_roi = result['rois'][i]
                x = int((x1_roi + x2_roi) / 2) - 5
                y = int((y1_roi + y2_roi) / 2) + 5

                # cv2.circle(image, (x, y), 3, (255,255,255), 2)

                if x1 < x < x2 and y1 < y < y2:
                    marker = 1
                    break

        for i in range(len(result['rois'])):
            if class_names[result['class_ids'][i]] == 'car':
                y1_roi, x1_roi, y2_roi, x2_roi = result['rois'][i]
                x = int((x1_roi + x2_roi) / 2) - 5
                y = int((y1_roi + y2_roi) / 2) + 5
                cv2.circle(image, (x, y), 3, (0,255,0), 3)

        true.append(int(label))
        detected.append(marker)

    os.makedirs(os.path.dirname(key.replace('image', 'resnet50')), exist_ok=True)
    cv2.imwrite(key.replace('image', 'resnet50'), image)

    # cv2.imshow('image', image)
    # cv2.waitKey(0)

    report = classification_report(true, detected, output_dict=True)
    # print(report)

    file = open((key.replace('image', 'resnet50')).replace('.jpg', '.txt'), 'w')
    file.write(json.dumps(report, indent=2))
    file.close()