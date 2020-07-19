import os
import sys
import time
import mysql.connector

import mrcnn.model as modellib

from cv2 import cv2
from glob import glob
from mrcnn import utils
from tracker import Tracker
from copy import deepcopy
from samples.coco import coco
from threading import Thread
from sklearn.metrics import classification_report

try:
    mode = sys.argv[1]
except:
    mode = ''

mydb = mysql.connector.connect(
  host="localhost",
  user="root",
  passwd="",
  database="stream"
)

cursor = mydb.cursor()
cursor.execute('truncate log')
mydb.commit()

# check model, download if not exist
if not os.path.exists('mask_rcnn_coco.h5'):
    utils.download_trained_weights('mask_rcnn_coco.h5') 

# create config
class InferenceConfig(coco.CocoConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()
config.display()

# modelling
model = modellib.MaskRCNN(mode = "inference", model_dir = 'logs/', config = config)

model.load_weights('mask_rcnn_coco.h5', by_name = True)

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

master = None
big_ROI = [50, 250, 1600, 425]
detection = []
true = []

def change_state(id, state):
    global cursor
    sql = "select * from log where id={}".format(id)
    cursor.execute(sql)
    result = cursor.fetchall()
    state = "'Ditempati'" if state == '1' else "'Kosong'"
    if len(result) > 0:
        sql = "update log set nama={}, timestamp=CURRENT_TIMESTAMP where id={}".format(state, id)
    else:
        sql = "insert into log (id, nama, timestamp) values ({}, {}, CURRENT_TIMESTAMP)".format(id, state)
    cursor.execute(sql)
    mydb.commit()

def worker(frame):
    global master
    results = model.detect([frame])
    master = results[0]

# datas = glob('videos/data1.mp4')
# datas = glob('videos/data24.mp4')
datas = glob('videos/edit_25.mp4')
# datas = glob('videos/malam.mp4')
for video in datas:
    parking_lot = Tracker()
    capture = cv2.VideoCapture(video)
    counter = 0
    while True:
        ret, frame = capture.read()
        # time.sleep(1/30)
        if not ret:
            break
        counter += 1

        if counter == 1:
            worker(frame)

        if counter % 20 == 0:
            cloned = deepcopy(frame)
            cloned[:,:big_ROI[0]] = 0
            cloned[:,big_ROI[2]:] = 0
            cloned[:big_ROI[1],:] = 0 
            cloned[big_ROI[3]:,:] = 0 
            t = Thread(target = worker, args = (cloned, ))
            t.start()
        
        
        cv2.rectangle(frame, (big_ROI[0], big_ROI[1]), (big_ROI[2], big_ROI[3]), (255,255,255), 3)

        if master != None:
            sub = deepcopy(master)
            current = []
            for i in range(len(sub['rois'])):
                if class_names[sub['class_ids'][i]] == 'car' or class_names[sub['class_ids'][i]] == 'boat':
                    y1, x1, y2, x2 = sub['rois'][i]
                    current.append([x1, y1, x2, y2])
            space, temporary = parking_lot.update(current)
            
            # USE ParkingLotBeta
            for item in temporary:
                x, y = item['coord']
                cv2.circle(frame, (x, y), 5, (0, 255, 0), 1)
                cv2.putText(frame, str(item['detected']), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            for item in space:
                change_state(item['id'], str(item['status']))
                x1, y1, x2, y2 = item['square']
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 1)
                cv2.putText(frame, str(item['id']) + ' | ' + str(item['status']), (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        frame = cv2.resize(frame, (1280, 720))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if mode == 'direct':
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            cv2.imshow('video', frame)
            # cv2.imwrite('videos/hasil/malam/{}.png'.format(counter), frame)
        else:
            sys.stdout.buffer.write(frame.tobytes())
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    capture.release()
    cv2.destroyAllWindows()