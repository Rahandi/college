import sys
import random

import numpy as np
import pandas as pd

from cv2 import cv2
from tqdm import trange, tqdm
from threading import Thread

class Util:
    def __init__(self):
        self.threshold_1 = 0.3
        self.threshold_2 = 0.7

    def parse_annotation(self, annotation_path):
        annotation = pd.read_csv(annotation_path, header=None, names=['filepath', 'xmin', 'ymin', 'xmax', 'ymax', 'label'])

        all_image = {}
        for i in trange(len(annotation)):
            data = annotation.iloc[i]
            filepath = data['filepath']
            xmin = data['xmin']
            ymin = data['ymin']
            xmax = data['xmax']
            ymax = data['ymax']
            label = data['label']

            if filepath not in all_image:
                all_image[filepath] = {
                    'groundtruth': []
                }

            all_image[filepath]['groundtruth'].append({
                'xmin': xmin,
                'ymin': ymin,
                'xmax': xmax,
                'ymax': ymax,
                'label': label
            })

        return all_image

    def generate_anchor(self, stride, anchor_sizes, anchor_ratios, image_width, image_height):
        anchors = []
        workers = []
        dictionary = {}

        def work(identifier, w, h, size, ratio, size_index, ratio_index):
            a1 = size * ratio[0]
            a2 = size * ratio[1]
            ax1 = w + (a1/2)
            ax2 = w - (a1/2)
            ay1 = h + (a2/2)
            ay2 = h - (a2/2)

            temp = {
                'x': w,
                'y': h,
                'xmin': ax1 if ax1<ax2 else ax2,
                'ymin': ax1 if ax1>ax2 else ax2,
                'xmax': ay1 if ay1<ay2 else ay2,
                'ymax': ay1 if ay1>ay2 else ay2,
                'type': 'negative',
                'size': size,
                'ratio': ratio,
                'size_index': size_index,
                'ratio_index': ratio_index
            }

            if temp['xmin'] < 0 or temp['ymin'] < 0 or temp['xmax'] > image_width or temp['ymax'] > image_height:
                dictionary[identifier] = None
            else:
                dictionary[identifier] = temp
        
        identifier = 0
        for size in trange(len(anchor_sizes), leave=None):
            for ratio in trange(len(anchor_ratios), leave=None):
                for w in trange(0, image_width, stride, leave=None):
                    for h in trange(0, image_height, stride, leave=None):
                        worker = Thread(
                                    target=work,
                                    args=(identifier, w, h, anchor_sizes[size], anchor_ratios[ratio], size, ratio)
                                )
                        workers.append(worker)
                        worker.start()
                        identifier += 1

        for worker in workers:
            worker.join()

        for i in range(identifier):
            if dictionary[i]:
                anchors.append(dictionary[i])

        return anchors

    def image_resize(self, shortest_side_target, image):
        width, height, _  = image.shape
        scale = shortest_side_target / (width if width<height else height)
        target_width = int(width * scale)
        target_height = int(height * scale)
        target_image = cv2.resize(image, (target_width,target_height), cv2.INTER_AREA)
        return target_image, target_width, target_height, scale

    def calc_iou(self, a, b):
        def intersection(a, b):
            xmin = max(a['xmin'], b['xmin'])
            ymin = max(a['ymin'], b['ymin'])
            xmax = min(a['xmax'], b['xmax'])
            ymax = min(a['ymax'], b['ymax'])
            area = (xmax - xmin) * (ymax - ymin)
            if area < 0:
                return 0
            return float(area)

        def union(a, b, intersection_area):
            area_a = (a['xmax']-a['xmin']) * (a['ymax']-a['ymin'])
            area_b = (b['xmax']-b['xmin']) * (b['ymax']-b['ymin'])
            area = area_a + area_b - intersection_area
            return float(area)

        if a['xmax'] < a['xmin'] or a['ymax'] < a['ymin'] or b['xmax'] < b['xmin'] or b['ymax'] < b['ymin']:
            return 0.0

        area_intersection = intersection(a, b)
        area_union = union(a, b, area_intersection)

        return area_intersection / area_union

    def anchor_selection(self, groundtruths, anchors, resize_scale, image_width, image_height, num_anchor_per_pixel, num_anchor_ratio, max_anchor_num = 256, ignore_max_anchor_num = False):        
        gt = []
        for item in groundtruths:
            gt.append({
                'xmin': item['xmin'] * resize_scale,
                'ymin': item['ymin'] * resize_scale,
                'xmax': item['xmax'] * resize_scale,
                'ymax': item['ymax'] * resize_scale,
                'label': item['label']
            })
        groundtruths = gt

        y_rpn_overlap = np.zeros((image_height, image_width, num_anchor_per_pixel))
        y_is_box_valid = np.zeros((image_height, image_width, num_anchor_per_pixel))
        y_rpn_regr = np.zeros((image_height, image_width, num_anchor_per_pixel * 4))

        num_anchors_for_groundtruth = np.zeros(len(gt)).astype(int)
        best_anchor_for_groundtruth = -1*np.ones((len(gt), 4)).astype(int)
        best_iou_for_groundtruth = np.zeros(len(gt)).astype(np.float32)
        best_anchor_coordinate_for_groundtruth = np.zeros((len(gt), 4)).astype(int)
        best_anchor_regression_for_groundtruth = np.zeros((len(gt), 4)).astype(np.float32)

        def work(anchor_index):
            center_anchor_x = (anchors[anchor_index]['xmin'] + anchors[anchor_index]['xmax']) / 2.0
            center_anchor_y = (anchors[anchor_index]['ymin'] + anchors[anchor_index]['ymax']) / 2.0

            best_iou_for_anchor = 0
            best_regression_for_anchor = (0, 0, 0, 0)
            for groundtruth_index in range(len(groundtruths)):
                center_groundtruth_x = (groundtruths[groundtruth_index]['xmin'] + groundtruths[groundtruth_index]['xmax']) / 2.0
                center_groundtruth_y = (groundtruths[groundtruth_index]['ymin'] + groundtruths[groundtruth_index]['ymax']) / 2.0
                current_iou = self.calc_iou(groundtruths[groundtruth_index], anchors[anchor_index])
                if current_iou > best_iou_for_groundtruth[groundtruth_index] or current_iou > self.threshold_2:
                    tx = (center_groundtruth_x - center_anchor_x) / (anchors[anchor_index]['xmax'] - anchors[anchor_index]['xmin'])
                    ty = (center_groundtruth_y - center_anchor_y) / (anchors[anchor_index]['ymax'] - anchors[anchor_index]['ymin'])
                    tw = np.log((groundtruths[groundtruth_index]['xmax'] - groundtruths[groundtruth_index]['xmin']) / (anchors[anchor_index]['xmax'] - anchors[anchor_index]['xmin']))
                    th = np.log((groundtruths[groundtruth_index]['ymax'] - groundtruths[groundtruth_index]['ymin']) / (anchors[anchor_index]['ymax'] - anchors[anchor_index]['ymin']))
                
                if groundtruths[groundtruth_index]['label'] != 'bg':
                    if current_iou > best_iou_for_groundtruth[groundtruth_index]:
                        best_anchor_for_groundtruth[groundtruth_index] = [anchors[anchor_index]['y'], anchors[anchor_index]['x'], anchors[anchor_index]['ratio_index'], anchor_index[anchor_index]['size_index']]
                        best_iou_for_groundtruth[groundtruth_index] = current_iou
                        best_anchor_coordinate_for_groundtruth[groundtruth_index, :] = [anchors[anchor_index]['xmax'], anchors[anchor_index]['xmin'], anchors[anchor_index]['ymax'], anchors[anchor_index]['ymin']]
                        best_anchor_regression_for_groundtruth[groundtruth_index, :] = [tx, ty, tw, th]

                    if current_iou > self.threshold_2:
                        anchors[anchor_index]['type'] = 'positive'
                        num_anchors_for_groundtruth[groundtruth_index] += 1
                        if current_iou > best_iou_for_anchor:
                            best_iou_for_anchor = current_iou
                            best_regression_for_anchor = (tx, ty, tw, th)

                    if self.threshold_1 < current_iou < self.threshold_2:
                        if anchors[anchor_index]['type'] != 'positive':
                            anchors[anchor_index]['type'] = 'neutral'
            
            if anchors[anchor_index]['type'] == 'negative':
                y_is_box_valid[anchors[anchor_index]['y'], anchors[anchor_index]['x'], anchors[anchor_index]['ratio_index'] + num_anchor_ratio * anchors[anchor_index['size_index']]] = 1
                y_rpn_overlap[anchors[anchor_index]['y'], anchors[anchor_index]['x'], anchors[anchor_index]['ratio_index'] + num_anchor_ratio * anchors[anchor_index['size_index']]] = 0
            elif anchors[anchor_index]['type'] == 'neutral':
                y_is_box_valid[anchors[anchor_index]['y'], anchors[anchor_index]['x'], anchors[anchor_index]['ratio_index'] + num_anchor_ratio * anchors[anchor_index['size_index']]] = 0
                y_rpn_overlap[anchors[anchor_index]['y'], anchors[anchor_index]['x'], anchors[anchor_index]['ratio_index'] + num_anchor_ratio * anchors[anchor_index['size_index']]] = 0
            elif anchors[anchor_index]['type'] == 'positive':
                y_is_box_valid[anchors[anchor_index]['y'], anchors[anchor_index]['x'], anchors[anchor_index]['ratio_index'] + num_anchor_ratio * anchors[anchor_index['size_index']]] = 1
                y_rpn_overlap[anchors[anchor_index]['y'], anchors[anchor_index]['x'], anchors[anchor_index]['ratio_index'] + num_anchor_ratio * anchors[anchor_index['size_index']]] = 1
                splice_index = 4 * (anchors[anchor_index]['ratio_index'] + num_anchor_ratio * anchors[anchor_index['size_index']])
                y_rpn_regr[anchors[anchor_index]['y'], anchors[anchor_index]['x'], splice_index:splice_index+4] = best_regression_for_anchor

        pool = []
        for anchor_index in range(len(anchors)):
            worker = Thread(target=work, args=(anchor_index,))
            pool.append(worker)
            worker.start()

        for worker in pool:
            worker.join()

        for index in range(num_anchors_for_groundtruth.shape[0]):
            if num_anchors_for_groundtruth[index] == 0:
                if best_anchor_for_groundtruth[index] == 0:
                    continue
                y_is_box_valid[best_anchor_for_groundtruth[index, 0], best_anchor_for_groundtruth[index, 1], best_anchor_for_groundtruth[index, 2], best_anchor_for_groundtruth[index, 3]] = 1
                y_is_box_valid[best_anchor_for_groundtruth[index, 0], best_anchor_for_groundtruth[index, 1], best_anchor_for_groundtruth[index, 2], best_anchor_for_groundtruth[index, 3]] = 1
                splice_index = 4 * (best_anchor_for_groundtruth[index, 2] + num_anchor_ratio * best_anchor_for_groundtruth[index, 3])
                y_rpn_regr[best_anchor_for_groundtruth[index, 0], best_anchor_for_groundtruth[index, 1], splice_index:splice_index+4] = best_anchor_regression_for_groundtruth[index, :]
        
        y_rpn_overlap = np.transpose(y_rpn_overlap, (2, 0, 1))
        y_rpn_overlap = np.expand_dims(y_rpn_overlap, axis=0)
        y_is_box_valid = np.transpose(y_is_box_valid, (2, 0, 1))
        y_is_box_valid = np.expand_dims(y_is_box_valid, axis=0)
        y_rpn_regr = np.transpose(y_rpn_regr, (2, 0, 1))
        y_rpn_regr = np.expand_dims(y_rpn_regr, axis=0)

        positive_location = np.where(np.logical_and(y_rpn_overlap[0, :, :, :] == 1, y_is_box_valid[0, :, :, :] == 1))
        negative_location = np.where(np.logical_and(y_rpn_overlap[0, :, :, :] == 0, y_is_box_valid[0, :, :, :] == 1))

        if len(positive_location[0]) > max_anchor_num:
            if not ignore_max_anchor_num:
                raise Exception('positive anchor amount is more than max_anchor_num, consider raising max_anchor_num value, or set ignore_max_anchor_num to True')
        if len(negative_location[0]) + len(positive_location[0]) > max_anchor_num:
            val_locs = random.sample(range(len(negative_location[0])), len(negative_location[0]) - len(positive_location[0]))
            y_is_box_valid[0, negative_location[0][val_locs], negative_location[1][val_locs], negative_location[2][val_locs]] = 0
        
        y_rpn_cls = np.concatenate([y_is_box_valid, y_rpn_overlap], axis=1)
        y_rpn_regr = np.concatenate([np.repeat(y_rpn_overlap, 4, axis=1), y_rpn_regr], axis=1)

        return np.copy(y_rpn_cls), np.copy(y_rpn_regr), len(positive_location[0])

    def get_target(all_image_data):
        pass