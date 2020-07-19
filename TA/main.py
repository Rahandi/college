from copy import deepcopy
from cv2 import cv2
from tqdm import tqdm
from util import Util

util = Util()

# config
anchor_stride = 16
anchor_sizes = [16, 32, 64]
anchor_ratios = [[1,1], [1,2], [2,1]]

shortest_side_image = 300

master_anchors = {}
all_image_data = util.parse_annotation('annotation_cam8.txt')
for key, value in tqdm(all_image_data.items()):
    image = cv2.imread(key)
    image, width, height, scale = util.image_resize(shortest_side_image, image)
    key = '{},{}'.format(width, height)
    if key not in master_anchors:
        master_anchors[key] = util.generate_anchor(anchor_stride, anchor_sizes, anchor_ratios, width, height)
    anchor = deepcopy(master_anchors[key])