#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import cv2
import time
import torch
import random
import numpy as np
import torchvision
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as T

from detect_lib import fasterrcnn_resnet50_fpn, maskrcnn_resnet50_fpn

from IPython import embed


COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]


def get_prediction(task, img_path, threshold):

    transform = T.Compose([T.ToTensor()]) # Defing PyTorch Transform
    img  = Image.open(img_path) # Load the image
    img  = transform(img) # Apply the transform to the image
    pred = model([img]) # Pass the image to the model

    if task == 'detect':

        pred_class = [COCO_INSTANCE_CATEGORY_NAMES[i] for i in list(pred[0]['labels'].numpy())] # Get the Prediction Score
        pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].detach().numpy())] # Bounding boxes
        pred_score = list(pred[0]['scores'].detach().numpy())  # already sorted in ascend
        pred_t = [pred_score.index(x) for x in pred_score if x > threshold][-1] # Get list of index with score greater than threshold.
        pred_boxes = pred_boxes[:pred_t+1]
        pred_class = pred_class[:pred_t+1]

        return pred_boxes, pred_class

    else:

        pred_score = list(pred[0]['scores'].detach().numpy())
        pred_t = [pred_score.index(x) for x in pred_score if x>threshold][-1]
        masks  = (pred[0]['masks']>0.5).squeeze().detach().cpu().numpy()
        pred_class = [COCO_INSTANCE_CATEGORY_NAMES[i] for i in list(pred[0]['labels'].numpy())]  # Get the Prediction Score
        pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].detach().numpy())] # Bounding boxes
        masks = masks[:pred_t+1]
        pred_boxes = pred_boxes[:pred_t+1]
        pred_class = pred_class[:pred_t+1]

        return masks, pred_boxes, pred_class


def random_colour_masks(image):

    colours = [[0, 255, 0],   [0, 0, 255],   [255, 0, 0],
               [0, 255, 255], [255, 255, 0], [255, 0, 255],
               [80, 70, 180], [250, 80, 190],[245, 145, 50],
               [70, 150, 250],[50, 190, 190]]

    r = np.zeros_like(image).astype(np.uint8)
    g = np.zeros_like(image).astype(np.uint8)
    b = np.zeros_like(image).astype(np.uint8)
    r[image == 1], g[image == 1], b[image == 1] = colours[random.randrange(0,10)]
    coloured_mask = np.stack([r, g, b], axis=2)

    return coloured_mask


def instance_segmentation_api(img_path, threshold=0.5, rect_th=3, text_size=3, text_th=3):

    masks, boxes, pred_cls = get_prediction('segment', img_path, threshold)
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    for i in range(len(masks)):

      rgb_mask = random_colour_masks(masks[i])
      img = cv2.addWeighted(img, 1, rgb_mask, 0.5, 0)
      cv2.rectangle(img, boxes[i][0], boxes[i][1],color=(0, 255, 0), thickness=rect_th)
      cv2.putText(img,pred_cls[i], boxes[i][0], cv2.FONT_HERSHEY_SIMPLEX, text_size, (0,255,0),thickness=text_th)

    fig, ax = plt.subplots()
    ax.imshow(img, aspect='equal'); plt.axis('off')
    height, width, channels = img.shape
    fig.set_size_inches(width/100.0/4.0, height/100.0/4.0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1,bottom=0,left=0,right=1,hspace=0,wspace=0)
    plt.margins(0,0)

    # plt.xticks([]); plt.yticks([]); plt.show()
    plt.savefig(os.path.join('dataset/mask_resimgs', img_path.split('/')[-1]), dpi=400); plt.close()


def object_detection_api(img_path, threshold=0.5):

    boxes, pred_cls = get_prediction('detect', img_path, threshold) # Get predictions
    img = cv2.imread(img_path) # Read image with cv2
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Convert to RGB

    # about display
    rect_th, text_size, text_th = 3, 2, 2
    bbox_color, txt_color = (0, 0, 255), (0, 255, 0)
    for i in range(len(boxes)):
      cv2.rectangle(img, boxes[i][0], boxes[i][1],color=bbox_color, thickness=rect_th) # Draw Rectangle with the coordinates
      cv2.putText(img,pred_cls[i], boxes[i][0],  cv2.FONT_HERSHEY_SIMPLEX, text_size, txt_color,thickness=text_th) # Write the prediction class

    fig, ax = plt.subplots()
    ax.imshow(img, aspect='equal'); plt.axis('off')
    height, width, channels = img.shape
    fig.set_size_inches(width/100.0/4.0, height/100.0/4.0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1,bottom=0,left=0,right=1,hspace=0,wspace=0)
    plt.margins(0,0)

    # plt.xticks([]); plt.yticks([]); plt.show()
    plt.savefig(os.path.join('dataset/det_resimgs', img_path.split('/')[-1]), dpi=400); plt.close()


if __name__ == '__main__':

    model = None
    task  = 'detect'  # [detect, segment]
    img_path = 'dataset/images/outdoor.jpg'

    if task == 'detect':
        model = fasterrcnn_resnet50_fpn()
        model.eval()
        object_detection_api(img_path, threshold=0.5)
    elif task == 'segment':
        model = maskrcnn_resnet50_fpn()
        model.eval()
        instance_segmentation_api(img_path, threshold=0.8)
    else:
        raise TypeError('Unknow task, it must be detect or segment ...')
