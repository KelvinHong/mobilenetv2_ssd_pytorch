"""
Author: Kelvin Hong
Evaluating MobileNetV2-SSD model.
Using another folder from imagedb.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchsummary
import argparse
from PIL import Image
import os
import glob
from utils import *
import model
from math import sqrt
import random
import numpy as np
import datetime
import time
import cv2

def clamp_list(l):
    return [min(max(0,x),1) for x in l]

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
parser = argparse.ArgumentParser(description='Accept trained model for inferencing')
parser.add_argument("-n", "--number", type=int, default=5, help="Visualize how many images? Default 5.")
parser.add_argument("-m", "--modelpath", type=str, default=None, help='A trained model path. Default is None.')
parser.add_argument("-s", "--minscore", type=float, default=0.5, help="Minimum confidence score to be drawn. Default 0.5.")
parser.add_argument("-o", "--maxoverlap", type=float, default=0.5, help="Maximum IoU overlap for Non-maximum suppresion calculations. Default 0.5.")
args = parser.parse_args()



folder = "002"
name_labels = open('coco.names').read().splitlines()
num_classes = len(name_labels)

images = []
locs_labels = []
raw_paths = list(glob.iglob(os.path.join(folder, "*.jpg")))
random.shuffle(raw_paths)
for image_path in raw_paths:
    loc_label = []
    try:
        f = open(image_path[:-3]+"txt")
    except:
        pass
    else:
        lines = f.read().splitlines()
        images.append(image_path)
        for line in lines:
            label, *locs = line.split(' ')
            label = int(label)
            locs = list(map(float, locs))
            loc_label.append([label]+locs)
        locs_labels.append(loc_label)
try:
    checkpoint = torch.load(args.modelpath)
except:
    print("LoadModelError: The model cannot be load.")
    exit()
else:    
    # Initialize model and optimizer
    net = model.MobileNetV2_SSD(num_classes).to(device)
    optimizer = optim.Adam(net.parameters(), lr=0.005)
    # Prepare anchor boxes 
    prior_boxes = net.create_prior_boxes()
    # Loss function
    criterion = model.MultiBoxLoss(prior_boxes).to(device)
    # Load pretrained weights into model and optimizer
    net.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    # Evaluation mode
    net.eval()
    
    transform = transforms.Compose([
        transforms.Resize((300,300)),
        transforms.ToTensor()
    ])
    assert args.number>=0
    start = time.time()
    inter = start
    total = min(args.number, len(images))
    for i in range(total):
        image = Image.open(images[i])
        loc_label = locs_labels[i]
        image = transform(image).unsqueeze(0).to(device)
        pred_locs, pred_scores = net(image)
        boxes, labels, scores = net.detect_objects(pred_locs, pred_scores, min_score=args.minscore, max_overlap=args.maxoverlap, top_k = 30)
        # List of bounding boxes in one image, format xmin, ymin, xmax, ymax
        # Refer to utils.py: definition of cxcy_to_xy
        # print(boxes)
        boxes = boxes[0].tolist()
        # Lists of labels and scores of respective boxes.
        labels = labels[0].tolist()
        scores = scores[0].tolist()

        # Rendering image. Ground Truth boxes are green, Predictions are yellow. 
        cvimg = cv2.imread(images[i])
        H, W = cvimg.shape[:2]
        # Drawing ground truth boxes
        for ll in loc_label:
            label = int(ll[0])
            cx, cy, w, h = ll[1:]
            x, y, X, Y = int((cx-w/2)*W), int((cy-h/2)*H), int((cx+w/2)*W), int((cy+h/2)*H)
            cv2.rectangle(cvimg, (x,y), (X,Y), (0,255,0), 2)
        
        # Drawing prediction boxes
        for j in range(len(boxes)):
            x, y, X, Y = clamp_list(boxes[j])
            label = labels[j]
            score = scores[j]
            render_label = name_labels[label]
            if [x,y,X,Y] == [0,0,1,1]:
                render_label = "BACKGROUND"
            x, y, X, Y = int(x*W), int(y*H), int(X*W), int(Y*H)
            cv2.rectangle(cvimg, (x,y),(X,Y), (0, 255, 255), 2)
            cv2.putText(cvimg, f"{render_label}: {round(score,4)}", (x+5, y+15), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, thickness=4, color=(255,0,0))
        cv2.imwrite(f"output/image_test_{i+1}.jpg", cvimg)
        end = time.time()
        if end-inter > 5:
            progress = round(100*i/total,2)
            ETR = convert_sec((end-start)*(total-i)/i)
            inter = end
            print(f"Inferencing progress ----------{progress}%---------- ETR {ETR}")
    print(f"Inference completed on {total} images. Output images saved!")
    print("You can find the outputs in the output/ folder.")
        

        