"""
Author: Kelvin Hong
Training MobileNetV2-SSD model.
Reference: https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection
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
import math
from math import sqrt
import random
import numpy as np
import datetime
import time

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':
    folder = "001"
    parser = argparse.ArgumentParser(description='Accept number of epoch.')
    parser.add_argument("-e", "--epoch", type=int, default=50, help='Number of epochs for training. Default 50.')
    args = parser.parse_args()
    labels = open('coco.names').read().splitlines()
    num_classes = len(labels)
    # Creating dataset
    images = []
    locs_labels = []
    # Seed to make sure sampling is reproducible
    random.seed(0)
    raw_paths = list(glob.iglob(os.path.join(folder, "*.jpg")))
    random.shuffle(raw_paths)
    n = len(raw_paths)
    i = 0
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
        i += 1
        print(i, '/', n)
    
    # Take 70% as training data
    whole_n = len(images)
    train_n = int(0.7*whole_n)
    train_images = images[:train_n]
    train_locs_labels = locs_labels[:train_n]
    test_images = images[train_n:]
    test_locs_labels = locs_labels[train_n:]

    # Configuration
    num_epochs = args.epoch
    batch_size = 4

    transform = transforms.Compose([
        transforms.Resize((300,300)),
        transforms.ColorJitter(brightness=0.5, contrast=0.5,saturation=0.5),
        transforms.ToTensor()
    ])
    train_dataset = model.CustomDataset(train_images, train_locs_labels, transform = transform)
    test_dataset = model.CustomDataset(test_images, test_locs_labels, transform=transform)
    trainloader = DataLoader(train_dataset, batch_size = batch_size, collate_fn=model.collate_fn, shuffle=True, drop_last=True)
    testloader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=model.collate_fn, shuffle=False, drop_last=True)

    # Define model and start training
    net = model.MobileNetV2_SSD(num_classes).to(device)
    net.apply(model.weights_init_rule)
    torchsummary.summary(net, (3,300,300),1,"cuda")
    optimizer = optim.Adam(net.parameters(), lr=0.005)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20,40,60,80], gamma=0.5)
    prior_boxes = net.create_prior_boxes()
    criterion = model.MultiBoxLoss(prior_boxes).to(device)
    # scheduler = 
    step = 20
    
    net.eval()
    dummy_input = torch.randn(1,3,300,300).to(device)
    onnx_model_path = f"models/mobilenetv2_ssd_{datetime.datetime.now()}.onnx"
    torch.onnx.export(net, dummy_input, onnx_model_path, verbose=False)


    loader_n = len(trainloader)
    test_loader_n = len(testloader)
    for epoch in range(num_epochs):
        # Training
        net.train()
        accu_loss = 0
        accu_conf_loss = 0
        accu_loc_loss = 0
        start = time.time()
        for i, data in enumerate(trainloader, 0):
            
            batch_images = data['images']
            batch_locs = data['locs']
            batch_labels = data['labels']
            pred_locs, pred_scores = net(batch_images)
            # print("Predicted Locations: ", pred_locs)
            # print("Predicted Scores: ", pred_scores)
            conf_loss, loc_loss, loss = criterion(pred_locs, pred_scores, batch_locs, batch_labels)
            accu_conf_loss += conf_loss.item()
            accu_loc_loss += loc_loss.item()
            accu_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if i%step == step-1:
                print(f"Epoch {epoch+1} avg loss at step {str(i+1).rjust(4)} of {str(loader_n).rjust(4)}: {round(accu_loss/i,2)}. Conf loss: {round(accu_conf_loss/i,2)}. Loc loss: {round(accu_loc_loss/i,2)}. lr={round(get_lr(optimizer),6)}.")
            # Freeing memories
            del pred_scores, pred_locs, batch_images, batch_locs, batch_labels
        end = time.time()
        te = convert_sec(end-start)
        print(f"Epoch {epoch+1} training finished. Proceed to validation. Time Elapsed {te}.")
        
        # Evaluating
        net.eval()
        taccu_loss = 0
        taccu_conf_loss = 0
        taccu_loc_loss = 0
        start = time.time()
        with torch.no_grad():
            for i, data in enumerate(testloader, 0):
                batch_images = data['images']
                batch_locs = data['locs']
                batch_labels = data['labels']
                pred_locs, pred_scores = net(batch_images)
                conf_l, loc_l, loss = criterion(pred_locs, pred_scores, batch_locs, batch_labels)
                taccu_conf_loss += conf_l.item()
                taccu_loc_loss += loc_l.item()
                taccu_loss += loss.item()
                if i%step == step-1:
                    print(f"Epoch {epoch+1} validation avg loss at step {str(i+1).rjust(4)} of {str(test_loader_n).rjust(4)}: {round(taccu_loss/i,2)}. Conf loss: {round(taccu_conf_loss/i,2)}. Loc loss: {round(taccu_loc_loss/i,2)}. lr={round(get_lr(optimizer),6)}.")
                # Freeing memories
                del pred_scores, pred_locs, batch_images, batch_locs, batch_labels
        end = time.time()
        te = convert_sec(end-start)

        scheduler.step()
        print(f"Epoch {epoch+1} validation finished. Time Elapsed {te}.")
        if (epoch+1)%10 == 0 or epoch+1==num_epochs:
            print(f"Saving Model epoch {epoch+1}...")
            torch.save({
                'epoch': epoch+1,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
            }, f'models/model_mobilenetv2_ssd_Epoch{epoch+1}_{datetime.datetime.now()}.pth')
            print(f"Saved!")
            print("You can find the model in the models/ folder.")
        print("="*50)
    


