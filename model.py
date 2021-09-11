"""
Author: Kelvin Hong
Model definitions can be found in here.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchsummary
from PIL import Image
import os
import glob
from utils import *
from math import sqrt
import random
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class CustomDataset(Dataset):
    def __init__(self, images, locs_labels, transform = None):
        super().__init__()
        self.images = images
        self.locs_labels = locs_labels
        if transform:
            self.transform = transform
        else:
            self.transform = transforms.ToTensor()

    def __len__(self):
        images_n = len(self.images)
        locs_labels_n = len(self.locs_labels)
        assert images_n == locs_labels_n
        return images_n
        
    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert('RGB')
        label = [obj[0] for obj in self.locs_labels[index]]
        loc = [obj[1:] for obj in self.locs_labels[index]]
        label = torch.LongTensor(label).to(device)
        loc = torch.Tensor(loc).to(device)
        sample = {'image': self.transform(img).to(device), 'label': label, 'loc': loc}
        return sample

def collate_fn(batch):
	images = []
	locs = []
	labels = []

	for pack in batch:
		images.append(pack["image"])
		locs.append(pack["loc"])
		labels.append(pack["label"])

	return {
		"images": torch.stack(images),
		"locs": locs, 
		"labels": labels
	}
        
def SSD_miniblock(loc_or_conf, in_c, feature, num_classes=None):
    if loc_or_conf == "conf":
        if not num_classes:
            print("In 'conf' mode, please provide number of classes.")
            return 
        return nn.Sequential(
                    nn.Conv2d(in_c, 6*num_classes, 3, 1, 1),
                    nn.BatchNorm2d(6*num_classes),
                    Permute(0,2,3,1),
                    Reshape(-1, feature*feature*6, num_classes)
                )
    elif loc_or_conf == "loc":
        return nn.Sequential(
                    nn.Conv2d(in_c, 6 * 4, 3, 1, 1),
                    Permute(0,2,3,1),
                    Reshape(-1, feature*feature*6, 4)
                ) 
    else:
        return None

# Building model.
# Input is C*H*W = 3*300*300.
class BottleNeck(nn.Module):
    def __init__(self, in_c, out_c, t=1, s=1):
        super().__init__()
        self.expanded = int(in_c * t)
        self.conv1 = nn.Conv2d(in_c, self.expanded, 1)
        self.bn1 = nn.BatchNorm2d(self.expanded)
        self.conv2 = nn.Conv2d(self.expanded, self.expanded, 3, s, (s+1)//2, groups=self.expanded)
        self.bn2 = nn.BatchNorm2d(self.expanded)
        self.conv3 = nn.Conv2d(self.expanded, out_c, 1)
        self.bn3 = nn.BatchNorm2d(out_c)

        self.shortcut = nn.Sequential()
        if s!=1 or in_c!=out_c:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_c, out_c, 1, s),
                nn.BatchNorm2d(out_c)
            )

    def forward(self, x):
        out = F.relu6(self.bn1(self.conv1(x)))
        out = F.relu6(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu6(out)
        return out

class BottleNeckBlock(nn.Module):
    def __init__(self, in_c, out_c, t, n, s):
        super().__init__()
        self.first = BottleNeck(in_c, out_c, t, s)
        self.other = BottleNeck(out_c, out_c, t, 1)
        self.n = n

    def forward(self, x):
        out = self.first(x)
        for _ in range(self.n-1):
            out = self.other(out)
        return out

class Permute(nn.Module):
    def __init__(self, *dims):
        super().__init__()
        self.dims = dims

    def forward(self, x):
        x = x.permute(self.dims)
        return x

class Reshape(nn.Module):
    def __init__(self, *dims):
        super().__init__()
        self.dims = dims
    def forward(self, x):
        x = x.reshape(self.dims)
        return x

class MobileNetV2_SSD(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # Conv2d parameters: Channel Now, Channel next, kernelsize, stride, padding.
        botnec_channels = [32, 16, 24, 32, 64, 96]
        ts = [1,6,6,6,6]
        strides = [1,2,2,2,1]
        ns = [1,2,3,4,3]
        self.n_classes = num_classes
        self.conv1 = nn.Conv2d(3,32,3,2,1)
        self.batchnorm32 = nn.BatchNorm2d(32)
        self.bottlenecks = nn.Sequential(*[BottleNeckBlock(i, o, t, n, s) for (i, o, t, n, s) in zip(botnec_channels, botnec_channels[1:], ts, ns, strides)])
        self.ssd_get_feature_1 = nn.Sequential(
                                        nn.Conv2d(96,576,1),
                                        nn.BatchNorm2d(576),
                                        nn.ReLU6()
                                    )

        self.ssd_feature_1_2 = nn.Sequential(
                                        nn.Conv2d(576, 576, 3, 2, 1, groups=576),
                                        nn.BatchNorm2d(576),
                                        nn.ReLU6(),
                                        nn.Conv2d(576, 160, 1),
                                        nn.BatchNorm2d(160),
                                        BottleNeckBlock(160, 160, 6, 2, 1),
                                        BottleNeck(160, 320, 6, 1),
                                        nn.Conv2d(320, 1280, 1),
                                        nn.BatchNorm2d(1280),
                                        nn.ReLU6()
                                    )

        self.ssd_feature_2_3 = nn.Sequential(
                                    nn.Conv2d(1280, 256, 1), 
                                    nn.BatchNorm2d(256),
                                    nn.ReLU6(),
                                    nn.Conv2d(256,256,3, 2, 1, groups=256),
                                    nn.BatchNorm2d(256),
                                    nn.ReLU6(),
                                    nn.Conv2d(256,512,1),
                                    nn.BatchNorm2d(512)
                                )

        self.ssd_feature_3_4 = nn.Sequential(
                                    nn.Conv2d(512,128,1),
                                    nn.BatchNorm2d(128),
                                    nn.ReLU6(),
                                    nn.Conv2d(128,128,3,2,1,groups=128),
                                    nn.BatchNorm2d(128),
                                    nn.ReLU6(),
                                    nn.Conv2d(128,256,1),
                                    nn.BatchNorm2d(256)
                                )

        self.ssd_feature_4_5 = nn.Sequential(
                                    nn.Conv2d(256,128,1),
                                    nn.BatchNorm2d(128),
                                    nn.ReLU6(),
                                    nn.Conv2d(128,128,3,2,1,groups=128),
                                    nn.BatchNorm2d(128),
                                    nn.ReLU6(),
                                    nn.Conv2d(128,256,1),
                                    nn.BatchNorm2d(256)
                                )
        self.ssd_feature_5_6 = nn.Sequential(
                                    nn.Conv2d(256,64,1),
                                    nn.BatchNorm2d(64),
                                    nn.ReLU6(),
                                    nn.Conv2d(64,64,3,2,1,groups=64),
                                    nn.BatchNorm2d(64),
                                    nn.ReLU6(),
                                    nn.Conv2d(64,64,1),
                                    nn.BatchNorm2d(64)
                                )

        self.num_classes = num_classes
        self.process_1 = SSD_miniblock("conf", 576, 19, self.num_classes)
        self.process_2 = SSD_miniblock("conf", 1280, 10, self.num_classes)
        self.process_3 = SSD_miniblock("conf", 512, 5, self.num_classes)
        self.process_4 = SSD_miniblock("conf", 256, 3, self.num_classes)
        self.process_5 = SSD_miniblock("conf", 256, 2, self.num_classes)
        self.process_6 = SSD_miniblock("conf", 64, 1, self.num_classes)
        
        # # Refer to below commented code, to see how SSD_miniblock works
        # self.process_1 = nn.Sequential(
        #                         nn.Conv2d(576, 6*self.num_classes, 3, 1, 1),
        #                         nn.BatchNorm2d(6*self.num_classes),
        #                         Permute(0,2,3,1),
        #                         Reshape(-1, 19*19*6, self.num_classes)
        #                     )
        # self.process_2 = nn.Sequential(
        #                         nn.Conv2d(1280, 6*self.num_classes, 3, 1, 1),
        #                         nn.BatchNorm2d(6*self.num_classes),
        #                         Permute(0,2,3,1),
        #                         Reshape(-1, 10*10*6, self.num_classes)
        #                     )
        # self.process_3 = nn.Sequential(
        #                         nn.Conv2d(512, 6*self.num_classes, 3, 1, 1),
        #                         nn.BatchNorm2d(6*self.num_classes),
        #                         Permute(0,2,3,1),
        #                         Reshape(-1, 5*5*6, self.num_classes)
        #                     )
        # self.process_4 = nn.Sequential(
        #                         nn.Conv2d(256, 6*self.num_classes, 3, 1, 1),
        #                         nn.BatchNorm2d(6*self.num_classes),
        #                         Permute(0,2,3,1),
        #                         Reshape(-1, 3*3*6, self.num_classes)
        #                     )
        # self.process_5 = nn.Sequential(
        #                         nn.Conv2d(256, 6*self.num_classes, 3, 1, 1),
        #                         nn.BatchNorm2d(6*self.num_classes),
        #                         Permute(0,2,3,1),
        #                         Reshape(-1, 2*2*6, self.num_classes)
        #                     )
        # self.process_6 = nn.Sequential(
        #                         nn.Conv2d(64, 6*self.num_classes, 3, 1, 1),
        #                         nn.BatchNorm2d(6*self.num_classes),
        #                         Permute(0,2,3,1),
        #                         Reshape(-1, 1*1*6, self.num_classes)
        #                     )
        
        self.loc_1 = SSD_miniblock("loc", 576, 19)
        self.loc_2 = SSD_miniblock("loc", 1280, 10)
        self.loc_3 = SSD_miniblock("loc", 512, 5)
        self.loc_4 = SSD_miniblock("loc", 256, 3)
        self.loc_5 = SSD_miniblock("loc", 256, 2)
        self.loc_6 = SSD_miniblock("loc", 64, 1)
        # self.loc_1 = nn.Sequential(
        #                         nn.Conv2d(576, 6 * 4, 3, 1, 1),
        #                         Permute(0,2,3,1),
        #                         Reshape(-1, 19*19*6, 4)
        #                     ) 
        # self.loc_2 = nn.Sequential(
        #                         nn.Conv2d(1280, 6 * 4, 3, 1, 1),
        #                         Permute(0,2,3,1),
        #                         Reshape(-1, 10*10*6, 4)
        #                     ) 
        # self.loc_3 = nn.Sequential(
        #                         nn.Conv2d(512, 6 * 4, 3, 1, 1),
        #                         Permute(0,2,3,1),
        #                         Reshape(-1, 5*5*6, 4)
        #                     ) 
        # self.loc_4 = nn.Sequential(
        #                         nn.Conv2d(256, 6 * 4, 3, 1, 1),
        #                         Permute(0,2,3,1),
        #                         Reshape(-1, 3*3*6, 4)
        #                     ) 
        # self.loc_5 = nn.Sequential(
        #                         nn.Conv2d(256, 6 * 4, 3, 1, 1),
        #                         Permute(0,2,3,1),
        #                         Reshape(-1, 2*2*6, 4)
        #                     ) 
        # self.loc_6 = nn.Sequential(
        #                         nn.Conv2d(64, 6 * 4, 3, 1, 1),
        #                         Permute(0,2,3,1),
        #                         Reshape(-1, 1*1*6, 4)
        #                     ) 
        
        self.rescale_factors = nn.Parameter(torch.FloatTensor(1, 576, 1, 1))

        self.priors_cxcy = self.create_prior_boxes()


    def forward(self, x):
        x = F.relu6(self.batchnorm32(self.conv1(x)))
        x = self.bottlenecks(x)
        # Above part is Mobilenet V2
        # Below is SSD
        feature_1 = self.ssd_get_feature_1(x)
        feature_2 = self.ssd_feature_1_2(feature_1)
        feature_3 = self.ssd_feature_2_3(feature_2)
        feature_4 = self.ssd_feature_3_4(feature_3)
        feature_5 = self.ssd_feature_4_5(feature_4)
        feature_6 = self.ssd_feature_5_6(feature_5)
        """
        Shapes of six features: 
        576 x 19 x 19
        1280 x 10 x 10
        512 x 5 x 5
        256 x 3 x 3
        256 x 2 x 2
        64 x 1 x 1
        """
        # Below is a normalization for SSD with VGG16 backbone. 
        # I don't think we will need this normalization for MobileNetV2 backbone, so ignore it.
        if False:
            norm = feature_1.pow(2).sum(dim=1, keepdim=True).sqrt()
            feature_1 = feature_1 / norm
            feature_1 = feature_1 * self.rescale_factors
        # A list of heights of features, should be [19, 10, 5, 3, 2, 1].
        # sizes_of_features = [feature.shape[2] for feature in features]
        conf_1 = self.process_1(feature_1)
        conf_2 = self.process_2(feature_2)
        conf_3 = self.process_3(feature_3)
        conf_4 = self.process_4(feature_4)
        conf_5 = self.process_5(feature_5)
        conf_6 = self.process_6(feature_6)

        loc_1 = self.loc_1(feature_1)
        loc_2 = self.loc_2(feature_2)
        loc_3 = self.loc_3(feature_3)
        loc_4 = self.loc_4(feature_4)
        loc_5 = self.loc_5(feature_5)
        loc_6 = self.loc_6(feature_6)

        # dim 0 is batch size, so concatenate on dim 1
        locs = torch.cat([loc_1, loc_2, loc_3, loc_4, loc_5, loc_6], dim=1)
        classes_scores = torch.cat([conf_1, conf_2, conf_3, conf_4, conf_5, conf_6], dim=1)
        return locs, classes_scores

    # Prior boxes is in (cx, cy, w, h) format
    def create_prior_boxes(self):
        fmap_dims = [19, 10, 5, 3, 2, 1]
        obj_scales = [0.1, 0.2, 0.375, 0.55, 0.725, 0.9, 1]
        aspect_ratios = [[1., 2., 3., 0.5, 0.333]]*6 
        prior_boxes = []

        # use cx, cy, w, h notation
        for dim, obj_scale, obj_next, ratios in zip(fmap_dims, obj_scales, obj_scales[1:], aspect_ratios):
            for i in range(dim):
                for j in range(dim):
                    cx = (j+0.5) / dim
                    cy = (i+0.5) / dim

                    for ratio in ratios:
                        prior_boxes.append([cx, cy, obj_scale*sqrt(ratio), obj_scale/sqrt(ratio)])

                        if ratio == 1.:
                            additional_scale = sqrt(obj_scale * obj_next)
                            prior_boxes.append([cx, cy, additional_scale, additional_scale])
        
        prior_boxes = torch.FloatTensor(prior_boxes).to(device)
        prior_boxes.clamp_(0,1)

        return prior_boxes

    def detect_objects(self, predicted_locs, predicted_scores, min_score, max_overlap, top_k):
        """
        Decipher the 3000 locations and class scores (output of ths SSD300) to objects.
        For each class, perform Non-Maximum Suppression (NMS) on boxes that are above a minimum threshold.
        :param predicted_locs: predicted locations/boxes w.r.t the 3000 prior boxes, a tensor of dimensions (N, 3000, 4)
        :param predicted_scores: class scores for each of the encoded locations/boxes, a tensor of dimensions (N, 3000, n_classes)
        :param min_score: minimum threshold for a box to be considered a match for a certain class
        :param max_overlap: maximum overlap two boxes can have so that the one with the lower score is not suppressed via NMS
        :param top_k: if there are a lot of resulting detection across all classes, keep only the top 'k'
        :return: detections (boxes, labels, and scores), lists of length batch_size
        """
        batch_size = predicted_locs.size(0)
        n_priors = self.priors_cxcy.size(0)
        predicted_scores = F.softmax(predicted_scores, dim=2)  # (N, 3000, n_classes)

        # Lists to store final predicted boxes, labels, and scores for all images
        all_images_boxes = list()
        all_images_labels = list()
        all_images_scores = list()

        assert n_priors == predicted_locs.size(1) == predicted_scores.size(1)

        for i in range(batch_size):
            # Decode object coordinates from the form we regressed predicted boxes to
            decoded_locs = cxcy_to_xy(
                gcxgcy_to_cxcy(predicted_locs[i], self.priors_cxcy))  # (3000, 4), these are fractional pt. coordinates

            # Lists to store boxes and scores for this image
            image_boxes = list()
            image_labels = list()
            image_scores = list()

            max_scores, best_label = predicted_scores[i].max(dim=1)  # (3000)

            # Check for each class
            for c in range(1, self.n_classes):
                # Keep only predicted boxes and scores where scores for this class are above the minimum score
                class_scores = predicted_scores[i][:, c]  # (3000)
                score_above_min_score = class_scores > min_score  # torch.uint8 (byte) tensor, for indexing
                n_above_min_score = score_above_min_score.sum().item()
                if n_above_min_score == 0:
                    continue
                class_scores = class_scores[score_above_min_score]  # (n_qualified), n_min_score <= 3000
                class_decoded_locs = decoded_locs[score_above_min_score]  # (n_qualified, 4)

                # Sort predicted boxes and scores by scores
                class_scores, sort_ind = class_scores.sort(dim=0, descending=True)  # (n_qualified), (n_min_score)
                class_decoded_locs = class_decoded_locs[sort_ind]  # (n_min_score, 4)

                # Find the overlap between predicted boxes
                overlap = find_jaccard_overlap(class_decoded_locs, class_decoded_locs)  # (n_qualified, n_min_score)

                # Non-Maximum Suppression (NMS)

                # A torch.uint8 (byte) tensor to keep track of which predicted boxes to suppress
                # 1 implies suppress, 0 implies don't suppress
                suppress = torch.zeros((n_above_min_score)).bool().to(device)  # (n_qualified)

                # Consider each box in order of decreasing scores
                for box in range(class_decoded_locs.size(0)):
                    # If this box is already marked for suppression
                    if suppress[box] == 1:
                        continue

                    # Suppress boxes whose overlaps (with this box) are greater than maximum overlap
                    # Find such boxes and update suppress indices
                    suppress = suppress | (overlap[box] > max_overlap)
                    # The max operation retains previously suppressed boxes, like an 'OR' operation

                    # Don't suppress this box, even though it has an overlap of 1 with itself
                    suppress[box] = 0

                # Store only unsuppressed boxes for this class
                image_boxes.append(class_decoded_locs[~suppress])
                image_labels.append(torch.LongTensor((~suppress).sum().item() * [c]).to(device))
                image_scores.append(class_scores[~suppress])

            # If no object in any class is found, store a placeholder for 'background'
            if len(image_boxes) == 0:
                image_boxes.append(torch.FloatTensor([[0., 0., 1., 1.]]).to(device))
                image_labels.append(torch.LongTensor([0]).to(device))
                image_scores.append(torch.FloatTensor([0.]).to(device))

            # Concatenate into single tensors
            image_boxes = torch.cat(image_boxes, dim=0)  # (n_objects, 4)
            image_labels = torch.cat(image_labels, dim=0)  # (n_objects)
            image_scores = torch.cat(image_scores, dim=0)  # (n_objects)
            n_objects = image_scores.size(0)

            # Keep only the top k objects
            if n_objects > top_k:
                image_scores, sort_ind = image_scores.sort(dim=0, descending=True)
                image_scores = image_scores[:top_k]  # (top_k)
                image_boxes = image_boxes[sort_ind][:top_k]  # (top_k, 4)
                image_labels = image_labels[sort_ind][:top_k]  # (top_k)

            # Append to lists that store predicted boxes and scores for all images
            all_images_boxes.append(image_boxes)
            all_images_labels.append(image_labels)
            all_images_scores.append(image_scores)

        return all_images_boxes, all_images_labels, all_images_scores  # lists of length batch_size

# Customization on weights (and biases) initializations
def weights_init_rule(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        with torch.no_grad():
            m.weight.normal_(0,0.02)
            m.bias.fill_(0.5)
    if classname.find('Linear') != -1:
        with torch.no_grad():
            n = m.in_features
            y = 1.0/np.sqrt(n)
            m.weight.uniform_(-y,y)
            m.bias.fill_(1)

# Loss function for SSD. See the criterion variable in train.py
class MultiBoxLoss(nn.Module):
    def __init__(self, priors_cxcy, threshold=0.5, neg_pos_ratio=3, alpha=1.):
        super().__init__()
        self.priors_cxcy = priors_cxcy
        self.priors_xy = cxcy_to_xy(priors_cxcy)
        self.threshold = threshold
        self.neg_pos_ratio = neg_pos_ratio
        self.alpha = alpha

        self.smooth_l1 = nn.L1Loss()
        self.cross_entropy = nn.CrossEntropyLoss(reduction='none')
    
    def forward(self, predicted_locs, predicted_scores, boxes, labels):
        # boxes: a list of tensors, length of list = number of objects, each tensor has 4 entries for a box. 
        batch_size = predicted_locs.size(0)
        n_priors = self.priors_cxcy.size(0)
        n_classes = predicted_scores.size(2)
        # Boxes: Convert cxcy format to xy format
        boxes_xy = [cxcy_to_xy(objects) for objects in boxes]
        assert n_priors == predicted_locs.size(1) == predicted_scores.size(1)
        true_locs = torch.zeros((batch_size, n_priors, 4), dtype=torch.float).to(device)  # (N, 3000, 4)
        true_classes = torch.zeros((batch_size, n_priors), dtype=torch.long).to(device)  # (N, 3000)

        # For each image
        for i in range(batch_size):
            n_objects = boxes[i].size(0)
            
            overlap = find_jaccard_overlap(boxes_xy[i],self.priors_xy)# (n_objects, 3000)
            # For each prior, find the object that has the maximum overlap
            overlap_for_each_prior, object_for_each_prior = overlap.max(dim=0)  # (3000)
            # We don't want a situation where an object is not represented in our positive (non-background) priors -
            # 1. An object might not be the best object for all priors, and is therefore not in object_for_each_prior.
            # 2. All priors with the object may be assigned as background based on the threshold (0.5).

            # To remedy this -
            # First, find the prior that has the maximum overlap for each object.
            _, prior_for_each_object = overlap.max(dim=1)  # (n_objects)
            # print(prior_for_each_object.shape)

            # Then, assign each object to the corresponding maximum-overlap-prior. (This fixes 1.)
            object_for_each_prior[prior_for_each_object] = torch.LongTensor(range(n_objects)).to(device)

            # To ensure these priors qualify, artificially give them an overlap of greater than 0.5. (This fixes 2.)
            overlap_for_each_prior[prior_for_each_object] = 1.

            # Labels for each prior
            label_for_each_prior = labels[i][object_for_each_prior]  # (3000)
            # Set priors whose overlaps with objects are less than the threshold to be background (no object)
            label_for_each_prior[overlap_for_each_prior < self.threshold] = 0  # (3000)

            # Store
            true_classes[i] = label_for_each_prior # conf_loss

            # Encode center-size object coordinates into the form we regressed predicted boxes to
            # print(boxes[i][object_for_each_prior].shape)
            true_locs[i] = cxcy_to_gcxgcy(boxes[i][object_for_each_prior], self.priors_cxcy)  # (3000, 4) # loc_loss

        # Identify priors that are positive (object/non-background)
        positive_priors = true_classes != 0  # (N, 3000)

        # LOCALIZATION LOSS

        # Localization loss is computed only over positive (non-background) priors
        
        loc_loss = self.smooth_l1(predicted_locs[positive_priors], true_locs[positive_priors])  # (), scalar

        # Note: indexing with a torch.uint8 (byte) tensor flattens the tensor when indexing is across multiple dimensions (N & 3000)
        # So, if predicted_locs has the shape (N, 3000, 4), predicted_locs[positive_priors] will have (total positives, 4)

        # CONFIDENCE LOSS

        # Confidence loss is computed over positive priors and the most difficult (hardest) negative priors in each image
        # That is, FOR EACH IMAGE,
        # we will take the hardest (neg_pos_ratio * n_positives) negative priors, i.e where there is maximum loss
        # This is called Hard Negative Mining - it concentrates on hardest negatives in each image, and also minimizes pos/neg imbalance

        # Number of positive and hard-negative priors per image
        n_positives = positive_priors.sum(dim=1)  # (N)
        n_hard_negatives = self.neg_pos_ratio * n_positives  # (N)

        # First, find the loss for all priors
        conf_loss_all = self.cross_entropy(predicted_scores.view(-1, n_classes), true_classes.view(-1))  # (N * 3000)
        conf_loss_all = conf_loss_all.view(batch_size, n_priors)  # (N, 3000)

        # We already know which priors are positive
        conf_loss_pos = conf_loss_all[positive_priors]  # (sum(n_positives))

        # Next, find which priors are hard-negative
        # To do this, sort ONLY negative priors in each image in order of decreasing loss and take top n_hard_negatives
        conf_loss_neg = conf_loss_all.clone()  # (N, 3000)
        conf_loss_neg[positive_priors] = 0.  # (N, 3000), positive priors are ignored (never in top n_hard_negatives)
        conf_loss_neg, _ = conf_loss_neg.sort(dim=1, descending=True)  # (N, 3000), sorted by decreasing hardness
        hardness_ranks = torch.LongTensor(range(n_priors)).unsqueeze(0).expand_as(conf_loss_neg).to(device)  # (N, 3000)
        hard_negatives = hardness_ranks < n_hard_negatives.unsqueeze(1)  # (N, 3000)
        conf_loss_hard_neg = conf_loss_neg[hard_negatives]  # (sum(n_hard_negatives))

        # As in the paper, averaged over positive priors only, although computed over both positive and hard-negative priors
        conf_loss = (conf_loss_hard_neg.sum() + conf_loss_pos.sum()) / n_positives.sum().float()  # (), scalar

        # TOTAL LOSS
        return conf_loss, loc_loss, conf_loss + self.alpha * loc_loss

