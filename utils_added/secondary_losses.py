import torch
import cv2
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import torchvision.models as models

from typing import List, cast

from utils_added.utils_bloss import simplex, probs2one_hot, one_hot
from utils_added.utils_bloss import one_hot2hd_dist
from torch import Tensor, einsum
import random


# Sobel filter
def sobel_filter(device):
    # Give Sobel filters for x and y direction
    sobel_x = torch.tensor([[1, 0, -1],
                            [2, 0, -2],
                            [1, 0, -1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    sobel_y = torch.tensor([[1, 2, 1],
                            [0, 0, 0],
                            [-1, -2, -1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    return sobel_x, sobel_y


# Sobel edges for obtaining magnitudes
def sobel_edges(mask):
    # Obtain Sobel filters
    filter_x, filter_y = sobel_filter(mask.device)

    # If mask in higher than 4 dimensionality put it in 4 dimensionality
    while len(mask.shape) > 4:
        mask = mask.squeeze(0)

    # If mask in lower than 4 dimensionality put it in 4 dimensionality
    while len(mask.shape) < 4:
        mask = mask.unsqueeze(0)

    # Convolve over the mask in x and y directions
    Gx = F.conv2d(mask, filter_x, padding=1)
    Gy = F.conv2d(mask, filter_y, padding=1)

    if torch.isnan(Gx).any():
        print("NaN in scaled Gx")
    
    if torch.isnan(Gy).any():
        print("NaN in scaled Gy")

    # Calculate magnitude
    magnitude = torch.sqrt(torch.square(Gx) + torch.square(Gy))

    if torch.isnan(magnitude).any():
        print("NaN in scaled magnitude pre scaling")

    # Scale magnitude values to be between 0 and 1 (NaNs appear before this operation already)
    magnitude = magnitude

    return magnitude


# Method for transforming logits using Sobel mask
def transform_logits_sobel(logits, skip=False):
    # 4 Dimension format for sobel filter
    if not skip:
        if len(logits.shape) == 3:
            logits = logits.unsqueeze(0)
        if torch.isnan(logits).any():
            print("NaN in unscaled logits")

        # Transform logits into probabilities
        scaled_logits = F.softmax(logits, dim=0)
    else:
        scaled_logits = logits

    if torch.isnan(scaled_logits).any():
        print("NaN in scaled logits")

    # Reshape to have Class amount of single channel images.
    shapes = scaled_logits.shape
    scaled_logits = torch.reshape(scaled_logits, (shapes[1], shapes[0], shapes[2], shapes[3]))

    # Calculate magnitudes
    magnitudes = sobel_edges(scaled_logits)

    # Check if NaN values obtained
    if torch.isnan(magnitudes).any():
        print("NaN in magnitude")

    # Linear combination of logits and magnitudes to single channel "mask"
    logit_edges = scaled_logits * magnitudes
    logit_mask = torch.logsumexp(logit_edges, dim=0)

    return logit_mask


# Cross entropy loss for Boundary loss method
class CrossEntropy():
    def __init__(self, **kwargs):
        # Self.idc is used to filter out some classes of the target mask. Use fancy indexing
        self.idc: List[int] = kwargs["idc"]
        print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def __call__(self, probs: Tensor, target: Tensor) -> Tensor:
        assert simplex(probs) and simplex(target)

        log_p: Tensor = (probs[:, self.idc, ...] + 1e-10).log()
        mask: Tensor = cast(Tensor, target[:, self.idc, ...].type(torch.float32))

        loss = - einsum("bkwh,bkwh->", mask, log_p)
        loss /= mask.sum() + 1e-10

        return loss



class GeneralizedDice():
    def __init__(self, **kwargs):
        # Self.idc is used to filter out some classes of the target mask. Use fancy indexing
        self.idc: List[int] = kwargs["idc"]
        print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def __call__(self, probs: Tensor, target: Tensor) -> Tensor:
        assert simplex(probs) and simplex(target)

        pc = probs[:, self.idc, ...].type(torch.float32)
        tc = target[:, self.idc, ...].type(torch.float32)

        w: Tensor = 1 / ((einsum("bkwh->bk", tc).type(torch.float32) + 1e-10) ** 2)
        intersection: Tensor = w * einsum("bkwh,bkwh->bk", pc, tc)
        union: Tensor = w * (einsum("bkwh->bk", pc) + einsum("bkwh->bk", tc))

        divided: Tensor = 1 - 2 * (einsum("bk->b", intersection) + 1e-10) / (einsum("bk->b", union) + 1e-10)

        loss = divided.mean()

        return loss


class DiceLoss():
    def __init__(self, **kwargs):
        # Self.idc is used to filter out some classes of the target mask. Use fancy indexing
        self.idc: List[int] = kwargs["idc"]
        print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def __call__(self, probs: Tensor, target: Tensor) -> Tensor:
        assert simplex(probs) and simplex(target)

        pc = probs[:, self.idc, ...].type(torch.float32)
        tc = target[:, self.idc, ...].type(torch.float32)

        intersection: Tensor = einsum("bcwh,bcwh->bc", pc, tc)
        union: Tensor = (einsum("bkwh->bk", pc) + einsum("bkwh->bk", tc))

        divided: Tensor = torch.ones_like(intersection) - (2 * intersection + 1e-10) / (union + 1e-10)

        loss = divided.mean()

        return loss


# Surface loss for boundary loss
class SurfaceLoss():
    def __init__(self, **kwargs):
        # Self.idc is used to filter out some classes of the target mask. Use fancy indexing
        self.idc: List[int] = kwargs["idc"]
        print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def __call__(self, probs: Tensor, dist_maps: Tensor) -> Tensor:
        assert simplex(probs)
        assert not one_hot(dist_maps)

        pc = probs[:, self.idc, ...].type(torch.float32)
        dc = dist_maps[:, self.idc, ...].type(torch.float32)

        multipled = einsum("bkwh,bkwh->bkwh", pc, dc)

        loss = multipled.mean()

        return loss


BoundaryLoss = SurfaceLoss


class HausdorffLoss():
    """
    Implementation heavily inspired from https://github.com/JunMa11/SegWithDistMap
    """
    def __init__(self, **kwargs):
        # Self.idc is used to filter out some classes of the target mask. Use fancy indexing
        self.idc: List[int] = kwargs["idc"]
        print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def __call__(self, probs: Tensor, target: Tensor) -> Tensor:
        assert simplex(probs)
        assert simplex(target)
        assert probs.shape == target.shape

        B, K, *xyz = probs.shape  # type: ignore

        pc = cast(Tensor, probs[:, self.idc, ...].type(torch.float32))
        tc = cast(Tensor, target[:, self.idc, ...].type(torch.float32))
        assert pc.shape == tc.shape == (B, len(self.idc), *xyz)

        target_dm_npy: np.ndarray = np.stack([one_hot2hd_dist(tc[b].cpu().detach().numpy())
                                              for b in range(B)], axis=0)
        assert target_dm_npy.shape == tc.shape == pc.shape
        tdm: Tensor = torch.tensor(target_dm_npy, device=probs.device, dtype=torch.float32)

        pred_segmentation: Tensor = probs2one_hot(probs).cpu().detach()
        pred_dm_npy: np.nparray = np.stack([one_hot2hd_dist(pred_segmentation[b, self.idc, ...].numpy())
                                            for b in range(B)], axis=0)
        assert pred_dm_npy.shape == tc.shape == pc.shape
        pdm: Tensor = torch.tensor(pred_dm_npy, device=probs.device, dtype=torch.float32)

        delta = (pc - tc)**2
        dtm = tdm**2 + pdm**2

        multipled = einsum("bkwh,bkwh->bkwh", delta, dtm)

        loss = multipled.mean()

        return loss


class FocalLoss():
    def __init__(self, **kwargs):
        # Self.idc is used to filter out some classes of the target mask. Use fancy indexing
        self.idc: List[int] = kwargs["idc"]
        self.gamma: float = kwargs["gamma"]
        print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def __call__(self, probs: Tensor, target: Tensor) -> Tensor:
        assert simplex(probs) and simplex(target)

        masked_probs: Tensor = probs[:, self.idc, ...]
        log_p: Tensor = (masked_probs + 1e-10).log()
        mask: Tensor = cast(Tensor, target[:, self.idc, ...].type(torch.float32))

        w: Tensor = (1 - masked_probs)**self.gamma
        loss = - einsum("bkwh,bkwh,bkwh->", w, mask, log_p)
        loss /= mask.sum() + 1e-10

        return loss
    

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin=margin
    
    def forward(self, out1, out2, label):
        euclidian_dist = F.pairwise_distance(out1, out2)

        loss = torch.mean((label) * torch.pow(euclidian_dist, 2) + (1 - label) * torch.pow(torch.clamp(self.margin - euclidian_dist, min=0.0), 2))

        return loss


# Point selection method, selects the most and least confident points from our ground truth labels
def select_difficult_points(logits, gt_labels):
    if len(logits.shape) == 3:
        logits = logits.unsqueeze(0)
    _, _, H, W = logits.shape
    H2, W2 = gt_labels.shape
    if H != H2 or W != W2:
        print("Mismatch in dimension sizes logits and labels")
        print(H)
        print(H2)
        print(W)
        print(W2)

    points_base = []
    points_compare = []
    classes = []

    scaled_logits = F.softmax(logits, dim=1)

    labels = torch.unique(gt_labels)

    for label in labels:
        coordinates = gt_labels==label
        logit_image = scaled_logits[0][label] * coordinates
        x_max = torch.max(torch.max(logit_image, -2).values, 0).indices
        y_max = torch.max(torch.max(logit_image, -1).values, 0).indices

        label_1 = gt_labels[y_max][x_max]


        invalid = coordinates==0
        invalid = invalid * 2

        vals = logit_image + invalid

        x_min = torch.min(torch.min(vals, -2).values, 0).indices

        y_min = torch.min(torch.min(vals, -1).values, 0).indices


        label_2 = gt_labels[y_min][x_min]

        if x_max != x_min or y_max != y_min:
            points_base.append((y_max.item(), x_max.item()))
            points_compare.append((y_min.item(), x_min.item()))
            classes.append(label_1 == label_2)
        else:
          print("Skipping this point")
          print(label)
          break

    return points_base, points_compare, classes
  

# Point selection method, selects random points and random points to compare to
def select_random_points(logits, gt_labels):
    # print(gt_labels.shape)
    if len(logits.shape) == 3:
        logits = logits.unsqueeze(0)
    _, _, H, W = logits.shape
    H2, W2 = gt_labels.shape
    if H != H2 or W != W2:
        print("Mismatch in dimension sizes logits and labels")
        print(H)
        print(H2)
        print(W)
        print(W2)

    points_base = []
    points_compare = []
    classes = []

    scaled_logits = F.softmax(logits, dim=1)
    pred_obj = torch.max(scaled_logits,dim=1)

    pred_obj_mask = pred_obj.indices.squeeze(0)
    points_selected = 0

    while points_selected < 24:
        y_coords = random.randint(0, H-1)
        x_coords = random.randint(0, W-1)

        label_1 = gt_labels[y_coords][x_coords]

        if (y_coords, x_coords) not in points_base:
            selected_class = pred_obj_mask[y_coords][x_coords].item()

            pred_obj_mask[y_coords][x_coords] = -1

            coordinates = pred_obj_mask==selected_class
            coordinates = pred_obj_mask!=0

            pair = (coordinates == 1).nonzero(as_tuple=True)

            if len(pair[0]) > 0:
                pair_choice = random.randint(0, len(pair[0]) - 1)

                y_compares = pair[0][pair_choice]
                x_compares = pair[1][pair_choice]

                label_2 = gt_labels[y_compares][x_compares]

                points_base.append((y_coords, x_coords))
                points_compare.append((y_compares.item(), x_compares.item()))
                classes.append(label_1 == label_2)

                points_selected += 1

    return points_base, points_compare, classes


# Chamfer loss, unused in final implementation
def chamfer_loss(pred_bound, target_bound):
    # Assure no loss returned in case of nan
    if not torch.is_tensor(pred_bound) or not torch.is_tensor(target_bound):
        return None
    if len(pred_bound.shape) < 2 or len(target_bound).shape < 2:
        return None
    pred_to_target = torch.cdist(pred_bound, target_bound).min(dim=1)[0]
    target_to_pred = torch.cdist(target_bound, pred_bound).min(dim=1)[0]

    loss = pred_to_target.mean() + target_to_pred.mean()

    return loss


# Sobel loss function, unused in final implementation
def sobel_loss(predictions, gt_mask):
    pred_mag = sobel_edges(predictions)
    target_mag = sobel_edges(gt_mask)

    loss = F.mse_loss(pred_mag, target_mag)
    
    return loss


# Perceptual loss, unused in final implementation
class PerceptualLoss(torch.nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        vgg = models.vgg16(pretrained=True).features
        self.slice1 = nn.Sequential(*list(vgg)[:4])
        # print("SLICE1")
        # print(self.slice1)
        self.slice2 = nn.Sequential(*list(vgg)[4:9])
        self.slice3 = nn.Sequential(*list(vgg)[9:16])
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, pred, target):
        pred_feat = self.slice1(pred)
        target_feat = self.slice1(target)
        loss = F.mse_loss(pred_feat, target_feat)
        return loss


# Method for transforming logits to mask, unused in final implementation
def transform_logits(logits):
    # Transform logits into probabilities
    scaled_logits = F.softmax(logits)

    edge_set = []

    for logit in scaled_logits:
        logit_mask = logit.clone().detach().cpu().numpy()
        edges = cv2.Canny((logit_mask*255).astype(np.uint8),100,200) // 255
        edge_set.append(edges)
    
    edge_set = torch.from_numpy(np.array(edge_set)).to(logits.device)

    assert edge_set.shape == scaled_logits.shape, "Shape mismatch, edge set is of shape " + str(edge_set.shape) + ", while logits are of shape " * str(scaled_logits.shape) + "."

    logit_edges = scaled_logits * torch.from_numpy(np.array(edge_set))

    logit_mask = torch.sum(logit_edges, dim=-3)


    return logit_mask

# Filter class to train a filter similar in function to Sobel filter, unused in final implementation
class Filter(nn.Module):
    def __init__(self):
        super(Filter, self).__init__()
        self.conv = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        x1 = self.conv(x)
        x2 = self.conv2(x)
        x3 = torch.sqrt(torch.square(x1) + torch.square(x2))
        return x3


# Transform the logits for the learnable filter, unused in final implementation
def transform_logits_learnable(logits, filter):
    if len(logits.shape) == 3:
        logits = logits.unsqueeze(0)
    scaled_logits = F.softmax(logits, dim=0)

    # if len(scaled_logits.shape) == 4:
    shapes = scaled_logits.shape
    scaled_logits = torch.reshape(scaled_logits, (shapes[1], shapes[0], shapes[2], shapes[3]))


    magnitudes = filter(scaled_logits)


    logit_edges = scaled_logits * magnitudes

    logit_mask = torch.logsumexp(logit_edges, dim=-4)


    return logit_mask