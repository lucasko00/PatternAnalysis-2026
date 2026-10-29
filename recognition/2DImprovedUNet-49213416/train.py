import argparse
import os
import time
import torch
import torch.nn.functional as F
from dataset import get_dataloaders, HIPMRI_ROOT, NUM_CLASSES
from modules import ImprovedUNet2D, UNet2D


LABELS = ["background", "body", "bone", "bladder", "rectum", "prostate"]
MODELS = {"improved": ImprovedUNet2D, "baseline": UNet2D}
BATCH_SIZE = 16
N_EPOCHS = 50
LR = 5e-4

def dice_ce_loss(logits, masks):
    ce = F.cross_entropy(logits, masks)
    probs = logits.softmax(dim=1)
    one_hot = F.one_hot(masks, NUM_CLASSES).permute(0, 3, 1, 2).float()   # (B, 6, H, W)
    inter = (probs * one_hot).sum(dim=(0, 2, 3))
    total = probs.sum(dim=(0, 2, 3)) + one_hot.sum(dim=(0, 2, 3))
    return ce + 1 - ((2 * inter + 1) / (total + 1)).mean()
s

