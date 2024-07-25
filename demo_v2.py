

# %%
import os
import argparse
import time
import random
import sys

import sklearn
import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, datasets
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image

# Insert other imports

import warnings
warnings.filterwarnings("ignore")

%pylab inline
# %%
folder = "D:\\UCF\\Coursework\\CAP5415\\course project\\RGB"

lengths = []
heights = []
widths = []
# %%

import skvideo.io
from skvideo.io import ffprobe

# %%

for file in os.listdir(folder):

    file_name = os.path.join(folder,file)

    videodata = skvideo.io.vread(file_name)

    length, height, width, channel = videodata.shape

    lengths.append(length)
    heights.append(height)
    widths.append(width)
# %%

print(lengths)
print(heights)
print(widths)

# %%

action_ids = []
subject_ids = []
trial_ids = []
video_ids = []


for file in os.listdir(folder):
    
    file_name = os.path.join(folder,file)

    video_name = file.split(".")[0]

    action_id, subject_id, trial_id, _ = video_name.split("_")

    action_ids.append(action_id)
    subject_ids.append(subject_id)
    trial_ids.append(trial_id)
    video_ids.append(file)

video_data = {
    "action_ids":action_ids,
    "subject_ids":subject_ids,
    "trial_ids":trial_ids,
    "video_ids":video_ids
    }


video_df = pd.DataFrame(video_data)   


print(video_df.head)
# %%

video_df.to_csv("video_data.csv",index=None)
# %%

print(min(lengths))
# %%

from tqdm import tqdm

# %%

from skimage.io import imsave, imread

os.mkdir("frames")

frame_folder = "frames"

for file in os.listdir(folder):

    file_name = os.path.join(folder,file)

    video_name = file.split(".")[0]

    file_folder = os.path.join(frame_folder,video_name)

    os.mkdir(file_folder)

    videodata = skvideo.io.vread(file_name)

    length, height, width, channel = videodata.shape

    print(file)

    print(length, height, width, channel)

    mid_frame = length//2

    x = list(range(length))

    frame_ids = x[mid_frame-16:mid_frame:2]+x[mid_frame:mid_frame+16:2]

    counter = 0

    for id,frame in tqdm(enumerate(videodata)):

        if id in frame_ids:

            imsave(file_folder+"/frame_"+str(id).zfill(3)+"_"+str(counter).zfill(3)+".png",frame)

            counter +=1 


# %%
