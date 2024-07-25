"""
Description: 
Created on 12-06-2022 00:33:42
@author: Sirshapan
"""

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
data = pd.read_csv("data_sets.csv")

data.head()
# %%

# def f(xx):

#     action_id,train_subs =xx

#     train_paths = []

#     for x in train_subs:

#         train_path = "a"+str(action_id)+"_s"+str(x)+"_t1_color.avi"
#         train_paths.append(train_path)

#     return train_paths

    

# data["train_files"] = data["actions"]+data["train_subjects"]


action_paths = {}

for i,j in zip(data["actions"],data["train_subjects"]):

    k = j.strip("[")
    k = k.strip("]")
    k = k.strip()
    k = k.split(",")
    # print(k)

    train_paths = []

    for x in k:

        train_path = "a"+str(i)+"_s"+x.strip()+"_t1_color.avi"
        train_paths.append(train_path)

        train_path = "a"+str(i)+"_s"+x.strip()+"_t2_color.avi"
        train_paths.append(train_path)

        train_path = "a"+str(i)+"_s"+x.strip()+"_t3_color.avi"
        train_paths.append(train_path)

        train_path = "a"+str(i)+"_s"+x.strip()+"_t4_color.avi"
        train_paths.append(train_path)

    action_paths[i] = train_paths

print(train_paths)
print(action_paths)


# data["train_files"].head
# %%

import shutil

for action in action_paths:

    train_paths = action_paths[action]

    # print(action)
    # print(train_paths)

    d_name = "class_"+str(action)

    dir_name = "video_dataset_v2/"+d_name

    os.mkdir(dir_name)

    for idx,tp in enumerate(train_paths):

        print(tp)

        c_path = os.path.join("RGB",tp)

        p_name = "video"+str(idx+1)+".avi"

        p_path = os.path.join(dir_name,p_name)

        shutil.copy2(c_path,p_path)

    

    break

# %%
