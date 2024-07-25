

import os
import argparse
import time
import random
import sys

import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision import transforms, datasets
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
import skvideo.io
from skvideo.io import ffprobe

# Insert other imports


import warnings
warnings.filterwarnings("ignore")


class UtdMhad(Dataset):

    def __init__(self,root_dir,train_split_path,transforms):

        self.root_dir = root_dir

        self.train_split = pd.read_csv(train_split_path)["folders"]

        self.transforms = transforms

    def __len__(self):

        return len(self.train_split)
    
    def __getitem__(self, index):

        videoname = self.train_split[index]

        action_id, subject_id, trial_id, _ = videoname.split("_")

        frames_path = os.path.join(self.root_dir,videoname)

        frame_files = os.listdir(frames_path)

        frame_tensor = []

        for file in frame_files:

            frame = cv2.imread(os.path.join(frames_path,file),cv2.IMREAD_GRAYSCALE)
            frame_tensor.append(self.transforms(frame))

        clip = torch.stack(frame_tensor)

        clip = clip.squeeze(1)

        # action_vec = np.zeros(27)
        # action_vec[int(action_id[1:])-1]=1
        # action_vec = torch.tensor(action_vec)

        action_vec = F.one_hot(torch.tensor(int(action_id[1:])-1),num_classes=27)

        return clip[0].unsqueeze(0),clip,action_vec




def main():

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((32,32)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,),(0.3081))
    ])

    utd = UtdMhad("D:\\UCF\\Coursework\\CAP5415\\course project\\vae\\frames","D:\\UCF\\Coursework\\CAP5415\\course project\\vae\\train_list.csv",transform)



    for i,j,k in utd:

        print(f"{i.shape}, {j.shape} => {k.shape}")
        print(j.shape)
        break

if __name__=="__main__":

    main()








    

