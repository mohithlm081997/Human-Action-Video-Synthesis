
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
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, datasets
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
from model_v2 import AutoEncoder as autoenc
from utd_mhad_dataloder import UtdMhad
from tqdm import tqdm
import torchvision


# Insert other imports


import warnings
warnings.filterwarnings("ignore")

def train(model, device, train_loader, optimizer, criterion):

    model.train()

    losses = []

    for batch_idx, batch_sample in tqdm(enumerate(train_loader)):

        input_frame, output_vid, action_id = batch_sample

        input_frame = input_frame.to(device)
        output_vid = output_vid.to(device)
        action_id = action_id.to(device)

        output = model(input_frame,action_id)

        loss = criterion(output,output_vid)

        loss.backward()

        optimizer.step()

        losses.append(loss.item())

    train_loss = float(np.mean(losses))

    print(f"Train set: Averate Loss: {train_loss}")

    return train_loss




def test(model, device, test_loader, criterion, writer_fake, writer_real):

    pass

    # model.eval()

    # losses = []

    # with torch.no_grad():

    #     for batch_idx, batch_sample in tqdm(enumerate(test_loader)):

    #         input_frame, output_vid, action_id = batch_sample

    #         input_frame = input_frame.to(device)
    #         output_vid = output_vid.to(device)
    #         action_id = action_id.to(device)

    #         output = model(input_frame,action_id)

    #         loss = criterion(output, output_vid)

    #         losses.append(loss.item())

    #     fake_img_grid = torchvision.utils.make_grid(output,normalize=True)
    #     real_img_grid = torchvision.utils.make_grid(output_vid,normalize=True)



def main(FLAGS):

    use_cuda = torch.cuda.is_available()

    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Torch device selected: {device}")

    # Insert tensorboard codes

    # Transforms
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((16,16)),
        # transforms.Resize((32,32)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,),(0.3081))
    ])

    # Fetch the dataset

    train_dataset = UtdMhad("D:\\UCF\\Coursework\\CAP5415\\course project\\vae\\frames","D:\\UCF\\Coursework\\CAP5415\\course project\\vae\\train_list.csv",transform)
    

    # Get the dataloader

    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    # Fetch the model

    model = autoenc().to(device=device)

    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(),lr = 0.03)

    step = 1

    for epoch in range(1, FLAGS.num_epochs+1):

        print(f"Epoch: {epoch}/{FLAGS.num_epochs}")

        train_loss = train(model, device, train_dataloader, optimizer, criterion)

        break

        # test_loss = test(model, device, test_loader, criterion)

    print("Training and evaluation finished :-)")



if __name__=="__main__":

    parser = argparse.ArgumentParser("Auto Encoder")

    parser.add_argument("--batch_size",type=int, default=8)
    parser.add_argument("--num_epochs",type=int, default=20)

    FLAGS = None
    FLAGS, unparsed = parser.parse_known_args()

    print(FLAGS)

    main(FLAGS)


