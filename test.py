# -*- coding: utf-8 -*-
"""
Created on Thu Jun  2 10:52:13 2022

@author: Achintha
"""

from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils, models, datasets
from torchvision.utils import save_image
from lib import loaders, modules #, loadersUNETCGAN_f_unet - for baseline
import torch.optim as optim
from torch.optim import lr_scheduler
import time
import copy
from collections import defaultdict
import torch.nn.functional as F
import torch.nn as nn
from lib.EncoderModels import ResnetGenerator, Discriminator
from tqdm import tqdm
#############################################################
# For test evaluations 
############################################################

########################
# Load dataset         #
########################
# Setup 1 - uniform 1%
# Setup 2 - twoside 1% and 10%
# Setup 3 - nonuniform 1% to 10%

train_setup = 1         # REVISE index of training setup
test_setup = 3          # REVISE index of testing setup
setups = ['uniform', 'twoside', 'nonuniform']
setup_name = setups[train_setup-1]
test_name = setups[test_setup-1]


if test_setup == 1:          
    Radio_train = loaders.RadioUNet_s(phase="train", fix_samples=655, num_samples_low=10, num_samples_high=300)
    Radio_val = loaders.RadioUNet_s(phase="val", fix_samples=655, num_samples_low=10, num_samples_high=300)
    Radio_test = loaders.RadioUNet_s(phase="test", fix_samples=655, num_samples_low=10, num_samples_high=300)
    
elif test_setup == 2:
    Radio_train = loaders.RadioUNet_s(phase="train", fix_samples=1, num_samples_low=655, num_samples_high=655*10)
    Radio_val = loaders.RadioUNet_s(phase="val", fix_samples=1, num_samples_low=655, num_samples_high=655*10)
    Radio_test = loaders.RadioUNet_s(phase="test", fix_samples=1, num_samples_low=655, num_samples_high=655*10)

else:
    Radio_train = loaders.RadioUNet_s(phase="train", fix_samples=0, num_samples_low=655, num_samples_high=655*10)
    Radio_val = loaders.RadioUNet_s(phase="val", fix_samples=0, num_samples_low=655, num_samples_high=655*10)
    Radio_test = loaders.RadioUNet_s(phase="test", fix_samples=0, num_samples_low=655, num_samples_high=655*10)


image_datasets = {
    'train': Radio_train, 'val': Radio_val
}

batch_size = 15


dataloaders = {
    'train': DataLoader(Radio_train, batch_size=batch_size, shuffle=True, num_workers=1),
    'val': DataLoader(Radio_val, batch_size=batch_size, shuffle=True, num_workers=1)
}

def calc_loss_test(pred1, target, metrics, error="MSE"):
    criterion = nn.MSELoss()
    if error=="MSE":
        #print("here")
        loss1 = criterion(pred1, target)
        #print(loss1)
        #loss2 = criterion(pred2, target)
    else:
        loss1 = criterion(pred1, target)/criterion(target, 0*target)
        #loss2 = criterion(pred2, target)/criterion(target, 0*target)
    metrics['loss first U'] += loss1.data.cpu().numpy() * target.size(0)
    #metrics['loss second U'] += loss2.data.cpu().numpy() * target.size(0)

    return loss1

def print_metrics_test(metrics, epoch_samples, error):
    outputs = []
    if error=="MSE":
        for k in metrics.keys():
            outputs.append("{}: {:4f}".format(k, metrics[k] / (epoch_samples*256**2)))
    else:
        for k in metrics.keys():
            outputs.append("{}: {:4f}".format(k, metrics[k] / epoch_samples))
    print("{}: {}".format("Test"+" "+error, ", ".join(outputs)))
    #print('error=',outputs)


def test_loss(model, error="MSE", dataset="coarse", path=''):
    # dataset is "coarse" or "fine".
    since = time.time()
    l = []
    lm = []
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    model.to(device)
    model.eval()   # Set model to evaluate mode
    metrics = defaultdict(float)
    epoch_samples = 0
    if dataset=="coarse":
        i = 0
        for inputs, targets in DataLoader(Radio_test, batch_size=batch_size, shuffle=False, num_workers=1):
            inputs = inputs.to(device)
            targets = targets.to(device)
            up_sampled = inputs[:,3,:,:]
            inputs = inputs[:,:3,:,:]
            # do not track history if only in train
            with torch.set_grad_enabled(False):
                [outputs1,_]= model(inputs)
                loss1 = calc_loss_test(outputs1, targets, metrics, error)
                for j in range(inputs.shape[0]):
                    save_image(outputs1[j].to(torch.float)/255, os.path.join(path, f'test_out{i}{j}.png'), nrow=1, normalize=True)
                    save_image(targets[j].to(torch.float)/255, os.path.join(path, f'test_outgt{i}{j}.png'), nrow=1, normalize=True)

                epoch_samples += inputs.size(0)
            if i==3:
                break
            i += 1
    if dataset=="coarse":  
        i=0
        for inputs, targets in DataLoader(Radio_train, batch_size=batch_size, shuffle=False, num_workers=1):
            inputs = inputs.to(device)
            targets = targets.to(device)
            up_sampled = inputs[:,3,:,:]
            inputs = inputs[:,:3,:,:]
            # do not track history if only in train
            with torch.set_grad_enabled(False):
                [outputs1,_]= model(inputs)
                loss1 = calc_loss_test(outputs1, targets, metrics, error)
                epoch_samples += inputs.size(0)
                for j in range(inputs.shape[0]):
                    save_image(outputs1[j], os.path.join(path, f'train_out{i}{j}.png'), nrow=1, normalize=True)
                    save_image(targets[j], os.path.join(path, f'train_outgt{i}{j}.png'), nrow=1, normalize=True)
            if i == 3:
                break
            i +=1
    print_metrics_test(metrics, epoch_samples, error)
    #test_loss1 = metrics['loss U'] / epoch_samples
    #test_loss2 = metrics['loss W'] / epoch_samples
    time_elapsed = time.time() - since
    print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    
exp_index = 1           # REVISE index of experiment

exp_path = f"{setup_name}_{exp_index}_test_{test_name}"
os.makedirs(exp_path, exist_ok=True)

model = modules.RadioWNet(phase="firstU")

# REVISE weight path here
model.load_state_dict(torch.load('uniform_1/Trained_ModelMSE_G.pt'))
test_loss(model,error="MSE", path=exp_path)
test_loss(model,error="NMSE",path=exp_path)

# with torch.no_grad():
#     device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#     model = ResnetGenerator(input_nc=2,output_nc=1,ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6)
#     model.load_state_dict(torch.load('RadioWNet_c_DPM_Thr2_CGAN/Trained_Model_G.pt'))
#     model.to(device)
#     model.eval()
#     criterion = nn.MSELoss()
#     loss = []
#     for inps,gts in torch.utils.data.DataLoader(Radio_test,batch_size=15,shuffle=False,num_workers=2):
#         inps,gts = inps.to(device),gts.to(device)
#         R = model(inps)
#         l = criterion(R,gts)
#         loss += [l.item()]
#     print('Test Loss: ', np.mean(np.array(loss)))
