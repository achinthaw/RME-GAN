"""
Created on Mon Jun 20 13:59:34 2022

trying loss2 > loss 1 : done
Thanks: https://www.youtube.com/watch?v=Snqb7Usrauw
Thanks: https://github.com/RonLevie/RadioUNet


This code uses non uniform sampled observations in a range 1-10%
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
from lib import loadersUNETCGAN_f, modulesUNETCGAN1, losses  # cuda1
import torch.optim as optim
from torch.optim import lr_scheduler
import time
import copy
from collections import defaultdict
import torch.nn.functional as F
import torch.nn as nn
from EncoderModels import ResnetGenerator, Discriminator
from tqdm import tqdm
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
import scipy.stats as stats

TV_WEIGHT = 1e-7


def tvloss(y_hat):
    """ Total variation loss"""
    diff_i = torch.sum(torch.abs(y_hat[:, :, :, 1:] - y_hat[:, :, :, :-1]))
    diff_j = torch.sum(torch.abs(y_hat[:, :, 1:, :] - y_hat[:, :, :-1, :]))
    tv_loss = TV_WEIGHT * (diff_i + diff_j)
    return tv_loss


def gradient_img(img, device):
    """Gradient loss"""
    img = img.squeeze(0)
    a = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    conv1 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
    conv1.weight = nn.Parameter(torch.from_numpy(a).float().unsqueeze(0).unsqueeze(0).to(device))
    G_x = conv1(img)
    b = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    conv2 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
    conv2.weight = nn.Parameter(torch.from_numpy(b).float().unsqueeze(0).unsqueeze(0).to(device))
    G_y = conv2(img)

    c = np.array([[-2, -1, -0], [-1, 0, 1], [0, 1, 2]])
    conv3 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
    conv3.weight = nn.Parameter(torch.from_numpy(c).float().unsqueeze(0).unsqueeze(0).to(device))
    G_xy = conv3(img)  # conv1(Variable(x)).data.view(1,x.shape[2],x.shape[3])

    d = np.array([[0, 1, 2], [-1, 0, 1], [-2, -1, 0]])
    conv4 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
    conv4.weight = nn.Parameter(torch.from_numpy(d).float().unsqueeze(0).unsqueeze(0).to(device))
    G_yx = conv4(img)  # (Variable(x)).data.view(1,x.shape[2],x.shape[3])

    G = torch.cat([G_x, G_y, G_xy, G_yx], dim=1)
    # G = torch.cat([G_x,G_y,G_xy,G_yx],dim=1)

    # G=torch.sqrt(torch.pow(G_x,2)+ torch.pow(G_y,2))
    return G


def power_loss(images, device, E=100):
    """Power loss"""
    # print('images: ',images.shape)
    image = images.detach().cpu().numpy().astype(int)
    npix = images.shape[1]
    fourier_image = np.fft.fftn(image)
    fourier_amplitudes = np.abs(fourier_image) ** 2

    kfreq = np.fft.fftfreq(npix) * npix

    kfreq2D = np.meshgrid(kfreq, kfreq)
    knrm = np.sqrt(kfreq2D[0] ** 2 + kfreq2D[1] ** 2)

    knrm = knrm.flatten()
    fourier_amplitudes = fourier_amplitudes.reshape(images.shape[0], -1)
    kbins = np.arange(0.5, npix // 2 + 1, 1.)
    kvals = 0.5 * (kbins[1:] + kbins[:-1])
    Abins, _, _ = stats.binned_statistic(knrm, fourier_amplitudes,
                                         statistic="mean",
                                         bins=kbins)
    Abins *= np.pi * (kbins[1:] ** 2 - kbins[:-1] ** 2)
    ind = np.argpartition(Abins, E)
    return torch.FloatTensor(ind).to(device)


def calc_loss_dense(pred, target, metrics):
    """MSE for dense observations"""
    criterion = nn.MSELoss()
    loss = criterion(pred, target)
    metrics['loss'] += loss.data.cpu().numpy() * target.size(0)

    return loss


def calc_loss_sparse(pred, target, samples, metrics, num_samples):
    """MSE for sparse"""
    criterion = nn.MSELoss()
    loss = criterion(samples * pred, samples * target) * (256 ** 2) / num_samples
    metrics['loss'] += loss.data.cpu().numpy() * target.size(0)

    return loss


def print_metrics(metrics, epoch_samples, phase):
    outputs1 = []
    for k in metrics.keys():
        outputs1.append("{}: {:4f}".format(k, metrics[k] / epoch_samples))

    print("{}: {}".format(phase, ", ".join(outputs1)))


def concat_vectors(x, y):
    combined = torch.cat((x.float(), y.float()), 1)
    return combined


def ohe_vector_from_labels(labels, n_classes):
    """one hot encoding"""
    return F.one_hot(labels, num_classes=n_classes)


class GANTrain():
    """GAN trainer classs"""
    def __init__(self, netD, netG, trainset, valset=[], testset=[], phase='first', batchsize=30):
        self.cuda = torch.cuda.is_available()
        self.device = torch.device('cuda:0' if self.cuda else 'cpu')
        self.batch_sz = batchsize
        self.n_segments = 50  # 100
        self.c = 50  # 100
        self.phase = phase
        self.trainset = trainset
        self.testset = testset
        self.valset = valset
        self.lossD = nn.BCEWithLogitsLoss()
        self.lossG = nn.MSELoss()  # nn.L1Loss()#nn.MSELoss()
        self.lossGS = nn.CosineSimilarity(dim=1, eps=1e-08)
        self.lossMsSSIM = losses.MS_SSIM_L1_LOSS()  # lossescuda1.MS_SSIM_L1_LOSS()
        self.lossL1 = nn.MSELoss()  # nn.L1Loss()
        self.netG = netG.to(
            self.device)  # ResnetGenerator(input_nc=2,output_nc=1,ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6).to(self.device)
        self.netD = netD.to(self.device)  # Discriminator(self.device).to(self.device)
        self.optimG = torch.optim.Adam(self.netG.parameters(), lr=0.0001, betas=(0.9, 0.999))
        self.optimD = torch.optim.Adam(self.netD.parameters(), lr=0.0001, betas=(0.9, 0.999))
        self.train_loader = torch.utils.data.DataLoader(self.trainset, batch_size=self.batch_sz, shuffle=False,
                                                        num_workers=2)
        self.test_loader = torch.utils.data.DataLoader(self.testset, batch_size=self.batch_sz, shuffle=False,
                                                       num_workers=2)
        self.val_loader = torch.utils.data.DataLoader(self.valset, batch_size=self.batch_sz, shuffle=False,
                                                      num_workers=2)

    def model_train(self, epochs=10):
        self.lossDL = []
        self.lossGL = []
        self.lossMSE = []
        self.lossDL_m = []
        self.lossGL_m = []
        self.lossMSE_m = []
        self.val_loss_m = []
        best_model_wts_G = copy.deepcopy(self.netG.state_dict())
        best_model_wts_D = copy.deepcopy(self.netD.state_dict())
        best_loss = 22
        # Dense training
        for epoch in tqdm(range(epochs), unit='epochs'):
            if epoch % 20 == 0:
                for g in self.optimG.param_groups:
                    g['lr'] = g['lr'] / 10
                for d in self.optimD.param_groups:
                    d['lr'] = d['lr'] / 10
            print('learning rate gen:', g['lr'], ' learning rate dis:', d['lr'])
            for number, (inps, gts) in enumerate(self.train_loader):
                inps, gts = inps.to(self.device), gts.to(self.device)
                up_sampled = inps[:, 3, :, :].unsqueeze(1)
                inps = inps[:, :3, :, :]

                self.netG.train()
                self.netD.train()

                ############################
                # Train the Discriminator  #
                ############################

                one_hot_labelsF = ohe_vector_from_labels(torch.tensor([0] * self.batch_sz).to(self.device), 2)
                one_hot_labelsR = ohe_vector_from_labels(torch.tensor([1] * self.batch_sz).to(self.device), 2)

                image_one_hot_labelsF = one_hot_labelsF[:, :, None, None]
                image_one_hot_labelsR = one_hot_labelsR[:, :, None, None]


                self.netD.zero_grad()
                self.optimD.zero_grad()

                image_one_hot_labelsF = image_one_hot_labelsF.repeat(1, 1, inps.shape[2], inps.shape[3])
                image_one_hot_labelsR = image_one_hot_labelsR.repeat(1, 1, inps.shape[2], inps.shape[3])


                [fake, _] = self.netG(inps)
                fake_image_and_labels = concat_vectors(fake, image_one_hot_labelsF)
                real_image_and_labels = concat_vectors(gts, image_one_hot_labelsR)

                predD_fake = self.netD(fake_image_and_labels.detach())
                predD_real = self.netD(real_image_and_labels)

                lossD_fake = self.lossD(predD_fake, torch.zeros_like(predD_fake))
                lossD_real = self.lossD(predD_real, torch.zeros_like(predD_real))

                lossDT = (lossD_fake + lossD_real) / 2

                lossDT.backward(retain_graph=True)
                self.optimD.step()

                self.lossDL += [lossDT.data.cpu().numpy() / inps.shape[0]]


                #####################
                # Train Generator   #
                #####################

                self.netG.zero_grad()
                self.optimG.zero_grad()

                [fake, _] = self.netG(inps)

                fake_image_and_labels = concat_vectors(fake, image_one_hot_labelsR)
                predD_fake = self.netD(fake_image_and_labels)


                if self.phase == 'first':
                    lossG = self.lossD(predD_fake, torch.ones_like(predD_fake))
                    lossM = self.lossG(fake, gts)
                    g_fake = gradient_img(fake, self.device)
                    g_fake = torch.nan_to_num(g_fake)  # ,nan=0,posinf=100,neginf=100)
                    g_up_sampled = gradient_img(up_sampled, self.device)
                    g_up_sampled = torch.nan_to_num(g_up_sampled)  # ,nan=0,posinf=100,neginf=100)
                    #lossGS = self.lossGS(g_up_sampled.double(), g_fake.double())
                    lossGS = self.lossGS(g_up_sampled, g_fake)
                    lossGS = 1 - torch.mean(lossGS)
                    lossTV = tvloss(fake)
                    # print('Lg: ',lossG)
                    # print('lm: ',lossM)
                    # print('lGS: ',lossGS)
                    # print('ltv: ',lossTV)

                    lossT = 10 * lossG + 100 * lossM + 0.01 * lossTV + 0.1 * lossGS  # + lossTV

                else:

                    lossG = self.lossD(predD_fake, torch.ones_like(predD_fake))
                    # print('lossG: ',lossG)
                    lossSSIM = self.lossMsSSIM(fake, gts)
                    lossL1 = self.lossG(fake, gts)

                    lossE = self.lossL1(power_loss(up_sampled.squeeze(1), self.device),
                                        power_loss(fake.squeeze(1), self.device))
                    # print('lossE: ',lossE)
                    lossTV = tvloss(fake)
                    ##print('lossTV: ',lossTV)
                    # I donot get the super pixels of fake only the x sparse
                    segmentsi = torch.zeros(self.batch_sz, self.c)
                    segmentf = torch.zeros(self.batch_sz, self.c)
                    for i in range(self.batch_sz):
                        inps = inps.detach().cpu()
                        si = slic(inps[i, 2, :, :], n_segments=self.n_segments, sigma=5)
                        for n, j in enumerate(np.unique(si)):
                            x, y = np.where(si == j)
                            xm, ym = np.unravel_index(np.argmax(inps[i, 2, :, :][x, y], axis=None),
                                                      inps[i, 2, :, :].shape)
                            segmentsi[i][n] = inps[i, 2, :, :][xm, ym]
                            # print('shape: ',inps[i,2,:,:][xm,ym].shape,fake[i,0,xm,ym].shape,fake[i].shape)
                            segmentf[i][n] = fake[i, 0, xm, ym]
                    lossC = self.lossL1(segmentsi.to(self.device), segmentf.to(self.device))
                    # print('lossC: ',lossC)
                    # sf = slic(fake[i], n_segments = self.n_segments, sigma = 5)

                    # lossT =  lossG + lossL1 + 0.0001*lossTV + 0.1*lossSSIM + 0.01*lossE + 0.01*lossC #+ lossC #10*lossL1#+  lossE +  lossTV + lossC
                    lossT = 10 * lossG + 10 * lossL1 + 0.001 * lossTV + 0.1 * lossSSIM + 0.01 * lossE + 0.1 * lossC


                lossT.backward()

                self.optimG.step()

                self.lossGL += [lossT.data.cpu().numpy() / inps.shape[0]]
                with torch.no_grad():
                    loss = self.lossG(fake, gts)
                    self.lossMSE += [loss.data.cpu().numpy()]
                    if number % 100 == 0:
                        print(' epoch : ', epoch, ' number: ', number, ' MSE: ', np.mean(self.lossMSE))
                # print("Generator loss: ",lossG.item())
            #     break
            # break
            # print(lossT)
            print("mean D loss: ", np.mean(np.array(self.lossDL)))
            print("mean G loss: ", np.mean(np.array(self.lossGL)))
            print("mean MSE loss: ", np.mean(np.array(self.lossMSE)))
            self.lossDL_m += [np.mean(np.array(self.lossDL))]
            self.lossGL_m += [np.mean(np.array(self.lossGL))]
            self.lossMSE_m += [np.mean(np.array(self.lossMSE))]

            with torch.no_grad():
                self.netD.eval()
                self.netG.eval()
                self.val_loss = []
                for inps, gts in self.val_loader:
                    inps, gts = inps.to(self.device), gts.to(self.device)
                    up_sampled = inps[:, 3, :, :]
                    inps = inps[:, :3, :, :]
                    [fake, _] = self.netG(inps)
                    v_loss = self.lossG(fake, gts)
                    self.val_loss += [v_loss.item()]
                val = np.mean(np.array(self.val_loss))
                self.val_loss_m += [val]
                print('val loss:', np.mean(np.array(self.val_loss)))
                i = 2
                if val < best_loss:
                    best_loss = val
                    print("saving best model")
                    print("val MSE: ", val)
                    best_model_wts_D = copy.deepcopy(self.netD.state_dict())
                    best_model_wts_G = copy.deepcopy(self.netG.state_dict())
                    G_file = 'UNetCGANFinal_nonuniform/Trained_ModelMSE_Gen_best_' + str(i) + '.wgt'
                    D_file = 'UNetCGANFinal_nonuniform/Trained_ModelMSE_Dis_best_' + str(i) + '.wgt'
                    torch.save(self.netG.state_dict(), G_file)
                    torch.save(self.netD.state_dict(), D_file)
                    i += 1
        return best_model_wts_D, best_model_wts_G



########################
# Load dataset         #
########################
Radio_train = loadersUNETCGAN_f.RadioUNet_s(phase="train")
Radio_val = loadersUNETCGAN_f.RadioUNet_s(phase="val")
Radio_test = loadersUNETCGAN_f.RadioUNet_s(phase="test")

image_datasets = {
    'train': Radio_train, 'val': Radio_val
}

batch_size = 30

dataloaders = {
    'train': DataLoader(Radio_train, batch_size=batch_size, shuffle=True, num_workers=1),
    'val': DataLoader(Radio_val, batch_size=batch_size, shuffle=True, num_workers=1)
}

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

torch.set_default_dtype(torch.float32)
torch.set_default_tensor_type('torch.cuda.FloatTensor')
torch.backends.cudnn.enabled
# netG = ResnetGenerator(input_nc=2,output_nc=1,ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6)
netG = modulesUNETCGAN1.RadioWNet(phase="firstU")  # .to(device)
netG.load_state_dict(torch.load('UNetCGANFinal_nonuniform/Trained_ModelMSE_Gen_best_0.wgt'))
netD = Discriminator(device)
netD.load_state_dict(torch.load('UNetCGANFinal_nonuniform/Trained_ModelMSE_Dis_best_0.wgt'))

RadioGAN = GANTrain(netD, netG, Radio_train, Radio_val, Radio_test, phase='second')
bestD, bestG = RadioGAN.model_train(epochs=50)

np.savetxt('MSEofUNetCGANFinal_nonuniformlnew2.csv', RadioGAN.lossMSE_m, delimiter=",")
np.savetxt('valofUNetCGANFinal_nonuniformlnew2.csv', RadioGAN.val_loss_m, delimiter=",")

try:
    os.mkdir('UNetCGANFinal_nonuniform')
except OSError as error:
    print(error)

torch.save(bestD, 'UNetCGANFinal_nonuniform/Trained_ModelMSE_Dnewl2.pt')
torch.save(bestG, 'UNetCGANFinal_nonuniform/Trained_ModelMSE_Gnewl2.pt')