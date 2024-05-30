# -*- coding: utf-8 -*-
"""
Created on Thu May 26 09:30:42 2022

@author: Achintha
"""
import torch.nn as nn
import functools

class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim),
                       nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


class ResnetGenerator(nn.Module):
    def __init__(self, input_nc,output_nc,ngf = 64,norm_layer = nn.BatchNorm2d,use_dropout=False,n_blocks=3,gpu_id='cuda:0',padding_type='zeros'):#'reflect'
        super(ResnetGenerator,self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.gpu_ids = gpu_id
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d 
        #self.inpl = nn.ReflectionPad2d(3)
        self.inpl = nn.ZeroPad2d(3)
        self.conv1 = nn.Conv2d(self.input_nc, self.ngf, kernel_size=7,padding=0,bias=use_bias)
        self.norm1 = norm_layer(ngf)
        self.r1 = nn.ReLU(True)
        ds = 2
        self.conv2 = nn.Conv2d(self.ngf, self.ngf*ds,kernel_size=3,stride=2,padding=1,bias=use_bias)
        self.norm2 = norm_layer(ngf*ds)
        self.r2 = nn.ReLU(True)
        self.conv3 = nn.Conv2d(self.ngf*ds, self.ngf*ds**2,kernel_size=3,stride=2,padding=1,bias=use_bias)
        self.norm3 = norm_layer(ngf*ds**2)
        self.r3 = nn.ReLU(True)
        
        factor = 2**ds
        self.resnet1 = ResnetBlock(self.ngf*factor,padding_type=padding_type,norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)
        self.resnet2 = ResnetBlock(self.ngf*factor,padding_type=padding_type,norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)
        #self.resnet3 = ResnetBlock(self.ngf*factor,padding_type=padding_type,norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)
        #self.resnet4 = ResnetBlock(self.ngf*factor,padding_type=padding_type,norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)
        #self.resnet5 = ResnetBlock(self.ngf*factor,padding_type=padding_type,norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)
        #self.resnet6 = ResnetBlock(self.ngf*factor,padding_type=padding_type,norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)
        
        ### Decoder / Reconsruction
        
        uf1 = 2**ds
        uf2 = 2**(ds-1)
        self.upconv1 = nn.ConvTranspose2d(ngf*uf1, int(ngf*uf1/2), kernel_size=3,stride=2,padding=1,output_padding=1,bias=use_bias)
        self.upnorm1 = norm_layer(int(ngf*uf1/2))
        self.upr1    = nn.ReLU(True)
        self.upconv2 = nn.ConvTranspose2d(ngf*uf2, int(ngf*uf2/2), kernel_size=3,stride=2,padding=1,output_padding=1,bias=use_bias)
        self.upnorm2 = norm_layer(int(ngf*uf2/2))
        self.upr2    = nn.ReLU(True)
        
        self.uppad1 = nn.ReplicationPad2d(3)
        self.convf  = nn.Conv2d(ngf,output_nc,kernel_size=7,padding=0)
        self.tan    = nn.Tanh()
        
    def forward(self,x):
        x = self.r1(self.norm1(self.conv1(self.inpl(x))))
        x = self.r2(self.norm2(self.conv2(x)))
        x = self.r3(self.norm3(self.conv3(x)))
        
        f1 = self.resnet1(x)
        f6 = self.resnet2(f1)
        #f3 = self.resnet3(f2)
        #f6 = f3
        # f4 = self.resnet4(f3)
        # f5 = self.resnet5(f4)
        # f6 = self.resnet6(f5)
        #print("shape latent: ",f6.shape)
        
        y  = self.upr1(self.upnorm1(self.upconv1(f6)))
        y = self.upr2(self.upnorm2(self.upconv2(y)))
        
        y = self.tan(self.convf(self.uppad1(y)))
        return y
    
class Discriminator(nn.Module):
    def __init__(self, ngpu,nc = 3, ndf = 64):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)
        