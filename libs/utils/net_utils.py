# -*- coding: utf-8 -*-
# --------------------------------------------------------
# RefineDet in PyTorch
# Written by Dongdong Wang
# Official and original Caffe implementation is at
# https://github.com/sfzhang15/RefineDet
# --------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.init as init
import pdb

class BasicConv(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class BasicRFB(nn.Module):

    def __init__(self, in_planes, out_planes, stride=1, scale = 0.1, visual = 1):
        super(BasicRFB, self).__init__()
        self.scale = scale
        self.out_channels = out_planes
        inter_planes = in_planes // 8
        self.branch0 = nn.Sequential(
                BasicConv(in_planes, 2*inter_planes, kernel_size=1, stride=stride),
                BasicConv(2*inter_planes, 2*inter_planes, kernel_size=3, stride=1, padding=visual, dilation=visual, relu=False)
                )
        self.branch1 = nn.Sequential(
                BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
                BasicConv(inter_planes, 2*inter_planes, kernel_size=(3,3), stride=stride, padding=(1,1)),
                BasicConv(2*inter_planes, 2*inter_planes, kernel_size=3, stride=1, padding=visual+1, dilation=visual+1, relu=False)
                )
        self.branch2 = nn.Sequential(
                BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
                BasicConv(inter_planes, (inter_planes//2)*3, kernel_size=3, stride=1, padding=1),
                BasicConv((inter_planes//2)*3, 2*inter_planes, kernel_size=3, stride=stride, padding=1),
                BasicConv(2*inter_planes, 2*inter_planes, kernel_size=3, stride=1, padding=2*visual+1, dilation=2*visual+1, relu=False)
                )

        self.ConvLinear = BasicConv(6*inter_planes, out_planes, kernel_size=1, stride=1, relu=False)
        self.shortcut = BasicConv(in_planes, out_planes, kernel_size=1, stride=stride, relu=False)
        self.relu = nn.ReLU(inplace=False)
        '''self.deconv = nn.ConvTranspose2d(in_planes, out_planes,
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1)'''

    def forward(self,x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)

        out = torch.cat((x0,x1,x2),1)
        #print("out shape",out.shape)
        out = self.ConvLinear(out)
        short = self.shortcut(x)
        out = out*self.scale + short
        #out = self.deconv(out)
        out = self.relu(out)
        #print(out.shape)

        return out



class BasicRFB_a(nn.Module):

    def __init__(self, in_planes, out_planes, stride=1, scale = 0.1):
        super(BasicRFB_a, self).__init__()
        self.scale = scale
        self.out_channels = out_planes
        inter_planes = in_planes //4


        self.branch0 = nn.Sequential(
                BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
                BasicConv(inter_planes, inter_planes, kernel_size=3, stride=1, padding=1,relu=False)
                )
        self.branch1 = nn.Sequential(
                BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
                BasicConv(inter_planes, inter_planes, kernel_size=(3,1), stride=1, padding=(1,0)),
                BasicConv(inter_planes, inter_planes, kernel_size=3, stride=1, padding=3, dilation=3, relu=False)
                )
        self.branch2 = nn.Sequential(
                BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
                BasicConv(inter_planes, inter_planes, kernel_size=(1,3), stride=stride, padding=(0,1)),
                BasicConv(inter_planes, inter_planes, kernel_size=3, stride=1, padding=3, dilation=3, relu=False)
                )
        self.branch3 = nn.Sequential(
                BasicConv(in_planes, inter_planes//2, kernel_size=1, stride=1),
                BasicConv(inter_planes//2, (inter_planes//4)*3, kernel_size=(1,3), stride=1, padding=(0,1)),
                BasicConv((inter_planes//4)*3, inter_planes, kernel_size=(3,1), stride=stride, padding=(1,0)),
                BasicConv(inter_planes, inter_planes, kernel_size=3, stride=1, padding=5, dilation=5, relu=False)
                )

        self.ConvLinear = BasicConv(4*inter_planes, out_planes, kernel_size=1, stride=1, relu=False)
        self.shortcut = BasicConv(in_planes, out_planes, kernel_size=1, stride=stride, relu=False)
        self.relu = nn.ReLU(inplace=False)
        '''self.deconv = nn.ConvTranspose2d(in_planes, out_planes,
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1)'''
        

    def forward(self,x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)

        out = torch.cat((x0,x1,x2,x3),1)
        out = self.ConvLinear(out)
        short = self.shortcut(x)
        out = out*self.scale + short
        #out = self.deconv(out)
        out = self.relu(out)


        return out

def weights_init(m):
    '''
    Usage:
        model = Model()
        model.apply(weight_init)
    '''
    # print(m)
    if isinstance(m, nn.Conv1d):
        init.normal(m.weight.data)
        if m.bias is not None:
            init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Conv2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Conv3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.ConvTranspose1d):
        init.normal(m.weight.data)
        if m.bias is not None:
            init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.ConvTranspose2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.ConvTranspose3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm1d):
        init.normal(m.weight.data, mean=1, std=0.02)
        if m.bias is not None:
            init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):
        init.normal(m.weight.data, mean=1, std=0.02)
        if m.bias is not None:
            init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm3d):
        init.normal(m.weight.data, mean=1, std=0.02)
        if m.bias is not None:
            init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.constant_(m.bias.data, 0)
    # else:
    #     print('Warning, an unknowned instance!!')
    #     print(m)

class L2Norm(nn.Module):
    def __init__(self,n_channels, scale):
        super(L2Norm, self).__init__()
        self.n_channels = n_channels
        self.gamma = scale or None
        self.eps = 1e-10
        self.weight = nn.Parameter(torch.Tensor(self.n_channels))
        self.reset_parameters()

    def reset_parameters(self):
        init.constant_(self.weight, self.gamma)

    def forward(self, x):
        # pdb.set_trace()
        norm = x.pow(2).sum(dim=1, keepdim=True).sqrt()+self.eps
        #x /= norm
        x = torch.div(x, norm)
        out = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(x) * x
        return out

class TCB(nn.Module):
    """
    Transfer Connection Block Architecture
    This block
    """
    def __init__(self, lateral_channels, channles,
                 internal_channels=256, is_batchnorm=True):
        """
        :param lateral_channels: number of forward feature channles
        :param channles: number of pyramid feature channles
        :param internal_channels: number of internal channels
        """
        super(TCB, self).__init__()
        self.is_batchnorm = is_batchnorm
        # Use bias if is_batchnorm is False, donot otherwise.
        use_bias = not self.is_batchnorm
        # conv + bn + relu
        self.conv1 = nn.Conv2d(lateral_channels, internal_channels,
                               kernel_size=3, padding=1, bias=use_bias)
        # ((conv2 + bn2) element-wise add  (deconv + deconv_bn)) + relu
        # batch normalization before element-wise addition
        self.conv2 = nn.Conv2d(internal_channels, internal_channels,
                               kernel_size=3, padding=1, bias=use_bias)
        self.deconv = nn.ConvTranspose2d(channles, internal_channels,
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias)
        # conv + bn + relu
        self.conv3 = nn.Conv2d(internal_channels, internal_channels,
                               kernel_size=3, padding=1, bias=use_bias)
        self.relu = nn.ReLU(inplace=True)
        
        if self.is_batchnorm:
            self.bn1 = nn.BatchNorm2d(internal_channels)
            self.bn2 = nn.BatchNorm2d(internal_channels)
            self.deconv_bn = nn.BatchNorm2d(internal_channels)
            self.bn3 = nn.BatchNorm2d(internal_channels)
        # attribution
        self.out_channels = internal_channels
    
    def forward(self, lateral, x):
        if self.is_batchnorm:
            lateral_out = self.relu(self.bn1(self.conv1(lateral)))
            # element-wise addation
            out = self.relu(self.bn2(self.conv2(lateral_out)) +
                            self.deconv_bn(self.deconv(x)))
            out = self.bn3(self.conv3(out))
            #print("is batch_norm")
        else:
            # no batchnorm
            lateral_out = self.relu(self.conv1(lateral))
            # element-wise addation
            out = self.relu(self.conv2(lateral_out) + self.deconv(x))
            out = self.conv3(out)
            
        
        return out

class TCB_a(nn.Module):
    """
    Transfer Connection Block Architecture
    This block
    """
    def __init__(self, lateral_channels, channles,
                 internal_channels=256, is_batchnorm=True):
        """
        :param lateral_channels: number of forward feature channles
        :param channles: number of pyramid feature channles
        :param internal_channels: number of internal channels
        """
        super(TCB_a, self).__init__()
        self.is_batchnorm = is_batchnorm
        # Use bias if is_batchnorm is False, donot otherwise.
        use_bias = not self.is_batchnorm
        # conv + bn + relu
        self.conv1 = nn.Conv2d(lateral_channels, internal_channels,
                               kernel_size=3, padding=1, bias=use_bias)
        # ((conv2 + bn2) element-wise add  (deconv + deconv_bn)) + relu
        # batch normalization before element-wise addition
        self.conv2 = nn.Conv2d(internal_channels, internal_channels,
                               kernel_size=3, padding=1, bias=use_bias)
        self.deconv = nn.ConvTranspose2d(channles, internal_channels,
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias)
        # conv + bn + relu
        self.conv3 = BasicRFB_a(internal_channels, internal_channels,
                               stride=1, scale = 1.0)
        self.relu = nn.ReLU(inplace=True)
        
        if self.is_batchnorm:
            self.bn1 = nn.BatchNorm2d(internal_channels)
            self.bn2 = nn.BatchNorm2d(internal_channels)
            self.deconv_bn = nn.BatchNorm2d(internal_channels)
            self.bn3 = nn.BatchNorm2d(internal_channels)
        # attribution
        self.out_channels = internal_channels
    
    def forward(self, lateral, x):
        if self.is_batchnorm:
            lateral_out = self.relu(self.bn1(self.conv1(lateral)))
            # element-wise addation
            out = self.relu(self.bn2(self.conv2(lateral_out)) +
                            self.deconv_bn(self.deconv(x)))
            out = self.bn3(self.conv3(out))
            #print("is batch_norm")
        else:
            # no batchnorm
            lateral_out = self.relu(self.conv1(lateral))
            # element-wise addation
            out = self.relu(self.conv2(lateral_out) + self.deconv(x))
            out = self.conv3(out)
            
        
        return out


def make_special_tcb_layer(in_channels, internal_channels,
                           is_batchnorm=True):
    # layers = list() Transfer the highest layer, so there is no deconv operation
    if is_batchnorm:
        layers = [nn.Conv2d(in_channels, internal_channels,
                            kernel_size=3, padding=1),
                  nn.BatchNorm2d(internal_channels),
                  nn.ReLU(inplace=True),
                  nn.Conv2d(internal_channels, internal_channels,
                            kernel_size=3, padding=1),
                  nn.BatchNorm2d(internal_channels),
                  nn.ReLU(inplace=True),
                  BasicRFB(internal_channels, internal_channels,
                            stride=1, scale = 1.0, visual=1)]
        #print("is batch_norm")

    else:
        layers = [nn.Conv2d(in_channels, internal_channels,
                            kernel_size=3, padding=1),
                  nn.ReLU(inplace=True),
                  nn.Conv2d(internal_channels, internal_channels,
                            kernel_size=3, padding=1),
                  nn.ReLU(inplace=True),
                  BasicRFB(internal_channels, internal_channels,
                            stride=1, scale = 1.0, visual=1)]
    return layers

