from __future__ import print_function
import torch
import torch.nn as nn
import torch.utils.data

def convbn(in_planes, out_planes, kernel_size, stride):

    return nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=1, bias=False),
                         nn.BatchNorm2d(out_planes))

def conv(in_planes, out_planes, kernel_size=3,stride=1):
    return nn.Sequential(
		nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=1, bias=False),
		nn.BatchNorm2d(out_planes),
		nn.ReLU(inplace=True)
	)

def deconv(in_planes, out_planes):
    return nn.Sequential(
		nn.ConvTranspose2d(in_planes, out_planes, kernel_size=4, stride=2, padding=1, bias=True),
		nn.BatchNorm2d(out_planes),
		nn.ReLU(inplace=True)
	)

def predict_normal(in_planes):
    return nn.Conv2d(in_planes, 1, kernel_size=3, stride=1, padding=1, bias=True)

def predict_normal2(in_planes):
    return nn.Conv2d(in_planes, 3, kernel_size=3, stride=1, padding=1, bias=True)

def predict_normalE2(in_planes):
    return nn.Conv2d(in_planes, 2, kernel_size=1, stride=1, padding=0, bias=True)

def adaptative_cat3(out_conv, out_deconv, out_depth_up):
    out_deconv = out_deconv[:, :, :out_conv.size(2), :out_conv.size(3)]
    out_depth_up = out_depth_up[:, :, :out_conv.size(2), :out_conv.size(3)]
    return torch.cat((out_conv, out_deconv, out_depth_up), 1)
def adaptative_cat2(out_conv,out_sparse):
    out_sparse = out_sparse[:, :, :out_conv.size(2), :out_conv.size(3)]
    return torch.cat((out_conv, out_sparse), 1)
def adaptative_cat4(out_conv, out_deconv, out_depth_up,out_sparse):
    out_deconv = out_deconv[:, :, :out_conv.size(2), :out_conv.size(3)]
    out_depth_up = out_depth_up[:, :, :out_conv.size(2), :out_conv.size(3)]
    out_sparse = out_sparse[:, :, :out_conv.size(2), :out_conv.size(3)]
    return torch.cat((out_conv, out_deconv, out_depth_up, out_sparse), 1)
def adaptative_cat(out_conv, out_deconv, out_depth_up):
    out_deconv = out_deconv[:, :, :out_conv.size(2), :out_conv.size(3)]
    out_depth_up = out_depth_up[:, :, :out_conv.size(2), :out_conv.size(3)]
    return torch.cat((out_conv, out_deconv, out_depth_up), 1)
