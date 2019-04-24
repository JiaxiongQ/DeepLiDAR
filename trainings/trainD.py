from __future__ import print_function
import argparse
import os
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import time
from dataloader import dataLoader as lsn
from dataloader import trainLoader as DA
from submodels import *

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

parser = argparse.ArgumentParser(description='deepCpmpletion')
parser.add_argument('--datapath', default='', help='datapath')
parser.add_argument('--epochs', type=int, default=40, help='number of epochs to train')
parser.add_argument('--batch_size', type=int, default=6, help='number of batch size to train')
parser.add_argument('--gpu_nums', type=int, default=3, help='number of gpu to train')
parser.add_argument('--loadmodel', default= '', help='load model')
parser.add_argument('--savemodel', default='my', help='save model')
parser.add_argument('--no-cuda', action='store_true', default=False, help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
args = parser.parse_args()

datapath = args.datapath
args.cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

all_left_img,all_sparse,all_depth = lsn.dataloader(datapath)

TrainImgLoader = torch.utils.data.DataLoader(
        DA.myImageFloder(all_left_img,all_sparse,all_depth, True),
        batch_size=args.batch_size , shuffle=True, num_workers=8, drop_last=True)

model = s2dN(args.batch_size / args.gpu_nums)

if args.cuda:
    model = nn.DataParallel(model)
    model.cuda()

MatI=np.zeros((args.batch_size,256,512), dtype=np.float32)
for i in range(MatI.shape[1]):
    MatI[:,i,:]= i
MatJ = np.zeros((args.batch_size,256,512), dtype=np.float32)
for j in range(MatJ.shape[2]):
    MatJ[:,:,j] = j

MatI = np.reshape(MatI, [args.batch_size,256,512, 1]).astype(np.float32)
MatJ = np.reshape(MatJ, [args.batch_size,256,512, 1]).astype(np.float32)
MatI = torch.FloatTensor(MatI).cuda()
MatJ = torch.FloatTensor(MatJ).cuda()
MatI = torch.squeeze(MatI)
MatJ = torch.squeeze(MatJ)

para_optim = []

if args.loadmodel is not None:
    state_dict = torch.load(args.loadmodel)["state_dict"]
    trained_list = list(state_dict.keys())
    model_dict = model.state_dict()
    new_list = list(model_dict.keys())
    for i in range(len(trained_list)):
        model_dict[new_list[i]] = state_dict[trained_list[i]]
    for param in model.module.parameters():
        param.requires_grad = False
    for param in model.module.outC_block.parameters():
        param.requires_grad = True
    for param in model.module.outN_block.parameters():
        param.requires_grad = True
    model.load_state_dict(model_dict)
optimizer = optim.Adam([{'params':model.module.outC_block.parameters()},
                        {'params':model.module.outN_block.parameters()}], lr=0.001, betas=(0.9, 0.999))

k = np.array([[0,1,0],[1,-4,1],[0,1,0]])
k1 = np.array([[0, 0, 0], [0, 0, 1], [0, 0, 0]])
k2 = np.array([[0, 0, 0], [0, 0, 0], [0, 1, 0]])

def nomal_loss(pred, targetN,params,depthI,depthJ):
    depthI = depthI.permute(0, 2, 3, 1)
    depthJ = depthJ.permute(0, 2, 3, 1)

    predN_1 = torch.zeros_like(targetN)
    predN_2 = torch.zeros_like(targetN)

    f = params[:, :, :, 0]
    cx = params[:, :, :, 1]
    cy = params[:, :, :, 2]

    z1 = depthJ - pred
    z1 = torch.squeeze(z1)
    depthJ = torch.squeeze(depthJ)
    predN_1[:, :, :, 0] = ((MatJ - cx) * z1 + depthJ) * 1.0 / f
    predN_1[:, :, :, 1] = (MatI - cy) * z1 * 1.0 / f
    predN_1[:, :, :, 2] = z1

    z2 = depthI - pred
    z2 = torch.squeeze(z2)
    depthI = torch.squeeze(depthI)
    predN_2[:, :, :, 0] = (MatJ - cx) * z2  * 1.0 / f
    predN_2[:, :, :, 1] = ((MatI - cy) * z2 + depthI) * 1.0 / f
    predN_2[:, :, :, 2] = z2

    predN = torch.cross(predN_1, predN_2)
    pred_n = F.normalize(predN)
    pred_n = pred_n.contiguous().view(-1, 3)
    target_n = targetN.contiguous().view(-1, 3)

    loss_function = nn.CosineEmbeddingLoss()
    loss = loss_function(pred_n, target_n, Variable(torch.Tensor(pred_n.size(0)).cuda().fill_(1.0)))
    return loss

def mse_loss(input,target):
    return torch.sum((input - target)**2) / input.data.nelement()

def total_loss(pred,predC,predN,target,params,normal):
    conv1 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
    w = torch.from_numpy(k1).float().unsqueeze(0).unsqueeze(0).cuda()
    conv1.weight = nn.Parameter(w)
    depthJ1 = conv1(pred)
    conv2 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
    w2 = torch.from_numpy(k2).float().unsqueeze(0).unsqueeze(0).cuda()
    conv2.weight = nn.Parameter(w2)
    depthI1 = conv2(pred)

    valid_mask = (target > 0.0).detach()
    pred = pred.permute(0, 2, 3, 1)
    predN = predN.permute(0, 2, 3, 1)
    predC = predC.permute(0, 2, 3, 1)

    loss4 = nomal_loss(pred, normal, params, depthI1, depthJ1)

    pred_n = pred[valid_mask]
    predN_n = predN[valid_mask]
    predC_n = predC[valid_mask]
    target_n = target[valid_mask]

    loss2 = mse_loss(predC_n, target_n)
    loss3 = mse_loss(predN_n, target_n)
    loss1_function = nn.MSELoss(size_average=True)
    loss1 =  loss1_function(pred_n, target_n)

    loss = 0.5 * loss1 + 0.3 * loss2 + 0.3 * loss3 + 0.1 * loss4

    return loss,loss1,loss2,loss3,loss4

def train(inputl,gt1,sparse,mask,params):
        model.train()
        inputl = Variable(torch.FloatTensor(inputl))
        gt1 = Variable(torch.FloatTensor(gt1))
        sparse = Variable(torch.FloatTensor(sparse))
        mask = Variable(torch.FloatTensor(mask))
        params = Variable(torch.FloatTensor(params))
        if args.cuda:
            inputl,gt1,sparse,params = inputl.cuda(),gt1.cuda(),sparse.cuda(),params.cuda()
            mask = mask.cuda()
        optimizer.zero_grad()

        outC, outN, normals2 = model(inputl, sparse, mask)
        tempMask = torch.zeros_like(outC)
        predC = outC[:, 0, :, :]
        predN = outN[:, 0, :, :]
        maskC = outC[:, 1, :, :]
        maskN = outN[:, 1, :, :]
        tempMask[:, 0, :, :] = maskC
        tempMask[:, 1, :, :] = maskN
        predMask = F.softmax(tempMask)
        predMaskC = predMask[:, 0, :, :]
        predMaskN = predMask[:, 1, :, :]
        pred = predC * predMaskC + predN * predMaskN

        pred = torch.unsqueeze(pred,1)
        predN = torch.unsqueeze(predN, 1)
        predC = torch.unsqueeze(predC, 1)

        b,ch, h, w = normals2.size()
        normals2 = normals2.permute(0,2, 3, 1).contiguous().view(-1, 3)
        normals2 = F.normalize(normals2)
        normals2 = normals2.view(b,h, w, 3)
        outputN = torch.zeros_like(normals2)
        outputN[:,:,:,0] = -normals2[:,:,:,0]
        outputN[:,:,:,1] = -normals2[:,:,:,2]
        outputN[:,:,:,2] = -normals2[:,:,:,1]

        loss,loss1,loss2,loss3,loss4 = total_loss(pred, predC, predN, gt1, params, outputN)
        loss.backward()
        optimizer.step()

        return loss.data[0],loss1.data[0],loss2.data[0],loss3.data[0],loss4.data[0]

def mae(gt,img):
    dif = gt[np.where(gt>0.0)] - img[np.where(gt>0.0)]
    error = np.mean(np.fabs(dif))
    return error

def adjust_learning_rate(optimizer, epoch):
    if epoch <= 5:
        learning_rate = 0.001
    if epoch>5 and epoch <= 10:
        learning_rate = 0.0005
    if epoch>10 and epoch <=15:
        learning_rate = 0.00025
    if epoch>15 and epoch <= 20:
        learning_rate = 0.000125
    if epoch>20 and epoch <= 25:
        learning_rate = 0.0000625
    if epoch > 25 and epoch <= 30:
        learning_rate = 0.00003125
    if epoch > 30 and epoch <= 35:
        learning_rate = 0.000015625
    if epoch > 35 and epoch <= 40:
        learning_rate = 0.00001

    print(learning_rate)
    for param_group in optimizer.param_groups:
        param_group['lr'] = learning_rate

def main():
    start_full_time = time.time()

    for epoch in range(1, args.epochs+1):
        total_train_loss = 0
        adjust_learning_rate(optimizer,epoch)

         ## training ##
        for batch_idx, (imgL_crop,input_crop1,sparse2,mask2,params) in enumerate(TrainImgLoader):
            start_time = time.time()

            loss,loss1,loss2,loss3,loss4 = train(imgL_crop,input_crop1,sparse2,mask2,params)
            print('Iter %d / %d training loss = %.4f, Ploss = %.4f, Closs = %.4f, Nloss = %.4f, n_loss = %.4f, time = %.2f' % (batch_idx, epoch, loss, loss1, loss2, loss3, loss4, time.time() - start_time))
            total_train_loss += loss

        print('epoch %d total training loss = %.10f' %(epoch, total_train_loss/len(TrainImgLoader)))

        #SAVE
        if epoch % 1 == 0:
            savefilename = args.savemodel + '.tar'
            torch.save({
	            'epoch': epoch,
	            'state_dict': model.state_dict(),
	            'train_loss': total_train_loss/len(TrainImgLoader),
	        }, savefilename)

    print('full finetune time = %.2f HR' %((time.time() - start_full_time)/3600))

if __name__ == '__main__':
   main()

