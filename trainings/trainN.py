from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import time
from dataloader import nomalLoader as lsn
from dataloader import trainLoaderN as DA
from submodels import *


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

parser = argparse.ArgumentParser(description='deepCpmpletion')
parser.add_argument('--model', default='normal',
                    help='select model')
parser.add_argument('--datatype', default='png',
                    help='datapath')
parser.add_argument('--datapath', default='',
                    help='datapath')
parser.add_argument('--epochs', type=int, default=40,
                    help='number of epochs to train')
parser.add_argument('--loadmodel', default='',
                    help='load model')
parser.add_argument('--savemodel', default='my',
                    help='save model')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
args = parser.parse_args()


datapath = args.datapath
args.cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

all_left_img = []
all_normal = []
all_gts = []
if args.model == 'normal':
    all_left_img, all_normal, all_gts = lsn.dataloader(datapath)


print(len(all_left_img))

TrainImgLoader = torch.utils.data.DataLoader(
        DA.myImageFloder(all_left_img,all_normal,all_gts ,True, args.model),
        batch_size = 12, shuffle=True, num_workers=8, drop_last=True)

model = s2dN()

if args.cuda:
    model = nn.DataParallel(model)
    model.cuda()

para_optim = []
if args.loadmodel is not None:
    state_dict = torch.load(args.loadmodel)["state_dict"]
    model.load_state_dict(state_dict)
print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))
optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))


def nomal_loss(pred, targetN,mask1):
    valid_mask = (mask1 > 0.0).detach()
    pred_n = pred.permute(0,2,3,1)
    pred_n = pred_n[valid_mask]
    target_n = targetN[valid_mask]

    pred_n = pred_n.contiguous().view(-1,3)
    pred_n = F.normalize(pred_n)
    target_n = target_n.contiguous().view(-1, 3)

    loss_function = nn.CosineEmbeddingLoss()
    loss = loss_function(pred_n, target_n, Variable(torch.Tensor(pred_n.size(0)).cuda().fill_(1.0)))
    return loss

def train(inputl,sparse,mask,mask1,gt1):
        model.train()
        inputl = Variable(torch.FloatTensor(inputl))
        gt1 = Variable(torch.FloatTensor(gt1))
        sparse = Variable(torch.FloatTensor(sparse))
        mask = Variable(torch.FloatTensor(mask))
        mask1 = Variable(torch.FloatTensor(mask1))
        if args.cuda:
            inputl,gt1 = inputl.cuda(),gt1.cuda()
            sparse=sparse.cuda()
            mask1 = mask1.cuda()
            mask = mask.cuda()

        optimizer.zero_grad()

        pred = model(inputl,sparse,mask)

        loss = nomal_loss(pred, gt1,mask1)

        loss.backward()
        optimizer.step()

        return loss.data[0]

def adjust_learning_rate(optimizer, epoch):
    if epoch <= 10:
        learning_rate = 0.001
    if epoch>10 and epoch <= 20:
        learning_rate = 0.0005
    if epoch>20 and epoch <=30:
        learning_rate = 0.00025
    if epoch>30 and epoch <= 40:
        learning_rate = 0.000125

    print(learning_rate)
    for param_group in optimizer.param_groups:
        param_group['lr'] = learning_rate

def main():
    start_full_time = time.time()

    for epoch in range(1, args.epochs+1):
        total_train_loss = 0
        adjust_learning_rate(optimizer,epoch)

         ## training ##
        for batch_idx, (imgL_crop,sparse_n,mask,mask1,data_in1) in enumerate(TrainImgLoader):
            start_time = time.time()
            loss= train(imgL_crop,sparse_n,mask,mask1,data_in1)
            print('%s Iter %d / %d training loss = %.4f, time = %.2f' % (args.model, batch_idx, epoch, loss, time.time() - start_time))
            total_train_loss += loss
        print('epoch %d total training loss = %.10f' %(epoch, total_train_loss/len(TrainImgLoader)))

        #SAVE
        if epoch % 1 == 0:
            savefilename = args.savemodel+'.tar'
            torch.save({
	            'epoch': epoch,
	            'state_dict': model.state_dict(),
	            'train_loss': total_train_loss/len(TrainImgLoader),
	        }, savefilename)

    print('full finetune time = %.2f HR' %((time.time() - start_full_time)/3600))

if __name__ == '__main__':
   main()
