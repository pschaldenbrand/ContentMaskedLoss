''' Adapted from https://github.com/hzwer/ICCV2019-LearningToPaint '''

import cv2
import torch
import numpy as np

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from utils.tensorboard import TensorBoard
from Renderer.model import FCN
from Renderer.stroke_gen import *
import os

import argparse

CONSTRAINT_BRUSH_WIDTH = 0.01
CONSTRAINT_OPACITY = 1.0
CONSTRAINT_MAX_STROKE_LENGTH = 0.3

parser = argparse.ArgumentParser(description='Train Neural Renderer')
parser.add_argument('--name', default='renderer', type=str, help='Name the output renderer file. Leave off ".pkl"')
parser.add_argument('--constrained', default=True, type=bool, help='If True, use the constraints from paper. If False, use Huang et al. 2019.')
parser.add_argument('--resume', action='store_true', help='Resume training from file name')
parser.add_argument('--batch_size', default=64, type=int, help='Batch Size')
    
args = parser.parse_args()

renderer_fn = args.name + '.pkl'

if not os.path.exists('train_log'): os.mkdir('train_log')
log_dir = os.path.join('train_log', args.name)
if not os.path.exists(log_dir): os.mkdir(log_dir)

writer = TensorBoard(log_dir)

criterion = nn.MSELoss()
net = FCN()
optimizer = optim.Adam(net.parameters(), lr=3e-6)
batch_size = args.batch_size

use_cuda = torch.cuda.is_available()
step = 0

# For the constrained model. Can't make the brush width tiny all at once. Need to decrease slowly.
curr_brush_width = 0.6 # Starting brush width
dec_brush_width = 0.01 # Every interval, decrease by this amount
dec_brush_width_int = 1000 # Every this number of steps, decrease the brush width until target

def save_model():
    if use_cuda:
        net.cpu()
    torch.save(net.state_dict(), renderer_fn)
    if use_cuda:
        net.cuda()


def load_weights():
    pretrained_dict = torch.load(renderer_fn)
    model_dict = net.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    net.load_state_dict(model_dict)


if args.resume:
    load_weights()
while step < 600000:
    net.train()
    train_batch = []
    ground_truth = []
    for i in range(batch_size):
        f = np.random.uniform(0, 1, 10)
        train_batch.append(f)
        if args.constrained:
            ground_truth.append(draw(f,
                    max_brush_width=(curr_brush_width, curr_brush_width),
                    opacity=(CONSTRAINT_OPACITY, CONSTRAINT_OPACITY),
                    max_length=CONSTRAINT_MAX_STROKE_LENGTH))
        else:
            ground_truth.append(draw(f))
    train_batch = torch.tensor(train_batch).float()
    ground_truth = torch.tensor(ground_truth).float()
    if use_cuda:
        net = net.cuda()
        train_batch = train_batch.cuda()
        ground_truth = ground_truth.cuda()
    gen = net(train_batch)
    optimizer.zero_grad()
    loss = criterion(gen, ground_truth)
    loss.backward()
    optimizer.step()
    print(step, loss.item())
    if step < 200000:
        lr = 1e-4
    elif step < 400000:
        lr = 1e-5
    else:
        lr = 1e-6
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    if args.constrained and (step + 1) % dec_brush_width_int == 0:
        curr_brush_width -= dec_brush_width
        curr_brush_width = max(curr_brush_width, CONSTRAINT_BRUSH_WIDTH)

    writer.add_scalar("train/loss", loss.item(), step)
    if step % 500 == 0:
        net.eval()
        gen = net(train_batch)
        loss = criterion(gen, ground_truth)
        writer.add_scalar("val/loss", loss.item(), step)
        for i in range(32):
            G = gen[i].cpu().data.numpy() * 255
            GT = ground_truth[i].cpu().data.numpy() * 255
            # G = np.array([G,G,G])
            # GT = np.array([GT,GT,GT])
            # writer.add_image("train/img{}.png".format(i), np.transpose(G, (1,2,0)), step)
            # writer.add_image("train/img{}_truth.png".format(i), np.transpose(GT, (1,2,0)), step)
            writer.add_image("train/img{}.png".format(i), G, step)
            writer.add_image("train/img{}_truth.png".format(i), GT, step)
    if step % 1000 == 0:
        save_model()
    step += 1
