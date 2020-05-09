import os
import cv2
import torch
import numpy as np
import argparse
import torch.nn as nn
import torch.nn.functional as F

from DRL.actor import *
from Renderer.stroke_gen import *
from Renderer.model import *

import matplotlib.pyplot as plt

from colormath.color_objects import sRGBColor, LabColor
from colormath.color_conversions import convert_color
from colormath.color_diff import delta_e_cie2000

from sklearn.cluster import KMeans

import copy
import pandas as pd
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
output_dir = 'output'
actions_dir = 'arduino_actions'

if not os.path.exists(output_dir): os.mkdir(output_dir)
if not os.path.exists(actions_dir): os.mkdir(actions_dir)

def discrete_color(color_stroke, just_inds=False): #(n*5, 3)
    allowed_colors_tensors = [torch.Tensor([allowed_colors[i]] * color_stroke.shape[0]).to(device) for i in range(len(allowed_colors))]
    
    l2_distances = torch.zeros((color_stroke.shape[0], len(allowed_colors_tensors)))
    for i in range(len(allowed_colors_tensors)):
        l2_distances[:, i] = torch.sum((color_stroke - allowed_colors_tensors[i])**2, dim=1)
        for j in range(l2_distances.shape[0]):
            color1_rgb = sRGBColor(color_stroke[j,2], color_stroke[j,1], color_stroke[j,0])
            color2_rgb = sRGBColor(allowed_colors[i][2], allowed_colors[i][1], allowed_colors[i][0])
            color1_lab = convert_color(color1_rgb, LabColor)
            color2_lab = convert_color(color2_rgb, LabColor)
            l2_distances[j, i] = delta_e_cie2000(color1_lab, color2_lab)

    color_inds = torch.argmin(l2_distances, dim=1, keepdims=True).repeat((1,3)).to(device)
    if just_inds:
        return color_inds
    
    new_color_stroke = torch.zeros(color_stroke.shape).to(device)
    for i in range(len(allowed_colors_tensors)):
        new_color_stroke = torch.where(color_inds == i, allowed_colors_tensors[i], new_color_stroke)

    return new_color_stroke

def color_cluster(img_fn, n_colors=6):
    global allowed_colors
    allowed_colors = []

    img = cv2.imread(img_fn, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (width, width)) # we want bgr

    colors = img.reshape((width*width), 3) / 255

    kmeans = KMeans(n_clusters=n_colors)
    kmeans.fit(colors)

    for i in range(n_colors):
        c = kmeans.cluster_centers_[i] # c is in BGR format
        allowed_colors.append(c) #BGR format appended
    return allowed_colors # They're global anyways

def decode(x, canvas, n_strokes=5, discrete_colors=True, width=128): # b * (10 + 3)
    x = x.view(-1, 10 + 3)
    stroke = 1 - Decoder(x[:, :10])
    stroke = stroke.view(-1, width, width, 1)
    
    color_stroke = stroke * x[:, -3:].view(-1, 1, 1, 3)
    if discrete_colors:
        color_stroke = stroke * discrete_color(x[:, -3:]).view(-1, 1, 1, 3)
        
    stroke = stroke.permute(0, 3, 1, 2)
    color_stroke = color_stroke.permute(0, 3, 1, 2)
    stroke = stroke.view(-1, n_strokes, 1, width, width)
    color_stroke = color_stroke.view(-1, n_strokes, 3, width, width)
    res = []
    
    for i in range(n_strokes):
        canvas = canvas * (1 - stroke[:, i]) + color_stroke[:, i]
        res.append(canvas)
    return canvas, res

def small2large(x):
    # (d * d, width, width) -> (d * width, d * width)    
    x = x.reshape(divide, divide, width, width, -1)
    x = np.transpose(x, (0, 2, 1, 3, 4))
    x = x.reshape(divide * width, divide * width, -1)
    return x

def large2small(x):
    # (d * width, d * width) -> (d * d, width, width)
    x = x.reshape(divide, width, divide, width, 3)
    x = np.transpose(x, (0, 2, 1, 3, 4))
    x = x.reshape(canvas_cnt, width, width, 3)
    return x

def smooth(img):
    def smooth_pix(img, tx, ty):
        if tx == divide * width - 1 or ty == divide * width - 1 or tx == 0 or ty == 0: 
            return img
        img[tx, ty] = (img[tx, ty] + img[tx + 1, ty] + img[tx, ty + 1] + img[tx - 1, ty] + img[tx, ty - 1] + img[tx + 1, ty - 1] + img[tx - 1, ty + 1] + img[tx - 1, ty - 1] + img[tx + 1, ty + 1]) / 9
        return img

    for p in range(divide):
        for q in range(divide):
            x = p * width
            y = q * width
            for k in range(width):
                img = smooth_pix(img, x + k, y + width - 1)
                if q != divide - 1:
                    img = smooth_pix(img, x + k, y + width)
            for k in range(width):
                img = smooth_pix(img, x + width - 1, y + k)
                if p != divide - 1:
                    img = smooth_pix(img, x + width, y + k)
    return img


def res_to_img(res, imgid, divide=False):
    output = res.detach().cpu().numpy() # d * d, 3, width, width    
    output = np.transpose(output, (0, 2, 3, 1))
    if divide:
        output = small2large(output)
        output = smooth(output)
    else:
        output = output[0]
    output = (output * 255).astype('uint8')
    output = cv2.resize(output, origin_shape)
    return output
def save_img(res, imgid, divide=False):
    output = res_to_img(res, imgid, divide)
    cv2.imwrite(os.path.join(output_dir, 'generated' + str(imgid) + '.png'), output)

def plot_canvas(res, imgid, divide=False):
    output = res_to_img(res, imgid, divide)
    ax.imshow(output[...,::-1])
    ax.set_xticks([])
    ax.set_yticks([])
    plt.show()

def paint(actor_fn, renderer_fn, max_step=40, div=5, img_width=128,
          img='../image/vangogh.png', discrete_colors=True, n_colors=6,
          white_canvas=True):
    global Decoder, width, divide, canvas_cnt, origin_shape
    width = img_width
    divide = div
    
    Decoder = FCN()
    Decoder.load_state_dict(torch.load(renderer_fn))

    actor = ResNet(9, 18, 65) # action_bundle = 5, 65 = 5 * 13
    actor.load_state_dict(torch.load(actor_fn))
    actor = actor.to(device).eval()
    Decoder = Decoder.to(device).eval()
    
    # Get the allowed colors if it's supposed to be discrete
    if discrete_colors:
        color_cluster(img, n_colors)

    imgid = 0
    canvas_cnt = divide * divide
    T = torch.ones([1, 1, width, width], dtype=torch.float32).to(device)
    img = cv2.imread(img, cv2.IMREAD_COLOR)
    origin_shape = (img.shape[1], img.shape[0])

    coord = torch.zeros([1, 2, width, width])
    for i in range(width):
        for j in range(width):
            coord[0, 0, i, j] = i / (width - 1.)
            coord[0, 1, i, j] = j / (width - 1.)
    coord = coord.to(device) # Coordconv

    canvas = torch.zeros([1, 3, width, width]).to(device)
    if white_canvas:
        canvas = torch.ones([1, 3, width, width]).to(device)
    canvas_discrete = canvas.detach().clone()

    patch_img = cv2.resize(img, (width * divide, width * divide))
    patch_img = large2small(patch_img)
    patch_img = np.transpose(patch_img, (0, 3, 1, 2))
    patch_img = torch.tensor(patch_img).to(device).float() / 255.

    img = cv2.resize(img, (width, width))
    img = img.reshape(1, width, width, 3)
    img = np.transpose(img, (0, 3, 1, 2))
    img = torch.tensor(img).to(device).float() / 255.
    
    actions_whole = None
    actions_divided = None
    all_canvases = []
        
    with torch.no_grad():
        if divide != 1:
            max_step = max_step // 2
        for i in range(max_step):
            stepnum = T * i / max_step
            actions = actor(torch.cat([canvas, img, stepnum, coord], 1))
            # Use the non discrete canvas for acting, but save the discrete canvas if painting with finite colors
            canvas_discrete, res_discrete = decode(actions, canvas_discrete, discrete_colors=discrete_colors)
            canvas, res = decode(actions, canvas, discrete_colors=False)

            if actions_whole is None:
                actions_whole = actions
            else:
                actions_whole = torch.cat([actions_whole, actions], 1)
            for j in range(5):
                # save_img(res[j], imgid)
                # plot_canvas(res[j], imgid)
                all_canvases.append(res_to_img(res_discrete[j], imgid)[...,::-1])
                imgid += 1
        if divide != 1:
            canvas = canvas[0].detach().cpu().numpy()
            canvas = np.transpose(canvas, (1, 2, 0))    
            canvas = cv2.resize(canvas, (width * divide, width * divide))
            canvas = large2small(canvas)
            canvas = np.transpose(canvas, (0, 3, 1, 2))
            canvas = torch.tensor(canvas).to(device).float()
            coord = coord.expand(canvas_cnt, 2, width, width)

            canvas_discrete = canvas_discrete[0].detach().cpu().numpy()
            canvas_discrete = np.transpose(canvas_discrete, (1, 2, 0))    
            canvas_discrete = cv2.resize(canvas_discrete, (width * divide, width * divide))
            canvas_discrete = large2small(canvas_discrete)
            canvas_discrete = np.transpose(canvas_discrete, (0, 3, 1, 2))
            canvas_discrete = torch.tensor(canvas_discrete).to(device).float()
            
            T = T.expand(canvas_cnt, 1, width, width)
            for i in range(max_step):
                stepnum = T * i / max_step
                actions = actor(torch.cat([canvas, patch_img, stepnum, coord], 1))
                canvas_discrete, res_discrete = decode(actions, canvas_discrete, discrete_colors=discrete_colors)
                canvas, res = decode(actions, canvas, discrete_colors=False)
                if actions_divided is None:
                    actions_divided = actions
                else:
                    actions_divided = torch.cat([actions_divided, actions], 1)

                for j in range(5):
                    # save_img(res[j], imgid, True)
                    # plot_canvas(res[j], imgid, True)
                    all_canvases.append(res_to_img(res_discrete[j], imgid, True)[...,::-1])
                    imgid += 1
        
        final_result = res_to_img(res_discrete[-1], imgid, True)[...,::-1]
    return actions_whole, actions_divided, all_canvases, final_result


def save_actions(actions_whole, actions_divided, group_colors=True, group_amount=15):
    # x0, y0, x1, y1, x2, y2, z0, z2, w0, w2 , R, G, B
    if actions_whole is None: actions_whole = np.array([[]])
    if actions_divided is None: actions_divided = np.array([[]])

    act = np.empty((int(actions_whole.shape[1] / 13 + actions_divided.shape[0] * (actions_divided.shape[1]/13)), 13))
    c = 0

    # Add the actions that were from looking at the whole canvas
    for i in range(0, actions_whole.shape[1], 13):
        act[c,:] = actions_whole[0,i:i+13].cpu().numpy().copy()
        act[c,10:13] = discrete_color(actions_whole[0,i+10:i+13].unsqueeze(0), just_inds=True).cpu().numpy()
        c += 1

    # Add the strokes for the divisions of the canvas.  X,Y's need to be converted
    for i in range(actions_divided.shape[0]):
        for j in range(0, actions_divided.shape[1], 13):
            a = actions_divided[i, j:j+13].cpu().numpy().copy()
            a[[0,2,4]] = a[[0,2,4]]/divide + math.floor(i/divide) / divide
            a[[1,3,5]] = a[[1,3,5]]/divide + i%divide / divide
            act[c, :] = a
            act[c,10:13] = discrete_color(torch.Tensor(a[10:13]).unsqueeze(0).to(device), just_inds=True).cpu().numpy()
            c += 1

    df = pd.DataFrame(act)
    df.head()

    if group_colors:
        r = True
        for i in range(0,len(act),group_amount):
            act[i:i+group_amount] = sorted(copy.deepcopy(act[i:i+group_amount]),key=lambda l:l[12], reverse=r)
            r = not r

    df.to_csv(os.path.join(actions_dir, 'actions_all.csv'), sep=",", header=False, index=False, float_format='%.5f')
    df[:int(actions_whole.shape[1]/13)].to_csv(os.path.join(actions_dir, 'actions_big.csv'), sep=",", header=False, index=False, float_format='%.5f')
    df[int(actions_whole.shape[1]/13):] .to_csv(os.path.join(actions_dir, 'actions_small.csv'), sep=",", header=False, index=False, float_format='%.5f')

    # Save the colors used as an image so you know how to mix the paints
    n_colors = len(allowed_colors)
    fig, ax = plt.subplots(1, n_colors, figsize=(2*n_colors, 2))
    i = 0 
    w = 128
    for c in allowed_colors:
        # print('[', int(255*c[2]), ', ', int(255*c[1]), ', ', int(255*c[0]), '],', end='', sep='')
        num_uses = np.sum(act[:,12] == i)
        ax[i].imshow(np.concatenate((np.ones((w,w,1))*c[2], np.ones((w,w,1))*c[1], np.ones((w,w,1))*c[0]), axis=-1))
        ax[i].set_xticks([])
        ax[i].set_yticks([])
        ax[i].set_title(i)
        ax[i].set_xlabel(str(num_uses) + ' uses')
        i += 1

    plt.savefig(os.path.join(actions_dir, 'colors.png'))




from facenet_pytorch import MTCNN, InceptionResnetV1

mtcnn = MTCNN(image_size=128, select_largest=False, keep_all=False, min_face_size=30)

def prob_of_face(img):
    return mtcnn(img, return_prob=True)

def paint_until_face_detected(img, actor_fn, renderer_fn, max_strokes=750, img_width=128,
                              white_canvas=True):
    global Decoder, width, divide, canvas_cnt, origin_shape
    width = img_width
    divide = 1
    
    max_step = int(max_strokes / 5)

    Decoder = FCN()
    Decoder.load_state_dict(torch.load(renderer_fn))

    actor = ResNet(9, 18, 65) # action_bundle = 5, 65 = 5 * 13
    actor.load_state_dict(torch.load(actor_fn))
    actor = actor.to(device).eval()
    Decoder = Decoder.to(device).eval()

    imgid = 0
    canvas_cnt = divide * divide
    T = torch.ones([1, 1, width, width], dtype=torch.float32).to(device)
    img = cv2.imread(img, cv2.IMREAD_COLOR)
    origin_shape = (img.shape[1], img.shape[0])

    coord = torch.zeros([1, 2, width, width])
    for i in range(width):
        for j in range(width):
            coord[0, 0, i, j] = i / (width - 1.)
            coord[0, 1, i, j] = j / (width - 1.)
    coord = coord.to(device) # Coordconv

    canvas = torch.zeros([1, 3, width, width]).to(device)
    if white_canvas:
        canvas = torch.ones([1, 3, width, width]).to(device)

    patch_img = cv2.resize(img, (width * divide, width * divide))
    patch_img = large2small(patch_img)
    patch_img = np.transpose(patch_img, (0, 3, 1, 2))
    patch_img = torch.tensor(patch_img).to(device).float() / 255.

    img = cv2.resize(img, (width, width))
    img = img.reshape(1, width, width, 3)
    img = np.transpose(img, (0, 3, 1, 2))
    img = torch.tensor(img).to(device).float() / 255.
        
    with torch.no_grad():
        for i in range(max_step):
            stepnum = T * i / max_step
            actions = actor(torch.cat([canvas, img, stepnum, coord], 1))
            canvas, res = decode(actions, canvas, discrete_colors=False)
            for j in range(5):
                imgid += 1
                c = res_to_img(res[j], imgid)[...,::-1]
                x_aligned, prob = prob_of_face(cv2.resize(c, (width, width)))
                if prob is not None:
                    if prob > 0.5:
                        return x_aligned, prob, imgid, c
    return None, None, None, c