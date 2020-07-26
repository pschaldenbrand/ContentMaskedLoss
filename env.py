import sys
import json
import torch
import numpy as np
import argparse
import torchvision.transforms as transforms
import cv2
import os
from DRL.ddpg import decode, decode_multiple_renderers
from utils.util import *
from PIL import Image
from torchvision import transforms, utils

from DRL.content_loss import *

import pickle
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

aug = transforms.Compose(
            [transforms.ToPILImage(),
             transforms.RandomHorizontalFlip(),
             ])

width = 128
convas_area = width * width

# img_train = []
# img_test = []

# mask_train = []
# mask_test = []

# train_num = 0
# test_num = 0

class Paint:
    def __init__(self, opt):
        self.batch_size = opt.env_batch
        self.max_step = opt.max_step
        self.action_space = (13)
        self.observation_space = (self.batch_size, width, width, 7)
        self.test = False
        self.canvas_color = opt.canvas_color
        self.loss_fcn = opt.loss_fcn
        self.use_multiple_renderers = opt.use_multiple_renderers

        self.img_train = []
        self.img_test = []

        self.mask_train = []
        self.mask_test = []

        self.train_num = 0
        self.test_num = 0

        self.opt = opt
        
    def load_data_celeba(self):
        # CelebA
        #global train_num, test_num, img_train, img_test, mask_train, mask_test
        if os.path.exists('img_train.pkl') and os.path.exists('img_test.pkl'):
            self.img_train = pickle.load(open("img_train.pkl", "rb"))
            self.img_test = pickle.load(open("img_test.pkl", "rb"))
            self.train_num = len(self.img_train)
            self.test_num = len(self.img_test)
        if os.path.exists('mask_train.pkl') and os.path.exists('mask_test.pkl') \
                and (self.loss_fcn == 'cm' or self.loss_fcn == 'cml1'):
            self.mask_train = pickle.load(open("mask_train.pkl", "rb"))
            self.mask_test = pickle.load(open("mask_test.pkl", "rb"))
            
        if self.train_num == 0:
            # for i in range(100000):
            for i in range(200000):
                img_id = '%06d' % (i + 1)
                try:
                    img = cv2.imread('./data/img_align_celeba/' + img_id + '.jpg', cv2.IMREAD_UNCHANGED)
                    img = cv2.resize(img, (width, width))
                    if i > 2000:                
                        self.train_num += 1
                        self.img_train.append(img)
                        if self.loss_fcn == 'cm' or self.loss_fcn == 'cml1':
                            mask = get_l2_mask(torch.unsqueeze(torch.tensor(np.transpose(img.astype('float32'), (2, 0, 1))), 0) / 255).cpu()[:,0,:,:]
                            mask = mask.numpy() * 255
                            mask = mask.astype(np.uint8)
                            #print(mask.shape, img.shape, len(img_train), len(mask_train))
                            self.mask_train.append(mask)
                    else:
                        self.test_num += 1
                        self.img_test.append(img)
                        if self.loss_fcn == 'cm' or self.loss_fcn == 'cml1':
                            mask = get_l2_mask(torch.unsqueeze(torch.tensor(np.transpose(img.astype('float32'), (2, 0, 1))), 0) / 255).cpu()[:,0,:,:]
                            mask = mask.numpy() * 255
                            mask = mask.astype(np.uint8)
                            #print(mask.shape, img.shape, len(img_test), len(mask_test), type(img[0,0,0]), type(mask[0,0,0]))
                            self.mask_test.append(mask)
                finally:
                    if (i + 1) % 10000 == 0:                    
                        print('loaded {} images'.format(i + 1))
            pickle.dump( self.img_train, open( "img_train.pkl", "wb" ) )
            pickle.dump( self.img_test, open( "img_test.pkl", "wb" ) )
            if self.loss_fcn == 'cm' or self.loss_fcn == 'cml1':
                pickle.dump( self.mask_train, open( "mask_train.pkl", "wb" ) )
                pickle.dump( self.mask_test, open( "mask_test.pkl", "wb" ) )
        print('finish loading data, {} training images, {} testing images'.format(str(self.train_num), str(self.test_num)))

    def load_data_bird(self):
        # Birds
        if os.path.exists('img_train_bird.pkl') and os.path.exists('img_test_bird.pkl'):
            self.img_train = pickle.load(open("img_train_bird.pkl", "rb"))
            self.img_test = pickle.load(open("img_test_bird.pkl", "rb"))
            self.train_num = len(self.img_train)
            self.test_num = len(self.img_test)
        if os.path.exists('mask_train_bird.pkl') and os.path.exists('mask_test_bird.pkl') \
                and (self.loss_fcn == 'cm' or self.loss_fcn == 'cml1'):
            self.mask_train = pickle.load(open("mask_train_bird.pkl", "rb"))
            self.mask_test = pickle.load(open("mask_test_bird.pkl", "rb"))
            
        if self.train_num == 0:
            for subdir, dirs, files in os.walk('data/birds/'):
                for file in files:
                    if not (file.endswith('.jpg') or file.endswith('.png') or file.endswith('.jpeg')):
                        continue
                    try:
                        img = cv2.imread(os.path.join(subdir, file), cv2.IMREAD_UNCHANGED)
                        img = cv2.resize(img, (width, width))
                        if i > 2000:                
                            self.train_num += 1
                            self.img_train.append(img)
                            if self.loss_fcn == 'cm' or self.loss_fcn == 'cml1':
                                mask = get_l2_mask(torch.unsqueeze(torch.tensor(np.transpose(img.astype('float32'), (2, 0, 1))), 0) / 255).cpu()[:,0,:,:]
                                mask = mask.numpy() * 255
                                mask = mask.astype(np.uint8)
                                self.mask_train.append(mask)
                        else:
                            self.test_num += 1
                            self.img_test.append(img)
                            if self.loss_fcn == 'cm' or self.loss_fcn == 'cml1':
                                mask = get_l2_mask(torch.unsqueeze(torch.tensor(np.transpose(img.astype('float32'), (2, 0, 1))), 0) / 255).cpu()[:,0,:,:]
                                mask = mask.numpy() * 255
                                mask = mask.astype(np.uint8)
                                self.mask_test.append(mask)
                    finally:
                        if (i + 1) % 10000 == 0:                    
                            print('loaded {} images'.format(i + 1))
            pickle.dump( self.img_train, open( "img_train_bird.pkl", "wb" ) )
            pickle.dump( self.img_test, open( "img_test_bird.pkl", "wb" ) )
            if self.loss_fcn == 'cm' or self.loss_fcn == 'cml1':
                pickle.dump( self.mask_train, open( "mask_train_bird.pkl", "wb" ) )
                pickle.dump( self.mask_test, open( "mask_test_bird.pkl", "wb" ) )
        print('finish loading data, {} training images, {} testing images'.format(str(self.train_num), str(self.test_num)))

    def load_data_cat(self):
        # Cats
        if os.path.exists('data/img_train_cat.pkl') and os.path.exists('data/img_test_cat.pkl'):
            self.img_train = pickle.load(open("data/img_train_cat.pkl", "rb"))
            self.img_test = pickle.load(open("data/img_test_cat.pkl", "rb"))
            self.train_num = len(self.img_train)
            self.test_num = len(self.img_test)
        if os.path.exists('data/mask_train_cat.pkl') and os.path.exists('data/mask_test_cat.pkl') \
                and (self.loss_fcn == 'cm' or self.loss_fcn == 'cml1'):
            self.mask_train = pickle.load(open("data/mask_train_cat.pkl", "rb"))
            self.mask_test = pickle.load(open("data/mask_test_cat.pkl", "rb"))
        
        if self.train_num == 0:
            i=0
            for subdir, dirs, files in os.walk('data/cats/images'):
                for file in files:
                    if not (file.endswith('.jpg') or file.endswith('.png') or file.endswith('.jpeg')):
                        continue
                    try:
                        img = cv2.imread(os.path.join(subdir, file), cv2.IMREAD_UNCHANGED)
                        img = cv2.resize(img, (width, width))
                        if i > 2000:                
                            self.train_num += 1
                            self.img_train.append(img)
                            if self.loss_fcn == 'cm' or self.loss_fcn == 'cml1':
                                mask = get_l2_mask(torch.unsqueeze(torch.tensor(np.transpose(img.astype('float32'), (2, 0, 1))), 0) / 255).cpu()[:,0,:,:]
                                mask = mask.numpy() * 255
                                mask = mask.astype(np.uint8)
                                self.mask_train.append(mask)
                        else:
                            self.test_num += 1
                            self.img_test.append(img)
                            if self.loss_fcn == 'cm' or self.loss_fcn == 'cml1':
                                mask = get_l2_mask(torch.unsqueeze(torch.tensor(np.transpose(img.astype('float32'), (2, 0, 1))), 0) / 255).cpu()[:,0,:,:]
                                mask = mask.numpy() * 255
                                mask = mask.astype(np.uint8)
                                self.mask_test.append(mask)
                    except:
                        #print('exception')
                        continue
                    finally:
                        if (i + 1) % 10000 == 0:                    
                            print('loaded {} images'.format(i + 1))
                        i += 1
            pickle.dump( self.img_train, open( "data/img_train_cat.pkl", "wb" ) )
            pickle.dump( self.img_test, open( "data/img_test_cat.pkl", "wb" ) )
            if self.loss_fcn == 'cm' or self.loss_fcn == 'cml1':
                pickle.dump( self.mask_train, open( "data/mask_train_cat.pkl", "wb" ) )
                pickle.dump( self.mask_test, open( "data/mask_test_cat.pkl", "wb" ) )
        print('finish loading data, {} training images, {} testing images'.format(str(self.train_num), str(self.test_num)))

    def load_data_all(self):
        # All image files in dir "data"
        #global train_num, test_num, img_train, img_test, mask_train, mask_test, file_ind
        global file_ind
        file_ind = 1
        if os.path.exists('img_train_all0.pkl') and os.path.exists('img_test_all.pkl'):
            self.img_train = pickle.load(open("img_train_all0.pkl", "rb"))
            self.img_test = pickle.load(open("img_test_all.pkl", "rb"))
            self.train_num = len(self.img_train)
            self.test_num = len(self.img_test)
        if os.path.exists('mask_train_all0.pkl') and os.path.exists('mask_test_all.pkl') \
                and (self.loss_fcn == 'cm' or self.loss_fcn == 'cml1'):
            self.mask_train = pickle.load(open("mask_train_all0.pkl", "rb"))
            self.mask_test = pickle.load(open("mask_test_all.pkl", "rb"))
            
        if self.train_num > 0:
            print('finish loading data, {} training images, {} testing images'.format(str(self.train_num), str(self.test_num)))
            return

        i = 0
        for subdir, dirs, files in os.walk('data'):
            print(subdir)
            for file in files:
                if not (file.endswith('.jpg') or file.endswith('.png') or file.endswith('.jpeg')):
                    continue
                try:
                    img = cv2.imread(os.path.join(subdir, file), cv2.IMREAD_UNCHANGED)
                    if img is None: continue
                    img = cv2.resize(img, (width, width))
                    if i > 2000:                
                        self.train_num += 1
                        self.img_train.append(img)
                        if self.loss_fcn == 'cm' or self.loss_fcn == 'cml1':
                            self.mask_train.append(img_to_mask(img))
                    else:
                        self.test_num += 1
                        self.img_test.append(img)
                        if self.loss_fcn == 'cm' or self.loss_fcn == 'cml1':
                            self.mask_test.append(img_to_mask(img))
                    i += 1
                    if i % 100000 == 0:
                        pickle.dump( self.img_train, open( "img_train_all" + str(file_ind) + ".pkl", "wb" ) )
                        file_ind += 1
                        self.img_train = []
                finally:
                    if (i + 1) % 10000 == 0:                    
                        print('loaded {} images'.format(i + 1))
        pickle.dump( self.img_train, open( "img_train_all" + str(file_ind) + ".pkl", "wb" ) )
        pickle.dump( self.img_test, open( "img_test_all.pkl", "wb" ) )
        if self.loss_fcn == 'cm' or self.loss_fcn == 'cml1':
            pickle.dump( self.mask_train, open( "mask_train_all" + str(file_ind) + ".pkl", "wb" ) )
            pickle.dump( self.mask_test, open( "mask_test_all.pkl", "wb" ) )
        self.train_num = len(self.img_train)
        self.test_num = len(self.img_test)
        print('finish loading data, {} training images, {} testing images'.format(str(self.train_num), str(self.test_num)))

    def load_new_file(self):
        #global train_num, img_train, mask_train, file_ind
        global file_ind
        file_ind += 1
        if not os.path.exists('img_train_all' + str(file_ind) + '.pkl'):
            file_ind = 0
        self.img_train = pickle.load(open("img_train_all" + str(file_ind) + ".pkl", "rb"))
        if self.loss_fcn == 'cm' or self.loss_fcn == 'cml1':
            self.mask_train = pickle.load(open("mask_train_all" + str(file_ind) + ".pkl", "rb"))
        self.train_num = len(self.img_train)

    def img_to_mask(img):
        mask = get_l2_mask(torch.unsqueeze(torch.tensor(np.transpose(img.astype('float32'), (2, 0, 1))), 0) / 255).cpu()[:,0,:,:]
        mask = mask.numpy() * 255
        mask = mask.astype(np.uint8)

    def load_data_pascal(self):
        # PASCAL dataset. From http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html#devkit
        #global train_num, test_num

        i = 0
        img_dir = './VOC2012/JPEGImages/'
        fn_ar = []
        for filename in os.listdir(img_dir):
            fn_ar.append(filename)
        for filename in sorted(fn_ar): # Sort so that every time we redo this, we get the same test images
            img_id = '%06d' % (i + 1)
            try:
                img = cv2.imread(os.path.join(img_dir, filename), cv2.IMREAD_UNCHANGED)
                img = cv2.resize(img, (width, width))
                if i > 2000:                
                    self.train_num += 1
                    self.img_train.append(img)
                else:
                    self.test_num += 1
                    self.img_test.append(img)
            finally:
                if (i + 1) % 5000 == 0:                    
                    print('loaded {} images'.format(i + 1))
            i += 1
        print('finish loading data, {} training images, {} testing images'.format(str(self.train_num), str(self.test_num)))

    def load_data_sketchy(self):
        # Sketchy dataset. From https://sketchy.eye.gatech.edu/
        #global train_num, test_num

        from sketchy.classifier import SketchyClassifier

        i = 0
        for class_name in SketchyClassifier.class_names:
            class_dir = os.path.join(SketchyClassifier.sketchy_img_dir, class_name)
            class_img_ind = 0
            for filename in sorted(os.listdir(class_dir)): # Sort so that every time we redo this, we get the same test images
                img_id = '%06d' % (i + 1)
                try:
                    img = cv2.imread(os.path.join(class_dir, filename), cv2.IMREAD_UNCHANGED)
                    img = cv2.resize(img, (width, width))
                    if class_img_ind < 90:                
                        self.train_num += 1
                        self.img_train.append(img)
                    else:
                        self.test_num += 1
                        self.img_test.append(img)
                finally:
                    if (i + 1) % 5000 == 0:                    
                        print('loaded {} images'.format(i + 1))
                i += 1
                class_img_ind += 1
        print('finish loading data, {} training images, {} testing images'.format(str(self.train_num), str(self.test_num)))
        
    def pre_data(self, id, test):
        if test:
            img = self.img_test[id][:,:,:3]
        else:
            img = self.img_train[id][:,:,:3]
        if not test:
            img = aug(img)
        img = np.asarray(img)
        return np.transpose(img, (2, 0, 1))
    
    def get_mask(self, id, test):
        if test:
            img = self.mask_test[id]
        else:
            img = self.mask_train[id]
        # if not test:
        #     img = aug(img)
        img = torch.tensor(img)
        return img.to(device)

    def reset(self, test=False, begin_num=False):
        self.test = test
        self.imgid = [0] * self.batch_size
        self.gt = torch.zeros([self.batch_size, 3, width, width], dtype=torch.uint8).to(device)
        self.mask = None
        if self.loss_fcn == 'cm' or self.loss_fcn == 'cml1':
            self.mask = torch.zeros([self.batch_size, 1, width, width], dtype=torch.uint8).to(device)
        for i in range(self.batch_size):
            while True:
                if test:
                    id = (i + begin_num)  % self.test_num
                else:
                    id = np.random.randint(self.train_num)
                self.imgid[i] = id
                try:
                    self.gt[i] = torch.tensor(self.pre_data(id, test))
                except:
                    continue
                if self.loss_fcn == 'cm' or self.loss_fcn == 'cml1':
                    self.mask[i] = self.get_mask(id, test)
                break
            
        self.tot_reward = ((self.gt.float() / 255) ** 2).mean(1).mean(1).mean(1)
        self.stepnum = 0

        if self.canvas_color == 'white':
            self.canvas = torch.zeros([self.batch_size, 3, width, width], dtype=torch.uint8).to(device) + 255
        elif self.canvas_color == 'none':
            # init with -1
            self.canvas = torch.zeros([self.batch_size, 3, width, width], dtype=torch.uint8).to(device) - 255
        else:
            # Black canvas
            self.canvas = torch.zeros([self.batch_size, 3, width, width], dtype=torch.uint8).to(device)

        self.lastdis = self.ini_dis = self.cal_dis()
        return self.observation()
    
    def observation(self):
        # canvas B * 3 * width * width
        # gt B * 3 * width * width
        # mask B * 1 * width * width Only if using content masking
        # T B * 1 * width * width
        ob = []
        T = torch.ones([self.batch_size, 1, width, width], dtype=torch.uint8) * self.stepnum
        if self.loss_fcn == 'cm' or self.loss_fcn == 'cml1':
            return torch.cat((self.canvas, self.gt, self.mask, T.to(device)), 1), self.mask # canvas, img, mask, T

        return torch.cat((self.canvas, self.gt, T.to(device)), 1), None # canvas, img, T and no need for mask

    def cal_trans(self, s, t):
        return (s.transpose(0, 3) * t).transpose(0, 3)
    
    def step(self, action, episode_num):
        if self.use_multiple_renderers:
            self.canvas = (decode_multiple_renderers(action, self.canvas.float() / 255, episode_num) * 255).byte()
        else:
            self.canvas = (decode(action, self.canvas.float() / 255) * 255).byte()
        self.stepnum += 1
        ob, mask = self.observation()
        done = (self.stepnum == self.max_step)
        reward = self.cal_reward() # np.array([0.] * self.batch_size)
        return ob.detach(), reward, np.array([done] * self.batch_size), None, mask

    def cal_dis(self):
        return (((self.canvas.float() - self.gt.float()) / 255) ** 2).mean(1).mean(1).mean(1)
    
    def cal_reward(self):
        dis = self.cal_dis()
        reward = (self.lastdis - dis) / (self.ini_dis + 1e-8)
        self.lastdis = dis
        return to_numpy(reward)
