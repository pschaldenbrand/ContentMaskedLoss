import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms

from utils.tensorboard import TensorBoard

import cv2
import numpy as np
import datetime
import os

''' Data from https://sketchy.eye.gatech.edu/ '''

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SketchyClassifier(nn.Module):
    sketchy_img_dir = 'sketchy/photo/tx_000000000000/'
    # sketchy_img_dir = 'sketchy/sketch/tx_000000000000/'
    sketchy_dir = 'sketchy/'
    
    width = 224 # Pretrained requires size be at least this

    class_names = []
    for dir_name in sorted(os.listdir(sketchy_img_dir)):
        class_names.append(dir_name)
    
    normalize = transforms.Compose([transforms.ToPILImage(),
                         transforms.Resize((width,width)),
                         transforms.ToTensor(),
                         transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])])
    
    num_classes = 125

    num_files_total = 0
    for class_name in class_names:
        class_dir = os.path.join(sketchy_img_dir, class_name)
        num_files_total += len(os.listdir(class_dir))
    num_train_imgs = int(.9 * num_files_total) # 100 images per class. 90/10 train/val split
    num_val_imgs = int(.1 * num_files_total)

    def __init__(self, model='googlenet'):
        super(SketchyClassifier, self).__init__()
        self.model = model

        if model == 'vgg16':
            self.googlenet = models.vgg16(pretrained=True)
            self.googlenet.fc = nn.Linear(1000, self.num_classes)
        elif model == 'shape-resnet':
            from texture_vs_shape.models.load_pretrained_models import load_model
            self.googlenet = load_model("resnet50_trained_on_SIN_and_IN_then_finetuned_on_IN").module
            self.googlenet.fc = nn.Linear(2048, self.num_classes)
        elif model == 'robustness':
            from robustness.model_utils import make_and_restore_model
            from robustness.datasets import ImageNet
            ds = ImageNet('/tmp')
            model, _ = make_and_restore_model(arch='resnet50', dataset=ds,
                         resume_path='sketchy/imagenet_linf_4.pt')
            self.googlenet = model.model
            self.googlenet.fc = nn.Linear(2048, self.num_classes)
        else: # googlenet
            self.googlenet = models.googlenet(pretrained=True)
            self.googlenet.fc = nn.Linear(1024, self.num_classes)
        
        for name, param in self.googlenet.named_parameters():
            if name == 'fc.weight' or name=='fc.bias':
                param.requires_grad = True
            else:
                param.requires_grad = False
            print(name, param.requires_grad)

        if os.path.exists('{}/SketchyClassifierRobustness.pkl'.format(self.sketchy_dir)):
            self.load_weights()
        self.googlenet.to(device)

    def forward(self, x):
        out = self.googlenet(x)
        return out

    def classify(self, x):
        self.googlenet.eval()
        outputs = self.googlenet(x)
        if self.model == 'vgg16':
            outputs = self.googlenet.fc(outputs)
        class_indices = torch.argmax(outputs, dim=1)
        output_classes = []
        for i in range(len(class_indices)):
            output_classes.append(self.class_names[class_indices[i]])

        confidences = outputs - torch.min(outputs)
        confidences = confidences / (torch.sum(confidences) + 1e-20)
        confidences = confidences[0, class_indices]

        return output_classes, confidences

    def load_weights(self):
        self.googlenet.load_state_dict(torch.load('{}/SketchyClassifierRobustness.pkl'.format(self.sketchy_dir)))

    def save_model(self):
        self.googlenet.cpu()
        torch.save(self.googlenet.state_dict(),'{}/SketchyClassifierRobustness.pkl'.format(self.sketchy_dir))
        self.googlenet.to(device)

    def one_hot_vec(self, ind):
        one_hot = torch.zeros(125)
        one_hot[ind] = 1.
        return one_hot

    def train(self, epochs, batch_size=64):
        self.load_data()

        date_and_time = datetime.datetime.now()
        run_name = 'SketchyClassifier_' + date_and_time.strftime("%m_%d__%H_%M_%S")
        writer = TensorBoard('train_log/{}'.format(run_name))

        lr = 1e-5

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.googlenet.parameters(), lr=lr)

        self.googlenet.train()

        for epoch in range(epochs):
            running_loss = 0.0

            scrambled_inds = np.arange(self.num_train_imgs)
            np.random.shuffle(scrambled_inds)

            for batch_ind in range(0, self.num_train_imgs, batch_size):
                x = self.x_train[scrambled_inds[batch_ind:batch_ind + batch_size]].float().to(device)
                y = self.y_train[scrambled_inds[batch_ind:batch_ind + batch_size]].float().to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.googlenet(x)
                if self.model == 'vgg16':
                    outputs = self.googlenet.fc(outputs) # just for VGG16

                loss = criterion(outputs, torch.argmax(y, dim=1))
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
            print('Train loss: {:.5f}\tValidation loss: {:.5f}\tVal Acc: {:.2f}' \
                  .format(running_loss / self.num_train_imgs, self.validation_loss(), self.validation_accuracy()*100))
            self.googlenet.train()
            self.save_model()

    def validation_loss(self, batch_size=64):
        running_loss = 0.0
        criterion = nn.CrossEntropyLoss()
        self.googlenet.eval()
        with torch.no_grad():
            for batch_ind in range(0, self.num_val_imgs, batch_size):
                x = self.x_val[batch_ind:batch_ind + batch_size].float().to(device)
                y = self.y_val[batch_ind:batch_ind + batch_size].float().to(device)

                outputs = self.googlenet(x)
                if self.model == 'vgg16':
                    outputs = self.googlenet.fc(outputs) # just for VGG16
                loss = criterion(outputs, torch.argmax(y, dim=1))
                running_loss += loss.item()
        return running_loss / self.num_val_imgs

    def validation_accuracy(self, batch_size=64):
        running_acc = 0.0
        criterion = nn.CrossEntropyLoss()
        self.googlenet.eval()
        with torch.no_grad():
            for batch_ind in range(0, self.num_val_imgs + 1, batch_size):
                x = self.x_val[batch_ind:batch_ind + batch_size].float().to(device)
                y = self.y_val[batch_ind:batch_ind + batch_size].float().to(device)

                outputs = self.googlenet(x)
                if self.model == 'vgg16':
                    outputs = self.googlenet.fc(outputs) # just for VGG16
                match = torch.argmax(outputs, dim=1) == torch.argmax(y, dim=1)
                running_acc += torch.sum(match)
        return running_acc / self.num_val_imgs

    def load_data(self):
        self.x_train = torch.zeros((self.num_train_imgs, 3, self.width, self.width), dtype=torch.float16)
        self.x_val = torch.zeros((self.num_val_imgs, 3, self.width, self.width), dtype=torch.float16)

        self.y_train = torch.zeros((self.num_train_imgs, self.num_classes), dtype=torch.float16)
        self.y_val = torch.zeros((self.num_val_imgs, self.num_classes), dtype=torch.float16)

        i_train, i_val = 0, 0

        for class_name in self.class_names:
            class_dir = os.path.join(self.sketchy_img_dir, class_name)
            
            img_fns = []
            
            # Sort filenames for consistency in train/val split
            for filename in os.listdir(class_dir):
                img_fns.append(filename)

            class_img_ind = 0
            train_ind_cuttoff = int(0.9*len(img_fns))
            for img_fn in sorted(img_fns):
                img = cv2.imread(os.path.join(class_dir, img_fn), cv2.IMREAD_UNCHANGED)
                img = img[:,:,::-1] # BGR to RGB
                img = cv2.resize(img, (self.width, self.width))

                if class_img_ind < train_ind_cuttoff:
                    self.x_train[i_train,:,:,:] = self.normalize(img)
                    self.y_train[i_train, :] = self.one_hot_vec(self.class_names.index(class_name))
                    i_train += 1
                else:
                    if i_val < self.num_val_imgs:
                        self.x_val[i_val,:,:,:] = self.normalize(img)
                        self.y_val[i_val, :] = self.one_hot_vec(self.class_names.index(class_name))
                        i_val += 1
                class_img_ind += 1

                if (i_train + i_val + 1) % 1000 == 0:
                    print((i_train + i_val + 1), ' Images Loaded.')