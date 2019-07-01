import pandas as pd
import cv2
import numpy as np

from torchvision.transforms.functional import to_tensor

import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim

from tqdm import tqdm


#%%
'''
Loading Files and setting varibales
'''
image_rows = 1024
image_cols = 1024
image_channels = 3

initial_learning_rate = 1e-3
epochs = 20
steps = 5
batch_size = 2
safe = 1e-7

read_path = 'D:\\text\\'
write_path = 'D:\\weights\\'

train = pd.read_csv(read_path + 'train_with_sign.txt', sep = ",", 
                    names = ["filepath", "xmin", "ymin", "xmax", "ymax"])
val = pd.read_csv(read_path + 'val_with_sign.txt', sep = ",", 
                  names = ["filepath", "xmin", "ymin", "xmax", "ymax"])

train = train.sample(frac = 1, random_state = 200).reset_index(drop = True)
val = val.sample(frac = 1, random_state = 200).reset_index(drop = True)

len_train = len(train)
len_val = len(val)
len_train_step = len_train // steps
len_val_step = len_val // steps


#%%
'''
Input-target generation function
'''
def get_input_target(dataframe, ind):
    try:
        file_name = dataframe['filepath'][ind]
    except:
        return
    
    img = cv2.imread(file_name)
    while (np.all(img) == None):
        ind += 1
        file_name = dataframe['filepath'][ind]
        img = cv2.imread(file_name)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    bb_boxes = dataframe[dataframe['filepath'] == file_name].reset_index()
    img_mask = np.zeros_like(img[:, :, 0])
    
    for i in range(len(bb_boxes)):
        bb_box_i = [bb_boxes['xmin'][i], bb_boxes['ymin'][i],
                    bb_boxes['xmax'][i], bb_boxes['ymax'][i]]
        img_mask[int(bb_box_i[1]) : int(bb_box_i[3]) + 1, int(bb_box_i[0]) : int(bb_box_i[2]) + 1] = 1
        img_mask = np.reshape(img_mask, (np.shape(img_mask)[0], np.shape(img_mask)[1], 1))
        
        rndm = np.random.randint(0, 2, 2)
        image = img[rndm[0] * 212 : rndm[0] * 212 + 1024, rndm[1] * 604 : rndm[1] * 604 + 1024, :]
        mask = img_mask[rndm[0] * 212 : rndm[0] * 212 + 1024, rndm[1] * 604 : rndm[1] * 604 + 1024, :]
        
        input_image = to_tensor(image).to(device = device, dtype =  torch.float32)
        target_image = to_tensor(mask).to(device = device, dtype =  torch.float32)

    return input_image, target_image


#%%
'''
Netwrok Architecture
'''
class conv_block(nn.Module):
    
    def __init__(self, input_channels, output_channels):
        super(conv_block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels = input_channels, 
                               out_channels = output_channels, 
                               kernel_size = 3, 
                               padding = 1)
        self.batch1 = nn.BatchNorm2d(num_features = output_channels)
        self.conv2 = nn.Conv2d(in_channels = output_channels, 
                               out_channels = output_channels, 
                               kernel_size = 3, 
                               padding = 1)
        self.batch2 = nn.BatchNorm2d(num_features = output_channels)
        
    def forward(self, x):
        x = f.relu(self.batch1(self.conv1(x)))
        x = f.relu(self.batch2(self.conv2(x)))
        
        return x


class segunet(nn.Module):
    
    def __init__(self):
        super(segunet, self).__init__()
        self.conv_block1 = conv_block(input_channels = 3, output_channels = 16)
        self.conv_block2 = conv_block(input_channels = 16, output_channels = 32)
        self.conv_block3 = conv_block(input_channels = 32, output_channels = 64)
        self.conv_block4 = conv_block(input_channels = 64, output_channels = 128)
        self.conv_block5 = conv_block(input_channels = 128, output_channels = 128)
        self.conv_block6 = conv_block(input_channels = 128, output_channels = 128)
        self.conv_block7 = conv_block(input_channels = 256, output_channels = 128)
        self.conv_block8 = conv_block(input_channels = 256, output_channels = 64)
        self.conv_block9 = conv_block(input_channels = 128, output_channels = 32)
        self.conv_block10 = conv_block(input_channels = 64, output_channels = 16)
        self.conv21 = nn.Conv2d(in_channels = 32, out_channels = 16, 
                                kernel_size = 3, padding = 1)
        self.batch21 = nn.BatchNorm2d(num_features = 16)
        self.conv22 = nn.Conv2d(in_channels = 16, out_channels = 1, kernel_size = 1)
        
    def forward(self, x):
        x1 = self.conv_block1(x)
        p1, ind1 = f.max_pool2d(x1, kernel_size = 2, return_indices = True)
        x2 = self.conv_block2(p1)
        p2, ind2 = f.max_pool2d(x2, kernel_size = 2, return_indices = True)
        x3 = self.conv_block3(p2)
        p3, ind3 = f.max_pool2d(x3, kernel_size = 2, return_indices = True)
        x4 = self.conv_block4(p3)
        p4, ind4 = f.max_pool2d(x4, kernel_size = 2, return_indices = True)
        x5 = self.conv_block5(p4)
        p5, ind5 = f.max_pool2d(x5, kernel_size = 2, return_indices = True)
        x6 = self.conv_block6(p5)
        u5 = f.max_unpool2d(x6, kernel_size = 2, indices = ind5)
        x7 = torch.cat((x5, u5), dim = 1)
        x8 = self.conv_block7(x7)
        u4 = f.max_unpool2d(x8, kernel_size = 2, indices = ind4)
        x9 = torch.cat((x4, u4), dim = 1)
        x10 = self.conv_block8(x9)
        u3 = f.max_unpool2d(x10, kernel_size = 2, indices = ind3)
        x11 = torch.cat((x3, u3), dim = 1)
        x12 = self.conv_block9(x11)
        u2 = f.max_unpool2d(x12, kernel_size = 2, indices = ind2)
        x13 = torch.cat((x2, u2), dim = 1)
        x14 = self.conv_block10(x13)
        u1 = f.max_unpool2d(x14, kernel_size = 2, indices = ind1)
        x15 = torch.cat((x1, u1), dim = 1)
        x16 = f.relu(self.batch21(self.conv21(x15)))
        x17 = torch.sigmoid(self.conv22(x16))
        
        return x17


net = segunet()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net.to(device)
print(net)


#%%
'''
Loss function
'''
def iou_calc(y_true, y_pred):
    ones = torch.ones_like(y_true)
    y_true_inv = ones - y_true
    y_pred_inv = ones - y_pred
    tp = torch.sum(y_true * y_pred, dim = 1)
    fp = torch.sum(y_true_inv * y_pred, dim = 1)
    fn = torch.sum(y_true * y_pred_inv, dim = 1)
    
    iou = torch.mean(tp / (tp + fp + fn + safe))

    return iou


def tversky_loss_l1_constraint(y_true, y_pred, alpha = 0.0, gamma = 0.0):
    beta = 1 - alpha
    ones = torch.ones_like(y_true)
    
    y_true_inv = ones - y_true
    y_pred_inv = ones - y_pred
    tp = torch.sum(y_true * y_pred, dim = 1)
    fp = torch.sum(y_true_inv * y_pred, dim = 1)
    fn = torch.sum(y_true * y_pred_inv, dim = 1)
    
    tversky = torch.mean(tp / (tp + alpha * fp + beta * fn + safe))
    l1_loss = f.l1_loss(y_pred, y_true, reduction = 'elementwise_mean')
    
    return 1 - torch.sum(tversky) / batch_size  + gamma * l1_loss


#%%
'''
Network optimizer
'''
optimizer = optim.Adam(net.parameters(), lr = initial_learning_rate)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode = 'max', factor = 0.5, patience = 1, verbose = True)

with open(write_path + 'segunet_torch_history.csv', 'a') as F:
    F.write('Epoch,Step,Training IOU,Validation IOU,Training Loss,Validation Loss\n')


#%%
'''
Training
'''
best_iou = 0.0
for e in range(epochs):
    print('\n\nEpoch : %d of %d' % (e + 1, epochs))
    train_iou_epoch = 0.0
    val_iou_epoch = 0.0
    train_loss_epoch = 0.0
    val_loss_epoch = 0.0

    for s in range(steps):
        print('\nStep : %d of %d' % (s + 1, steps))
        print('Training...')
        train_iou_step = 0.0
        train_loss_step = 0.0
        for j in tqdm(range(s * len_train_step, s * len_train_step + len_train_step, batch_size)):
            input_image_batch = torch.zeros((batch_size, image_channels, image_rows, image_cols), device = device, dtype = torch.float32)
            target_image_batch = torch.zeros((batch_size, 1, image_rows, image_cols), device = device, dtype = torch.float32)
            for k in range(batch_size):
                index = j + k
                input_image_batch[k], target_image_batch[k] = get_input_target(train, index)
            optimizer.zero_grad()
            outputs = net(input_image_batch)
            iou = iou_calc(target_image_batch, outputs)
            loss = tversky_loss_l1_constraint(target_image_batch, outputs)
            loss.backward()
            optimizer.step()
            train_iou_step += iou.item()
            train_loss_step += loss.item()
        train_iou_epoch += train_iou_step
        train_iou_step /= len_train_step
        train_loss_epoch += train_loss_step
        train_loss_step /= len_train_step
        
        print('Validating...')
        val_iou_step = 0.0
        val_loss_step = 0.0
        for j in tqdm(range(s * len_val_step, s * len_val_step + len_val_step, batch_size)):   
            input_image_batch = torch.zeros((batch_size, image_channels, image_rows, image_cols), device = device, dtype = torch.float32)
            target_image_batch = torch.zeros((batch_size, 1, image_rows, image_cols), device = device, dtype = torch.float32)
            for k in range(batch_size):
                index = j + k
                input_image_batch[k], target_image_batch[k] = get_input_target(val, index)
            outputs = net(input_image_batch)
            iou = iou_calc(target_image_batch, outputs)
            loss = tversky_loss_l1_constraint(target_image_batch, outputs)
            val_iou_step += iou.item()
            val_loss_step += loss.item()
        val_iou_epoch += val_iou_step
        val_iou_step /= len_val_step
        val_loss_epoch += val_loss_step
        val_loss_step /= len_val_step
        
        print('Training Loss: %f.\nValidation Loss: %f.' % (train_loss_step, val_loss_step))
        
        if val_iou_step > best_iou:
            print('Validation IOU improved from %f to %f.' % (best_iou, val_iou_step))
            best_iou = val_iou_step
            torch.save(net.state_dict(), write_path + 'segunet_torch.pt')
            print('Weights saved.')
        else:
            print('Validation IOU did not improve.')
            
        with open(write_path + 'segunet_torch_history.csv', 'a') as F:
            F.write('-,%d,%f,%f,%f,%f\n' % (s + 1, train_iou_step, val_iou_step, train_loss_step, val_loss_step))
    
    train_loss_epoch /= len_train
    val_loss_epoch /= len_val
    with open(write_path + 'segunet_torch_history.csv', 'a') as F:
        F.write('%d,-,%f,%f,%f,%f\n' % (e + 1, train_iou_epoch, val_iou_epoch, train_loss_epoch, val_loss_epoch))
    scheduler.step(val_iou_epoch)