import pytorch_lightning as pl
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
import torchvision 
import cv2
import os
import numpy as np
import albumentations as A
from torch.utils.data import Dataset
from typing import Optional
# from PIL import Image
from skimage.io import imread
from torch import nn
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

class MyDataset(Dataset):
    def __init__(self, mode, 
                data_dir, 
                data_gt, 
                fraction=0.8, transform=None):

        self._items = [] 

        data_gt = np.array(list(data_gt.values())).astype(np.float32)
        lst_dir = sorted(os.listdir(data_dir))

        if mode == 'train':
            lst_dir = lst_dir[:int(fraction * len(lst_dir))]
            data_gt = data_gt[:int(fraction * len(data_gt))]
        elif mode == 'val':
            lst_dir = lst_dir[int(fraction * len(lst_dir)):]
            data_gt = data_gt[int(fraction * len(data_gt)):]
            
        for i in range(len(lst_dir)):
            image = imread(data_dir + '/' + lst_dir[i])
            image = np.array(image).astype(np.float32)

            data_gt[i, 0::2] = data_gt[i, 0::2] * 100 / image.shape[1]
            data_gt[i, 1::2] = data_gt[i, 1::2] * 100 / image.shape[0]

            image = cv2.resize(image, (100, 100))
            if len(image.shape) == 2:
                image = np.stack((image, image, image)).transpose(1, 2, 0)

            for channel in range(3):
                image[:, :, channel] = (image[:, :, channel] - image[:, :, channel].mean()) / image[:, :, channel].std()

            self._items.append((image, data_gt[i]))

            if transform != None:
                for _ in range(2):
                    label = [(data_gt[i, j], data_gt[i, j+1]) for j in range(0, data_gt[i].shape[0], 2)]
                    aug       = transform(image=image, keypoints=label)
                    aug_img   = aug['image']
                    aug_label = aug['keypoints']
                    aug_label = np.array( sum([[x,y] for (x,y) in aug_label],[]) ).astype(np.float32)
                    self._items.append((aug_img, aug_label))

    def __len__(self):
        return len(self._items)

    def __getitem__(self, index):

        image, label = self._items[index]
        return torch.from_numpy(image.transpose(2, 0, 1)), label

class MyModel(pl.LightningModule):
    # REQUIRED
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 32, 3,     padding='same', bias=False)
        self.batch1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, 3,    padding='same', bias=False)
        self.batch2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(32, 64, 3,    padding='same', bias=False)
        self.batch3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, 3,    padding='same', bias=False)
        self.batch4 = nn.BatchNorm2d(64)
        self.pool4 = nn.MaxPool2d(2, 2)
        self.conv5 = nn.Conv2d(64, 96, 3,    padding='same', bias=False)
        self.batch5 = nn.BatchNorm2d(96)
        self.conv6 = nn.Conv2d(96, 96, 3,    padding='same', bias=False)
        self.batch6 = nn.BatchNorm2d(96)
        self.pool6 = nn.MaxPool2d(2, 2)
        self.conv7 = nn.Conv2d(96, 128, 3,   padding='same', bias=False)
        self.batch7 = nn.BatchNorm2d(128)
        self.conv8 = nn.Conv2d(128, 128, 3,  padding='same', bias=False)
        self.batch8 = nn.BatchNorm2d(128)
        self.pool8 = nn.MaxPool2d(2, 2)
        self.conv9 = nn.Conv2d(128, 256, 3,  padding='same', bias=False)
        self.batch9 = nn.BatchNorm2d(256)
        self.conv10 = nn.Conv2d(256, 256, 3, padding='same', bias=False)
        self.batch10 = nn.BatchNorm2d(256)
        self.pool10 = nn.MaxPool2d(2, 2)
        self.conv11 = nn.Conv2d(256, 512, 3, padding='same', bias=False)
        self.batch11 = nn.BatchNorm2d(512)
        self.conv12 = nn.Conv2d(512, 512, 3, padding='same', bias=False)
        self.batch12 = nn.BatchNorm2d(512)
        self.fc1 = nn.Linear(4608, 256)
        self.dropout = nn.Dropout(p=0.1)
        self.fc2 = nn.Linear(256, 28)
        self.f_act = nn.LeakyReLU(0.1)

        self.loss = F.mse_loss
    
    # REQUIRED
    def forward(self, x):
        """ Use for inference only (separate from training_step). """
        x = self.batch1(self.f_act(self.conv1(x)))
        x = self.pool2(self.batch2(self.f_act(self.conv2(x))))
        x = self.batch3(self.f_act(self.conv3(x)))
        x = self.pool4(self.batch4(self.f_act(self.conv4(x))))
        x = self.batch5(self.f_act(self.conv5(x)))
        x = self.pool6(self.batch6(self.f_act(self.conv6(x))))
        x = self.batch7(self.f_act(self.conv7(x)))
        x = self.pool8(self.batch8(self.f_act(self.conv8(x))))
        x = self.batch9(self.f_act(self.conv9(x)))
        x = self.pool10(self.batch10(self.f_act(self.conv10(x))))
        x = self.batch11(self.f_act(self.conv11(x)))
        x = self.batch12(self.f_act(self.conv12(x)))
        
        x = torch.flatten(x, 1)     
        
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x
    
    # REQUIRED
    def training_step(self, batch, batch_idx):
        """the full training loop"""
        x, y = batch

        loss = self.loss(self(x), y)
        return {'loss': loss}
    
    # REQUIRED
    def configure_optimizers(self):
        """ Define optimizers and LR schedulers. """
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3, weight_decay=5e-4)
        
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=5, verbose=True)
        lr_dict = {
            "scheduler": lr_scheduler,
            "interval": "epoch",
            "frequency": 1,
            "monitor": "val_loss"
        } 
        
        return [optimizer], [lr_dict]
    
    # OPTIONAL
    def validation_step(self, batch, batch_idx):
        """the full validation loop"""
        x, y = batch

        loss = self.loss(self(x), y)
        return {'val_loss': loss}

    # OPTIONAL
    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        print(f'| Train_loss: {avg_loss:.2f}' )
        self.log('train_loss', avg_loss, prog_bar=True, on_epoch=True, on_step=False)
     
    # OPTIONAL
    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        print(f'[Epoch {self.trainer.current_epoch:3}] Val_loss: {avg_loss:.2f}', end= ' ')
        self.log('val_loss', avg_loss, prog_bar=True, on_epoch=True, on_step=False)


MyAurmentation = A.Compose( [
        A.Rotate(limit=80, p=1), ],
        keypoint_params=A.KeypointParams(format='xy', remove_invisible=False) )

EarlyStop = EarlyStopping(monitor = "val_loss", 
                                mode = "min", 
                                patience = 10, 
                                verbose = True)


def train_detector(train_gt, train_img_dir, fast_train=True):

    gpus_count = 0
    max_epochs = 30

    model = MyModel()

    ds_val = MyDataset(mode='val', data_dir=train_img_dir, data_gt=train_gt)

    if fast_train == True:
        ds_train = MyDataset(mode='train', fraction=0.2, data_dir=train_img_dir, data_gt=train_gt)

        dl_train = DataLoader(ds_train, batch_size=100, shuffle=True)
        dl_val = DataLoader(ds_val, batch_size=100, shuffle=False)

        trainer = pl.Trainer(max_epochs=1, 
                            checkpoint_callback=False, 
                            logger=False)
        trainer.fit(model, dl_train, dl_val)
    else:
        ds_train = MyDataset(mode='train', data_dir=train_img_dir, data_gt=train_gt, transform=MyAurmentation)

        dl_train = DataLoader(ds_train, batch_size=100, shuffle=True)
        dl_val = DataLoader(ds_val, batch_size=100, shuffle=False)

        trainer = pl.Trainer(max_epochs=max_epochs, 
                            gpus=gpus_count, 
                            callbacks=[EarlyStop] )

        trainer.fit(model, dl_train, dl_val)
        trainer.save_checkpoint("facepoints_model.ckpt")
    
    return model

def detect(model_filename, test_img_dir):
    ans = {}

    model = MyModel.load_from_checkpoint(model_filename)
    model.eval()

    lst_dir = sorted(os.listdir(test_img_dir))
    
    for i in lst_dir:
        image = imread(test_img_dir + '/' + i)
        image = np.array(image).astype(np.float32)
        h, w = image.shape[0], image.shape[1]

        image = cv2.resize(image, (100, 100))

        if len(image.shape) == 2:
            image = np.stack((image, image, image)).transpose(1, 2, 0)

        for channel in range(3):
            image[:, :, channel] = (image[:, :, channel] - image[:, :, channel].mean()) / image[:, :, channel].std()

        image = torch.from_numpy(image.transpose(2, 0, 1))
        image = image.reshape(1, image.shape[0], image.shape[1], image.shape[2])
        new_val = model(image).detach().numpy()

        new_val[0, 0::2] = new_val[0, 0::2] * w / 100
        new_val[0, 1::2] = new_val[0, 1::2] * h / 100

        ans[i] = new_val[0]

    return ans