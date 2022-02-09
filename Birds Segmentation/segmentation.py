from numpy.core.numeric import False_
import torch
import torch.nn as nn
import torchvision.models
import pytorch_lightning as pl
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from torchvision import transforms
import torchvision.transforms as T
import numpy as np
import os
from PIL import Image
from tqdm import tqdm
import cv2
import albumentations
import albumentations as A
from albumentations.pytorch import ToTensorV2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

normalized = transforms.Compose([
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
class SimDataset(Dataset):
    def __init__(self, data_path, mode, transform = None):
        self.data_dir = data_path
        self._transform = transform
        self.samples = []
        image = os.path.join(data_path, 'images')
        gt = os.path.join(data_path, 'gt')
        
        if(mode == 'train'):
            data_dir = sorted(os.listdir(image))
            dir = data_dir[:-1]
            for i in dir:
                path = os.path.join(image, i)
                for j in os.listdir(path):
                    path_gt = os.path.join(gt, i, j[:-4] + '.png')
                    path_image = os.path.join(image, i, j)
                    self.samples.append((path_image, path_gt))
        else:
            data_dir = sorted(os.listdir(image))
            dir = data_dir[-1]
            path = os.path.join(image, dir)
            for j in os.listdir(path):
                path_gt = os.path.join(gt, dir, j[:-4] + '.png')
                path_image = os.path.join(image, dir, j)
                self.samples.append((path_image, path_gt))

    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):

        img_path, mask_path = self.samples[idx]
        img = np.array(Image.open(img_path).convert('RGB'), dtype = np.float32)
        mask = np.array(Image.open(mask_path).convert('L'), dtype = np.float32)

        image = cv2.resize(img, (192, 192))
        mask  = cv2.resize(mask, (192, 192))

        if(self._transform is not None):
            transformed = self._transform(image = image, mask = mask)
            image = transformed['image']
            mask = transformed['mask']

        image = image / 255.
        mask = mask / 255.
        image = torch.from_numpy(image.transpose(2, 0, 1))
        image = normalized(image)
        return image.to(DEVICE), torch.from_numpy(mask).unsqueeze(0).to(DEVICE)

def convrelu(in_channels, out_channels, kernel, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, padding=padding),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )

class ResNetUNet(nn.Module):
    def __init__(self, n_class = 2):
        super().__init__()

        self.base_model = torchvision.models.resnet18(pretrained = False)
        self.base_layers = list(self.base_model.children())

        self.layer0 = nn.Sequential(*self.base_layers[:3]) # size=(N, 64, x.H/2, x.W/2)
        self.layer0_1x1 = convrelu(64, 64, 1, 0)
        self.layer1 = nn.Sequential(*self.base_layers[3:5]) # size=(N, 64, x.H/4, x.W/4)
        self.layer1_1x1 = convrelu(64, 64, 1, 0)
        self.layer2 = self.base_layers[5]  # size=(N, 128, x.H/8, x.W/8)
        self.layer2_1x1 = convrelu(128, 128, 1, 0)
        self.layer3 = self.base_layers[6]  # size=(N, 256, x.H/16, x.W/16)
        self.layer3_1x1 = convrelu(256, 256, 1, 0)
        self.layer4 = self.base_layers[7]  # size=(N, 512, x.H/32, x.W/32)
        self.layer4_1x1 = convrelu(512, 512, 1, 0)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv_up3 = convrelu(256 + 512, 512, 3, 1)
        self.conv_up2 = convrelu(128 + 512, 256, 3, 1)
        self.conv_up1 = convrelu(64 + 256, 256, 3, 1)
        self.conv_up0 = convrelu(64 + 256, 128, 3, 1)

        self.conv_original_size0 = convrelu(3, 64, 3, 1)
        self.conv_original_size1 = convrelu(64, 64, 3, 1)
        self.conv_original_size2 = convrelu(64 + 128, 64, 3, 1)

        self.conv_last = nn.Conv2d(64, 1, 1)

    def forward(self, input):
        x_original = self.conv_original_size0(input)
        x_original = self.conv_original_size1(x_original)

        layer0 = self.layer0(input)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)

        layer4 = self.layer4_1x1(layer4)
        x = self.upsample(layer4)
        layer3 = self.layer3_1x1(layer3)
        x = torch.cat([x, layer3], dim=1)
        x = self.conv_up3(x)

        x = self.upsample(x)
        layer2 = self.layer2_1x1(layer2)
        x = torch.cat([x, layer2], dim=1)
        x = self.conv_up2(x)

        x = self.upsample(x)
        layer1 = self.layer1_1x1(layer1)
        x = torch.cat([x, layer1], dim=1)
        x = self.conv_up1(x)

        x = self.upsample(x)
        layer0 = self.layer0_1x1(layer0)
        x = torch.cat([x, layer0], dim=1)
        x = self.conv_up0(x)

        x = self.upsample(x)
        x = torch.cat([x, x_original], dim=1)
        x = self.conv_original_size2(x)

        out = self.conv_last(x)

        return out

def dice_loss(pred, target, smooth = 1.):
    pred = pred.contiguous()
    target = target.contiguous()    
    intersection = (pred * target).sum(dim=2).sum(dim=2)
    loss = (1 - ((2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) +\
                                                 target.sum(dim=2).sum(dim=2) + smooth)))

    return loss.mean()

class MyModel(pl.LightningModule):
    # REQUIRED
    def __init__(self, num_classes = 2):
        super().__init__()
        """ Define computations here. """
        
        self.model = ResNetUNet(num_classes)
        
        # freeze backbone layers
        for l in self.model.base_layers:
            for param in l.parameters():
                param.requires_grad = False
        
        
        self.bce_weight = 0.9
    
    # REQUIRED
    def forward(self, x):
        """ Use for inference only (separate from training_step). """
        x = self.model(x)
        return x
    
    
    # REQUIRED
    def training_step(self, batch, batch_idx):
        """the full training loop"""
        x, y = batch

        y_logit = self(x)        
        bce = F.binary_cross_entropy_with_logits(y_logit, y)
        
        pred = torch.sigmoid(y_logit)
        dice = dice_loss(pred, y)

        loss = bce * self.bce_weight + dice * (1 - self.bce_weight) * y.size(0)
        return {'loss': loss}#, 'acc': acc}
    
    # REQUIRED
    def configure_optimizers(self):
        """ Define optimizers and LR schedulers. """
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4, weight_decay=5e-4)
        # optimizer = torch.optim.SGD(self.parameters(), lr=1e-3, momentum=0.9)
        
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                                  mode='min', 
                                                                  factor=0.1, 
                                                                  patience=1, 
                                                                  verbose=True)
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

        y_logit = self(x)        
        bce = F.binary_cross_entropy_with_logits(y_logit, y)
        
        pred = torch.sigmoid(y_logit)
        dice = dice_loss(pred, y)

        loss = bce * self.bce_weight + dice * (1 - self.bce_weight) * y.size(0)
        return {'val_loss': loss, 'logs':{'dice':dice, 'bce': bce}}

    # OPTIONAL
    def training_epoch_end(self, outputs):
        """log and display average train loss and accuracy across epoch"""
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        
        print(f"| Train_loss: {avg_loss:.3f}" )
        self.log('train_loss', avg_loss, prog_bar=True, on_epoch=True, on_step=False)
     
    # OPTIONAL
    def validation_epoch_end(self, outputs):
        """log and display average val loss and accuracy"""
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        
        avg_dice = torch.stack([x['logs']['dice'] for x in outputs]).mean()
        avg_bce = torch.stack([x['logs']['bce'] for x in outputs]).mean()
        
        print(f"[Epoch {self.trainer.current_epoch:3}] Val_loss: {avg_loss:.3f}, Val_dice: {avg_dice:.3f}, Val_bce: {avg_bce:.3f}", end= " ")
        self.log('val_loss', avg_loss, prog_bar=True, on_epoch=True, on_step=False)

def get_model():
    return MyModel()

MyAugmetation = A.Compose([
                A.Rotate(limit = 30, p = 1.0),
                A.HorizontalFlip(p = 0.5),
])

def train_model(train_data_path):
    model = MyModel()
    model.to(DEVICE)
    train_set = SimDataset(train_data_path, mode = 'train', transform = MyAugmetation)
    val_set = SimDataset(train_data_path, mode = 'val')

    train_loader = DataLoader(train_set, batch_size = 8, shuffle=True)
    val_loader = DataLoader(val_set, batch_size = 8, shuffle=False)
    
    trainer = pl.Trainer(
            max_epochs = 3,
            gpus = 1)

    trainer.fit(model, train_loader, val_loader)
    normalized_state_dict = {}
    for name, weight in model.state_dict().items():
        mask = (weight.abs() > 1e-32).float()
        normalized_state_dict[name] = weight * mask
    torch.save(normalized_state_dict, 'segmentation_model.pth')

def predict(model, img_path):
    model.eval()
    image = np.array(Image.open(img_path).convert('RGB'), dtype = np.float32)
    h, w = image.shape[0], image.shape[1]
    image = cv2.resize(image, (192, 192)) / 255.
    tensor_image = torch.from_numpy(image.transpose(2, 0, 1))
    tensor_image = normalized(tensor_image).unsqueeze(0)

    res = torch.sigmoid(model(tensor_image))
    res = cv2.resize(res.squeeze().data.cpu().numpy(), (w, h))
    return res