from pytorch_lightning.callbacks import progress
from torch.nn.modules.activation import LogSoftmax
from torch.utils.data import Dataset
import os
import numpy as np
from PIL import Image
import torch
import torchvision
from torchvision import transforms
from torch import nn
import pytorch_lightning as pl
from torch.nn import functional as F
from torch.utils.data import DataLoader
import cv2
import albumentations as A
BASE_LR = 1e-4


BATCH_SIZE = 32
MAX_EPOCHS = 15

def crop_center(img, a):
    x, y = img.shape[0], img.shape[1]
    x = x//2-(a//2)
    y = y//2-(a//2)    
    return img[x:x+a,y:y+a]

preprocess = transforms.Compose([
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
class MyCustomDataset(Dataset):
    def __init__(self, 
                 mode, 
                 data_dir, 
                 data_gt = None,
                 fraction: float = 0.8, 
                 transform = None,
                ):
        
        ## list of tuples: (img_path, label)
        self._items = [] 
        self._mode = mode
        ## will use it later for augmentations
        self._data_dir = data_dir
        self._transform = transform

        ## we can't store all the images in memory at the same time, 
        ## because sometimes we have to work with very large datasets
        ## so we will only store data paths
        ## (also this is convenient for further augmentations)
        cl_dir = data_dir
        lst_dir = sorted(os.listdir(cl_dir))
        if mode == 'train':
            img_pathes = []
            for i in range(len(lst_dir)):
                if((i % 50) <= int(50 * fraction)):
                    img_pathes.append(lst_dir[i])
        elif mode == 'val':
            img_pathes = []
            for i in range(len(lst_dir)):
                if ((i % 50) > int(50 * fraction)):
                    img_pathes.append(lst_dir[i])
        elif(mode == 'test'):
            img_pathes = lst_dir[:]

        for img_path in img_pathes:
            if(mode == 'test'):
                data = None
            else:
                data = data_gt[img_path]
            self._items.append((os.path.join(cl_dir, img_path), data))

    def __len__(self):
        return len(self._items)

    def __getitem__(self, index):

        img, label = self._items[index]
        img_path = os.path.join(img)
        
        ## read image 
        image = Image.open(img_path).convert('RGB')
        image = np.array(image).astype(np.float32)
        if(self._transform != None):
            image = self._transform(image = image)['image']
        ## resize

        Standart_size = 256
        x = cv2.resize(image, (Standart_size, Standart_size))
        x = crop_center(x, 224)
        # for i in range(3):
        #     x[..., i] = (x[..., i] - x[..., i].mean()) / x[..., i].std()
        
        ## ToTensor
        x = preprocess(torch.from_numpy(x.transpose(2, 0, 1)))
        
        if(self._mode == 'test'):
            return x
        else:
            return x, label

class MobileNetClassifierl(pl.LightningModule):
    def __init__(self, num_classes, transfer=True):
        super().__init__()        

        self.mobile_model = torchvision.models.mobilenet_v2(pretrained=transfer, progress = True)
        #print(self.mobile_model)
        linear_size_in = list(self.mobile_model.children())[-1][-1].in_features
        linear_size_out = list(self.mobile_model.children())[-1][-1].out_features
        self.mobile_model.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=False),
            nn.AdaptiveAvgPool1d(linear_size_in),
            nn.Linear(in_features=linear_size_in, out_features=linear_size_out, bias=True),
            nn.BatchNorm1d(linear_size_out),
            nn.ReLU(),
            nn.Linear(in_features=linear_size_out, out_features=num_classes, bias=True)
        )
        # print(self.mobile_model)
        for child in list(self.mobile_model.children()):
                for param in child.parameters():
                    param.requires_grad = True

        for child in list(self.mobile_model.children())[:-6]:
            for param in child.parameters():
                param.requires_grad = False
 
    def forward(self, x):
        return F.log_softmax(self.mobile_model(x), dim=1)
    
class MyModel(pl.LightningModule):

    def __init__(self, lr_rate=BASE_LR, freeze='most'):
        super(MyModel, self).__init__()
        
        self.model = MobileNetClassifierl(50, True)

        self.lr_rate = lr_rate

    def forward(self, x):
      return self.model(x)

    def cross_entropy_loss(self, logits, labels):
        return F.nll_loss(logits, labels)

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        logits = self.forward(x)
        loss = self.cross_entropy_loss(logits, y)

        logs = {'train_loss': loss}

        acc = torch.sum(logits.argmax(dim=1) == y) / y.shape[0]
        self.log('train_acc', acc, on_step=True, prog_bar=True)
        self.log('train_loss', loss, on_step=True, prog_bar=True)

        return {'loss': loss, 'log': logs}

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        logits = self.forward(x)
        loss = self.cross_entropy_loss(logits, y)
        acc = torch.sum(logits.argmax(dim=1) == y) / y.shape[0]

        self.log('val_loss', loss, on_step=True, on_epoch=False)
        self.log('val_acc', acc, on_step=True, on_epoch=False)

        return {'val_loss': loss, 'val_acc': acc}

    def test_step(self, val_batch, batch_idx):
        x, y = val_batch
        logits = self.forward(x)
        loss = self.cross_entropy_loss(logits, y)
        acc = torch.sum(logits.argmax(dim=1) == y) / y.shape[0]

        self.log('test_loss', loss, on_step=True, on_epoch=False)
        self.log('test_acc', acc, on_step=True, on_epoch=False)

        return {'test_loss': loss, 'test_acc': acc}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['val_acc'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss, 'val_acc': avg_acc}
        print('Accuracy: ', round(float(avg_acc), 3))
        self.log('val_acc', avg_acc, on_epoch=True, on_step=False)
        return {'val_loss': avg_loss, 'val_acc': avg_acc, 'log': tensorboard_logs}

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['test_acc'] for x in outputs]).mean()
        tensorboard_logs = {'test_loss': avg_loss, 'test_acc': avg_acc}
        self.log('test_acc', avg_acc, on_epoch=True, on_step=False)
        return {'test_loss': avg_loss, 'test_acc': avg_acc, 'log': tensorboard_logs}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr_rate)
        return [optimizer]

MyAugmetation = A.Compose([
    A.Rotate(limit = 10),
    A.HorizontalFlip(p = 0.3),
    A.RandomBrightnessContrast(p = 0.3)
])

def train_classifier(train_gt, train_img_dir, fast_train = False):
    gpus_count = 0
    max_epochs = MAX_EPOCHS

    ds_train = MyCustomDataset(mode = 'train', data_dir = train_img_dir, data_gt = train_gt, transform = MyAugmetation)
    ds_val = MyCustomDataset(mode = 'val', data_dir = train_img_dir, data_gt = train_gt)

    ## Init train and val dataloaders
    dl_train = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True)
    dl_val = DataLoader(ds_val, batch_size=BATCH_SIZE, shuffle=False)
    
    model = MyModel()

    if(fast_train == True):
        max_epochs = 1
        trainer = pl.Trainer(
            max_epochs=max_epochs,
            gpus=gpus_count,
            checkpoint_callback = False,
            logger = False)

        trainer.fit(model, dl_train, dl_val)
    else:
        trainer = pl.Trainer(
            max_epochs=max_epochs,
            gpus=gpus_count)
        trainer.fit(model, dl_train, dl_val)
        trainer.save_checkpoint("birds_model.ckpt")
    
    test_accuracy = trainer.test(model, dl_val)[0]['test_acc']
    print('Test accuracy: ', round(test_accuracy, 3))
    return model

def classify(model_filename, test_img_dir):
    ans = {}
    for path in sorted(os.listdir(test_img_dir)):
        ans[path] = 0      
    ds_test = MyCustomDataset(mode = 'test', data_dir = test_img_dir)
    
    model = MyModel.load_from_checkpoint(checkpoint_path=model_filename)
    model.eval()
    dl_test = DataLoader(ds_test, batch_size=len(ds_test), shuffle=False, num_workers=4)
                
    elems = list(ans.keys())

    with torch.no_grad():
        for i in range(len(dl_test.dataset)):
            im = dl_test.dataset[i] 
            new_im = im.reshape(1, im.shape[0], im.shape[1], im.shape[2]).type(torch.FloatTensor)
            new_val = model(new_im).detach().numpy()
            ans[elems[i]] = np.argmax(new_val)
                    
    return ans

    
    