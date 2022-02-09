# -*- coding: utf-8 -*-
import torch
import torchvision
import pytorch_lightning as pl
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
from torch.nn import functional as F

import os
import csv
import json
import tqdm
import pickle
import typing
import random 

import numpy as np
from concurrent.futures import ProcessPoolExecutor
from sklearn.neighbors import KNeighborsClassifier

from PIL import Image
import cv2

CLASSES_CNT = 205

def get_classes(path_to_classes_json):
    """
    Считывает из classes.json информацию о классах.
    :param path_to_classes_json: путь до classes.json
    """
    #class_to_idx = ... ### YOUR CODE HERE - словарь, class_to_idx['название класса'] = индекс
    #classes = ... ### YOUR CODE HERE - массив, classes[индекс] = 'название класса'
    class_to_idx = {}
    classes = []
    f = open(path_to_classes_json)
    data = json.load(f)
    for class_name in data:
        class_to_idx[class_name] = data[class_name]['id']
        classes.append(-1)
        
    for class_name in data:
        classes[data[class_name]['id']] = class_name
    return classes, class_to_idx

def read_csv(filename):
    res = {}
    with open(filename) as fhandle:
        reader = csv.DictReader(fhandle)
        for row in reader:
            res[row['filename']] = row['class']
    return res

def calc_metric(y_true, y_pred, cur_type, class_name_to_type):
    ok_cnt = 0
    all_cnt = 0
    for t, p in zip(y_true, y_pred):
        if cur_type == 'all' or class_name_to_type[t] == cur_type:
            all_cnt += 1
            if t == p:
                ok_cnt += 1
    return ok_cnt / max(1, all_cnt)


class DatasetRTSD(torch.utils.data.Dataset):
    """
    Класс для чтения и хранения датасета.
    :param root_folders: список путей до папок с данными
    :param path_to_classes_json: путь до classes.json
    """
    def __init__(self, root_folders, path_to_classes_json) -> None:
        super(DatasetRTSD, self).__init__()
        self.classes, self.class_to_idx = self.get_classes(path_to_classes_json)
        #self.samples = ... ### YOUR CODE HERE - список пар (путь до картинки, индекс класса)
        #self.classes_to_samples = ... ### YOUR CODE HERE - cловарь из списков картинок для каждого класса, classes_to_samples[индекс класса] = [список чисел-позиций картинок в self.samples]
        #self.transform = ... ### YOUR CODE HERE - аугментации + нормализация + ToTensorV2
        self.samples = []
        self.classes_to_samples = {}
        for idx in range(len(self.classes)):
            self.classes_to_samples[idx] = []
        for root_folder in root_folders:
            for class_folder in os.listdir(root_folder):
                class_index = -1
                #class_name = root_folder[-root_folder[::-1].find('/'):]
                #print(class_folder)
                class_index = self.class_to_idx[class_folder]
                imgs = os.listdir((os.path.join(root_folder, class_folder)))
                for img_path in imgs:
                    path = os.path.join(root_folder, class_folder)
                    path = os.path.join(path, img_path)
                    self.samples.append((path, class_index))
                    self.classes_to_samples[class_index].append(len(self.samples) - 1)
        self.transform = A.Compose([
            A.Rotate(limit=30, p=0.5),
            A.HorizontalFlip(p=0.5),
            #A.RandomBrightnessContrast(p=0.3)
        ])
        self.preprocess = torchvision.transforms.Compose([
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __getitem__(self, index):
        """
        Возвращает тройку: тензор с картинкой, путь до файла, номер класса файла (если нет разметки, то "-1").
        """
        path_image, class_image = self.samples[index]
        image = Image.open(path_image).convert('RGB')
        image = np.array(image).astype(np.float32)
        #image = self.transform(image=image)['image']
        x = cv2.resize(image, (64, 64)) / 255.
        x = torch.from_numpy(x.transpose(2, 0, 1))
        tensor_image = self.preprocess(x)
        return tensor_image, path_image, class_image
    
    def __len__(self):
        return len(self.samples)

    @staticmethod
    def get_classes(path_to_classes_json):
        """
        Считывает из classes.json информацию о классах.
        :param path_to_classes_json: путь до classes.json
        """
        #class_to_idx = ... ### YOUR CODE HERE - словарь, class_to_idx['название класса'] = индекс
        #classes = ... ### YOUR CODE HERE - массив, classes[индекс] = 'название класса'
        class_to_idx = {}
        classes = []
        f = open(path_to_classes_json)
        data = json.load(f)
        for class_name in data:
            class_to_idx[class_name] = data[class_name]['id']
            classes.append(-1)
        
        for class_name in data:
            classes[data[class_name]['id']] = class_name
        return classes, class_to_idx


class TestData(torch.utils.data.Dataset):
    """
    Класс для чтения и хранения тестового датасета.
    :param root: путь до папки с картинками знаков
    :param path_to_classes_json: путь до classes.json
    :param annotations_file: путь до .csv-файла с аннотациями (опциональный)
    """
    def __init__(self, root, path_to_classes_json, annotations_file=None):
        super(TestData, self).__init__()
        self.root = root
        #self.samples = ... ### YOUR CODE HERE - список путей до картинок
        #self.transform = ... ### YOUR CODE HERE - преобразования: ресайз + нормализация + ToTensorV2
        self.targets = None ### YOUR CODE HERE - словарь, targets[путь до картинки] = индекс класса
        self.samples = []
        for img in os.listdir(root):
            path = os.path.join(root, img)
            self.samples.append(img)
        self.preprocess = torchvision.transforms.Compose([
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        if annotations_file is not None:
            self.targets = {}
            file_to_class = read_csv(annotations_file)
            _, class_to_idx = self.get_classes(path_to_classes_json)
            for img_name in os.listdir(root):
                path = os.path.join(root, img_name)
                self.targets[img_name] = class_to_idx[file_to_class[img_name]]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        """
        Возвращает тройку: тензор с картинкой, путь до файла, номер класса файла (если нет разметки, то "-1").
        """
        path_image = os.path.join(self.root, self.samples[index])
        image = Image.open(path_image).convert('RGB')
        image = np.array(image).astype(np.float32)
        x = cv2.resize(image, (64, 64)) / 255.
        x = torch.from_numpy(x.transpose(2, 0, 1))
        tensor_image = self.preprocess(x)
        class_image = -1
        if self.targets is not None:
            class_image = self.targets[self.samples[index]]
        return tensor_image, self.samples[index], class_image
    
    @staticmethod
    def get_classes(path_to_classes_json):
        """
        Считывает из classes.json информацию о классах.
        :param path_to_classes_json: путь до classes.json
        """
        #class_to_idx = ... ### YOUR CODE HERE - словарь, class_to_idx['название класса'] = индекс
        #classes = ... ### YOUR CODE HERE - массив, classes[индекс] = 'название класса'
        class_to_idx = {}
        classes = []
        f = open(path_to_classes_json)
        data = json.load(f)
        for class_name in data:
            class_to_idx[class_name] = data[class_name]['id']
            classes.append(-1)
        
        for class_name in data:
            classes[data[class_name]['id']] = class_name
        return classes, class_to_idx


class CustomNetwork(pl.LightningModule):
    """
    Класс, реализующий нейросеть для классификации.
    :param features_criterion: loss-функция на признаки, извлекаемые нейросетью перед классификацией (None когда нет такого лосса)
    :param internal_features: внутреннее число признаков
    """
    def __init__(self, features_criterion = None, internal_features = 1024, transfer=False):
        super(CustomNetwork, self).__init__()
        ### YOUR CODE HERE
        self.model = torchvision.models.resnet50(pretrained=transfer)
        linear_size_in = list(self.model.children())[-1].in_features
        self.model.fc = torch.nn.Sequential(
            torch.nn.Linear(in_features=linear_size_in, out_features=internal_features, bias=True),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=internal_features, out_features=205, bias=True),
        )
        if features_criterion is not None:
            self.features_criterion = features_criterion(2.0)
        else:
            self.features_criterion = features_criterion
        self.feature_net = None
        if self.features_criterion is not None:
            children_list = []
            for c in list(self.model.children())[:-1]:
                children_list.append(c)
            self.feature_net = torch.nn.Sequential(*children_list)

        for child in list(self.model.children()):
          for param in child.parameters():
            param.requires_grad = True

        for child in list(self.model.children())[:-4]:
          for param in child.parameters():
            param.requires_grad = False

    def predict(self, x):
        """
        Функция для предсказания классов-ответов. Возвращает np-массив с индексами классов.
        :param x: батч с картинками
        """
        out = self.model(x)
        return torch.argmax(out, dim=1)
    
    def forward(self, x):
        return F.log_softmax(self.model(x), dim=1)

    def cross_entropy_loss(self, logits, labels):
        return F.nll_loss(logits, labels)

    def training_step(self, train_batch, batch_idx):
        x, _, y = train_batch
        additional_loss = 0
        if self.features_criterion is not None:
            out = []
            for xx in x:
                out.append(self.feature_net(xx.unsqueeze(0)))

            loss = self.features_criterion(out, y)
            logs = {'train_loss': loss}
            self.log('train_loss', loss, on_step=True, prog_bar=True)
            return {'loss': loss, 'log': logs}

        logits = self.forward(x)
        loss = self.cross_entropy_loss(logits, y) 
        logs = {'train_loss': loss}

        acc = torch.sum(logits.argmax(dim=1) == y) / y.shape[0]
        self.log('train_acc', acc, on_step=True, prog_bar=True)
        self.log('train_loss', loss, on_step=True, prog_bar=True)

        return {'loss': loss, 'log': logs}
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.0001)
        return [optimizer]



def train_simple_classifier():
    """Функция для обучения простого классификатора на исходных данных."""
    model = CustomNetwork(transfer=False)
    #model.load_state_dict(torch.load('simple_model.pth'))
    #model.train()
    ds_train = DatasetRTSD(root_folders=['./cropped-train'], path_to_classes_json='./classes.json')
    dl_train = DataLoader(ds_train, batch_size=32, shuffle=True, num_workers=6)
    trainer = pl.Trainer(gpus=0, precision=16, max_epochs=1, checkpoint_callback = False, logger = False)
    trainer.fit(model, dl_train)
    torch.save(model.state_dict(), "simple_model.pth")
    return model


def apply_classifier(model, test_folder, path_to_classes_json):
    """
    Функция, которая применяет модель и получает её предсказания.
    :param model: модель, которую нужно протестировать
    :param test_folder: путь до папки с тестовыми данными
    :param path_to_classes_json: путь до файла с информацией о классах classes.json
    """
    #results. ###  список словарей вида {'filename': 'имя файла', 'class': 'строка-название класса'}
    results = []
    ds_test = TestData(test_folder, path_to_classes_json)
    dl_test = DataLoader(ds_test, batch_size=1, shuffle=True)
    classes, class_to_idx = get_classes(path_to_classes_json)
    model.eval()
    for data in dl_test:
        tensor_image = data[0]
        name = data[1][0]
        pred_num_class = model.predict(tensor_image)
        print(pred_num_class)
        dic = {}
        dic['filename'] = name
        dic['class'] = classes[pred_num_class]
        results.append(dic)
    return results


def test_classifier(model, test_folder, path_to_classes_json, annotations_file):
    """
    Функция для тестирования качества модели.
    Возвращает точность на всех знаках, Recall на редких знаках и Recall на частых знаках.
    :param model: модель, которую нужно протестировать
    :param test_folder: путь до папки с тестовыми данными
    :param annotations_file: путь до .csv-файла с аннотациями (опциональный)
    """
    ### YOUR CODE HERE
    
    total_acc, rare_recall, freq_recall = test_classifier1(results, 'smalltest_annotations.csv', 'classes.json')
    return #total_acc, rare_recall, freq_recall

# train_simple_classifier()

class SignGenerator(object):
    """
    Класс для генерации синтетических данных.
    :param background_path: путь до папки с изображениями фона
    """
    def __init__(self, background_path):
        ### YOUR CODE HERE
        self.back_paths = []
        for img in os.listdir(background_path):
            path = os.path.join(background_path, img)
            self.back_paths.append(path)

    def __len__(self):
        return len(self.back_paths)

    def get_sample(self, icon_path):
        """
        Функция, встраивающая иконку на случайное изображение фона.
        :param icon: Массив с изображением иконки
        """
        #icon = ... ### YOUR CODE HERE
        #bg = ... ### YOUR CODE HERE - случайное изображение фона
        icon = np.array(Image.open(icon_path).convert("RGBA"))

        #resize to random size
        size = np.random.randint(16, 129)
        icon = cv2.resize(icon, (size, size))

        #add padding
        pad = int(np.random.randint(0, 16) / 100 * size) + 1
        icon_with_pad = np.zeros((size+2*pad, size + 2*pad, 4), dtype='uint8')
        icon_with_pad[pad:-pad, pad:-pad] = icon

        #change color
        icon_hsv = cv2.cvtColor(icon_with_pad[:,:,:3], cv2.COLOR_RGB2HSV)
        h = np.random.randint(0, 256)
        icon_hsv[:,:,0]=h
        icon_with_pad[:,:,:3] = cv2.cvtColor(icon_hsv, cv2.COLOR_HSV2RGB)

        #rotate image
        angle = np.random.randint(-15, 16)
        image_center = tuple(np.array(icon_with_pad.shape[1::-1]) / 2)
        rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
        icon = cv2.warpAffine(icon_with_pad, rot_mat, icon_with_pad.shape[1::-1], flags=cv2.INTER_LINEAR)

        # motion blur
        size += 2*pad
        s = np.random.randint(4, 11)
        kernel_motion_blur = np.zeros((s, s))
        kernel_motion_blur[int((s-1)/2), :] = np.ones(s)
        kernel_motion_blur = kernel_motion_blur / s
        # applying the kernel to the input image
        icon[:,:,:3] = cv2.filter2D(icon[:,:,:3], -1, kernel_motion_blur)

        #gaussian blur
        icon[:,:,:3] = cv2.GaussianBlur(icon[:,:,:3], (3,3), 10.0)

        #insert on random background
        i = np.random.randint(0, len(self.back_paths))
        back = np.array(Image.open(self.back_paths[i]).convert("RGB"))
        crop_x1 = np.random.randint(0, back.shape[0] - size)
        crop_y1 = np.random.randint(0, back.shape[1] - size)
        crop = back[crop_x1:crop_x1+size, crop_y1:crop_y1+size]
        mask = icon[:,:,3].astype('float32') / 255.
        mask3d = np.zeros((mask.shape[0], mask.shape[1], 3)).astype('float32')
        mask3d[:,:,0] = mask
        mask3d[:,:,1] = mask
        mask3d[:,:,2] = mask
        image = (mask3d * icon[:,:,:3] + (1 - mask3d) * crop).astype('uint8')
        return image


def generate_one_icon(args):
    """
    Функция, генерирующая синтетические данные для одного класса.
    :param args: Это список параметров: [путь до файла с иконкой, путь до выходной папки, путь до папки с фонами, число примеров каждого класса]
    """
    path_icon = args[0]
    path_out_dir = args[1]
    path_back_dir = args[2]
    n = args[3]
    name = args[4][:-4]
    generator = SignGenerator(path_back_dir)
    for i in range(n):
        image = generator.get_sample(path_icon)
        out_path = os.path.join(path_out_dir, name)
        out_path = os.path.join(out_path, f'{i:05}' +'.png')
        cv2.imwrite(out_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    return 


def generate_all_data(output_folder, icons_path, background_path, samples_per_class = 1000):
    """
    Функция, генерирующая синтетические данные.
    Эта функция запускает пул параллельно работающих процессов, каждый из которых будет генерировать иконку своего типа.
    Это необходимо, так как процесс генерации очень долгий.
    Каждый процесс работает в функции generate_one_icon.
    :param output_folder: Путь до выходной директории
    :param icons_path: Путь до директории с иконками
    :param background_path: Путь до директории с картинками фона
    :param samples_per_class: Количество примеров каждого класса, которые надо сгенерировать
    """
    for img_name in os.listdir(icons_path):
        os.mkdir(os.path.join(output_folder, img_name[:-4]))
    with ProcessPoolExecutor(8) as executor:
        params = [[os.path.join(icons_path, icon_file), output_folder, background_path, samples_per_class, icon_file]
                  for icon_file in os.listdir(icons_path)]
        list(tqdm.tqdm(executor.map(generate_one_icon, params)))


def train_synt_classifier():
    """Функция для обучения простого классификатора на смеси исходных и ситетических данных."""
    ### YOUR CODE HERE
    model = CustomNetwork(transfer=False)
    model.load_state_dict(torch.load("simple_model_with_synt.pth"))
    model.train()
    ds_train = DatasetRTSD(root_folders=['./cropped-train', './generated_images'], path_to_classes_json='./classes.json')
    #ds_train = DatasetRTSD(root_folders=['./new_generated_images'], path_to_classes_json='./classes.json')
    dl_train = DataLoader(ds_train, batch_size=128, shuffle=True, num_workers=6)
    trainer = pl.Trainer(gpus=0, precision=16, max_epochs=1, checkpoint_callback = False, logger = False)
    trainer.fit(model, dl_train)
    torch.save(model.state_dict(), "simple_model_with_synt1.pth")
    return model


class FeaturesLoss(torch.nn.Module):
    """
    Класс для вычисления loss-функции на признаки предпоследнего слоя нейросети.
    """
    def __init__(self, margin: float) -> None:
        super(FeaturesLoss, self).__init__()
        self.margin = margin
        self.eps = 1e-9
    
    def forward(self, outputs, labels):
        n = len(labels)
        sum1, sum2, zn1, zn2 = 0, 0, 0, 0
        for i in range(n):
            for j in range(n):
                if labels[i] == labels[j]:
                    zn1 += 1
                    sum1 += (outputs[i] - outputs[j]).square().sum()
                else:
                    zn2 += 1
                    sum2 += max(0, self.margin - (outputs[i] - outputs[j]).square().sum())
        return 0.5 * (sum1/zn1 + sum2/zn2)


class CustomBatchSampler(torch.utils.data.sampler.Sampler[typing.List[int]]):
    """
    Класс для семплирования батчей с контролируемым числом классов и примеров каждого класса.
    :param data_source: Это датасет RTSD
    :param elems_per_class: Число элементов каждого класса
    :param classes_per_batch: Количество различных классов в одном батче
    """
    def __init__(self, data_source, elems_per_class, classes_per_batch):
        self.data_source = data_source
        self.elems_per_class = elems_per_class
        self.classes_per_batch = classes_per_batch
        self.batch_size = elems_per_class * classes_per_batch
        self.n_batches = len(data_source) // self.batch_size

    def __iter__(self):
        
        class_cnt = len(self.data_source.classes_to_samples)
        batches = []
        for _ in range(self.n_batches):
            classes_idxs = []
            for _ in range(self.classes_per_batch):
                idx = np.random.randint(0, class_cnt)
                while idx in classes_idxs:
                    idx = np.random.randint(0, class_cnt)
                classes_idxs.append(idx)
            idxs = []
            for class_idx in classes_idxs:
                temp = []
                array = self.data_source.classes_to_samples[class_idx]
                n = len(array)
                for _ in range(self.elems_per_class):
                    i = np.random.randint(0, n)
                    while (array[i] in temp)and(len(temp) < n):
                        i = np.random.randint(0, n)
                    temp.append(array[i])
                idxs.extend(temp)
            random.shuffle(idxs)
            batches.append(idxs)
        return iter(batches)


def train_better_model():
    """Функция для обучения классификатора на смеси исходных и ситетических данных с новым лоссом на признаки."""
    model = CustomNetwork(features_criterion=FeaturesLoss, transfer=True)
    #model.load_state_dict(torch.load("simple_model_with_synt.pth"))
    model.train()
    ds_train = DatasetRTSD(root_folders=['./cropped-train', './generated_images'], path_to_classes_json='./classes.json')
    batch_sampler = CustomBatchSampler(ds_train, 4, 32)
    dl_train = DataLoader(ds_train, num_workers=6, batch_sampler=batch_sampler)
    trainer = pl.Trainer(gpus=0, precision='bf16', max_epochs=1, checkpoint_callback = False, logger = False)
    trainer.fit(model, dl_train)
    torch.save(model.state_dict(), "improved_features_model.pth")
    return model


class ModelWithHead:
    """
    Класс, реализующий модель с головой из kNN.
    :param n_neighbors: Количество соседей в методе ближайших соседей
    """
    def __init__(self, n_neighbors):
        self.n_neighbors = n_neighbors
        self.knn = KNeighborsClassifier(n_neighbors=n_neighbors)
        self.net = CustomNetwork(features_criterion=FeaturesLoss)

    def load_nn(self, nn_weights_path):
        """
        Функция, загружающая веса обученной нейросети.
        :param nn_weights_path: Это путь до весов обученной нейросети с улучшенными признаками на предпоследнем слое
        """
        self.net.load_state_dict(torch.load(nn_weights_path))

    def load_head(self, knn_path):
        """
        Функция, загружающая веса kNN (с помощью pickle).
        :param knn_path: Путь, откуда надо прочитать веса kNN
        """
        with open(knn_path, 'rb') as f:
            self.knn = pickle.load(f)

    def eval(self):
        self.net.eval()
        self.net.feature_net.eval()

    def predict(self, imgs):
        """
        Функция для предсказания классов-ответов. Возвращает np-массив с индексами классов.
        :param imgs: батч с картинками
        """
        #features, model_pred = ... ### YOUR CODE HERE - предсказание нейросетевой модели
        #knn_pred = ... ### YOUR CODE HERE - предсказание kNN на features
        knn_pred = []
        features = self.net.feature_net(imgs)
        features = features.detach().numpy()
        features = features / np.linalg.norm(features, axis=1)[:, None]
        print(features.shape)
        features = np.squeeze(features).reshape(1, -1)
        print(features.shape)
        knn_pred = self.knn.predict(features)
        return knn_pred[0]


class IndexSampler(torch.utils.data.sampler.Sampler[int]):
    """
    Класс для семплирования батчей с картинками индекса.
    :param data_source: Это датасет RTSD с синтетическими примерами
    :param examples_per_class: Число элементов каждого класса, которые должны попасть в индекс
    """
    def __init__(self, data_source, examples_per_class) -> None:
        self.data_source = data_source
        self.examples_per_class = examples_per_class

    def __iter__(self):
        """Функция, которая будет генерировать список индексов элементов в батче."""
        idxs = []
        class_cnt = len(self.data_source.classes_to_samples)
        for c in range(class_cnt):
            temp = []
            array = self.data_source.classes_to_samples[c]
            n = len(array)
            for _ in range(self.examples_per_class):
                i = np.random.randint(0, n)
                while (array[i] in temp)and(len(temp) < n):
                    i = np.random.randint(0, n)
                temp.append(array[i])
            idxs.extend(temp)
        random.shuffle(idxs)
        return iter(idxs)


def train_head(nn_weights_path, examples_per_class = 20):
    """
    Функция для обучения kNN-головы классификатора.
    :param nn_weights_path: Это путь до весов обученной нейросети с улучшенными признаками на предпоследнем слое
    :param examples_per_class: Число элементов каждого класса, которые должны попасть в индекс
    """
    torch.multiprocessing.set_sharing_strategy('file_system')
    knn = KNeighborsClassifier(n_neighbors=examples_per_class)
    net = CustomNetwork(features_criterion=FeaturesLoss)
    net.load_state_dict(torch.load(nn_weights_path))
    net.eval()
    net.feature_net.eval()
    ds_train = DatasetRTSD(root_folders=['./new_generated_images'], path_to_classes_json='./classes.json')
    sampler = IndexSampler(ds_train, examples_per_class)
    dl_train = DataLoader(ds_train, num_workers=6, sampler=sampler)
    data_x = []
    data_y = []
    with torch.no_grad():
        for data in dl_train:
            inputs, _, labels = data[0], data[1], data[2]
            print(labels)
            feats = net.feature_net(inputs)
            feats = feats.detach().numpy()
            feats = feats / np.linalg.norm(feats, axis=1)[:, None]
            labels = labels.detach().numpy()
            data_x.append(feats)
            data_y.append(labels)
    data_x = np.concatenate(data_x, axis=0)
    data_y = np.concatenate(data_y, axis=0)
    data_x = np.squeeze(data_x)
    knn.fit(data_x, data_y)
    #torch.multiprocessing.set_sharing_strategy('file_system')
    with open('knn_model.bin', 'wb') as f:
        pickle.dump(knn, f)
    return

# train_better_model()
# train_synt_classifier()
# generate_all_data('./generated_images', './icons', './background_images', samples_per_class = 300)
# train_head('improved_features_model.pth')