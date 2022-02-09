# ============================== 1 Classifier model ============================
import torch
from torch import nn
import numpy as np
def get_cls_model(input_shape):
    """
    :param input_shape: tuple (n_rows, n_cols, n_channels)
            input shape of image for classification
    :return: nn model for classification
    """
    # your code here \/
    from torch.nn import Sequential

    model = Sequential(
        nn.Conv2d(in_channels = input_shape[2], out_channels=16, kernel_size=3),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),
        nn.BatchNorm2d(16),
        nn.Dropout(p = 0.2),

        nn.Conv2d(in_channels = 16, out_channels=32, kernel_size=3),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),
        nn.BatchNorm2d(32),
        nn.Dropout(p = 0.2),

        nn.Conv2d(in_channels = 32, out_channels=64, kernel_size=3),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),
        nn.BatchNorm2d(64),
        nn.Dropout(p = 0.2),

        nn.Flatten(),

        nn.Linear(in_features= 30 * 64, out_features= 64),
        nn.ReLU(),
        nn.Linear(in_features= 64, out_features= 2),
        nn.Softmax()
    )
    return model
    # your code here /\

def fit_cls_model(X, y):
    """
    :param X: 4-dim tensor with training images
    :param y: 1-dim tensor with labels for training
    :return: trained nn model
    """
    # your code here \/
    model = get_cls_model((40, 100, 1))
    optimizer = torch.optim.SGD(model.parameters(), lr = 1e-3)
    cross_entopy_loss = nn.CrossEntropyLoss()
    # train model
    model.train()
    for epoch in range(3):
        for data, label in zip(X, y):
            data = data.reshape((1, data.shape[0], data.shape[1], data.shape[2]))
            output = model(data)
            loss = cross_entopy_loss(output,  torch.Tensor([label]).to(torch.long))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    #torch.save(model.state_dict(), "classifier_model.pth")
    return model
    # your code here /\


# ============================ 2 Classifier -> FCN =============================
def get_detection_model(cls_model):
    """
    :param cls_model: trained cls model
    :return: fully convolutional nn model with weights initialized from cls
             model
    """
    import torch
    from torch.nn import Sequential
    import torch.nn as nn

    # your code here \/
    detection_model = Sequential(
        nn.Conv2d(1, 32, padding=1, kernel_size=3),
        #nn.BatchNorm2d(32),
        nn.LeakyReLU(),
        nn.MaxPool2d(2, stride=2),
        nn.Conv2d(32, 64, padding=1, kernel_size=3),
        #nn.BatchNorm2d(64),
        nn.LeakyReLU(),
        nn.MaxPool2d(2, stride=2),
        nn.Conv2d(64, 128, padding=1, kernel_size=3),
        #nn.BatchNorm2d(128),
        nn.LeakyReLU(),
        nn.MaxPool2d(2, stride=2),
        nn.Conv2d(128, 1024, kernel_size=(5, 12)),
        nn.LeakyReLU(),
        nn.Conv2d(1024, 128, kernel_size=1),
        nn.LeakyReLU(),
        nn.Conv2d(128, 2, kernel_size=1),
        nn.Softmax(dim=1)
    )
    layers = [0, 3, 6]
    detection_model.eval()
    with torch.no_grad():
        for i in layers:
            detection_model[i].weight = cls_model[i].weight
        detection_model[9].weight.data = cls_model[10].weight.reshape((1024, 128, 5, 12))
        detection_model[11].weight.data = cls_model[12].weight.reshape((128, 1024, 1, 1))
        detection_model[13].weight.data = cls_model[14].weight.reshape((2, 128, 1, 1))
    return detection_model
    # your code here /\
# ============================ 3 Simple detector ===============================
def get_detections(detection_model, dictionary_of_images):
    """
    :param detection_model: trained fully convolutional detector model
    :param dictionary_of_images: dictionary of images in format
        {filename: ndarray}
    :return: detections in format {filename: detections}. detections is a N x 5
        array, where N is number of detections. Each detection is described
        using 5 numbers: [row, col, n_rows, n_cols, confidence].
    """
    # your code here \/
    import numpy as np
    import torch

    threshold = 0.35
    preds = {}
    detection_model.eval()
    for filename in dictionary_of_images:
        detections = []
        image = dictionary_of_images[filename]
        img_shape = image.shape
        image = torch.FloatTensor(np.pad(image, ((0, 220 - img_shape[0]), (0, 370 - img_shape[1]))))
        image = image.reshape(1, 1, 220, 370)
        pred = detection_model(image).detach()[0][1]
        pred_shape = (img_shape[0] // 8 - 4, img_shape[1] // 8 - 11)
        pred = pred[:pred_shape[0], :pred_shape[1]]
        for m in range(pred.shape[0]):
            for n in range(pred.shape[1]):
                if pred[m, n].item() > threshold:
                    detections.append([m * 8, n * 8, 40, 100, pred[m, n].item()])
        preds[filename] = detections
    return preds
    # your code here /\

# =============================== 5 IoU ========================================
def calc_iou(first_bbox, second_bbox):
    """
    :param first bbox: bbox in format (row, col, n_rows, n_cols)
    :param second_bbox: bbox in format (row, col, n_rows, n_cols)
    :return: iou measure for two given bboxes
    """
    # your code here \/
    xA = max(first_bbox[0], second_bbox[0])
    yA = max(first_bbox[1], second_bbox[1])
    xB = min(first_bbox[0] + first_bbox[2], second_bbox[0] + second_bbox[2])
    yB = min(first_bbox[1] + first_bbox[3], second_bbox[1] + second_bbox[3])

    interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
    
    if interArea == 0:
        return 0

    boxAArea = abs(first_bbox[2] * first_bbox[3])
    boxBArea = abs(second_bbox[2] * second_bbox[3])
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou
    # your code here /\


# =============================== 6 AUC ========================================
def calc_auc(pred_bboxes, gt_bboxes):
    """
    :param pred_bboxes: dict of bboxes in format {filename: detections}
        detections is a N x 5 array, where N is number of detections. Each
        detection is described using 5 numbers: [row, col, n_rows, n_cols,
        confidence].
    :param gt_bboxes: dict of bboxes in format {filenames: bboxes}. bboxes is a
        list of tuples in format (row, col, n_rows, n_cols)
    :return: auc measure for given detections and gt
    """
    # your code here \/
    # 1
    tp, fp = [], []
    del_id = []
    gt_len = np.sum([len(x) for x in gt_bboxes.values()])
    for filename in pred_bboxes:
        gt = gt_bboxes[filename]
        detections = list(pred_bboxes[filename])
        detections.sort(key=lambda x: x[4], reverse=True)

        for box in detections:
            max_i = -1
            max_iou = 0.5
            for i, gt_box in enumerate(gt):
                if(i in del_id):
                    continue
                iou = (calc_iou(box, gt_box))
                if (iou >= max_iou):
                    max_i = i
                    max_iou = iou
            if max_i >= 0:
                tp.append(box)
                del_id.append(max_i)
            else:
                fp.append(box)
    # 2
    union = tp + fp
        
    # 3, 4
    union.sort(key=lambda x: x[4])
    union = np.array(union)
    tp.sort(key=lambda x: x[4])
    tp = np.array(tp)
    precision = []
    recall = []
    for c in np.concatenate([[0], union[:, -1], [1]]):
        total_cnt = np.count_nonzero(union[:, -1] >= c)
        tp_cnt = np.count_nonzero(tp[:, -1] >= c)
        p = 1 if total_cnt == 0 else tp_cnt / total_cnt
        r = tp_cnt / gt_len
        precision.append(p)
        recall.append(r)
    # 5
    auc = 0
    for i in range(len(recall) - 1):
        a, b = recall[i], recall[i + 1]
        c, d = precision[i], precision[i + 1]
        auc += 0.5 * (c + d) * (a - b)
    
    return auc
    # your code here /\


# =============================== 7 NMS ========================================
def nms(detections_dictionary, iou_thr = 0.5):
    """
    :param detections_dictionary: dict of bboxes in format {filename: detections}
        detections is a N x 5 array, where N is number of detections. Each
        detection is described using 5 numbers: [row, col, n_rows, n_cols,
        confidence].
    :param iou_thr: IoU threshold for nearby detections
    :return: dict in same format as detections_dictionary where close detections
        are deleted
    """
    # your code here \/ 
    for filename in detections_dictionary:
        detections = detections_dictionary[filename]
        detections.sort(key=lambda x: x[4], reverse=True)
        del_id = []
 
        for i, first in enumerate(detections):
            if(i in del_id):
                continue
            first_bbox = first[0:4]
            for j, second in enumerate(detections):
                if((j <= i) or (j in del_id)):
                    continue
 
                second_bbox = second[0:4]
                if (calc_iou(first_bbox, second_bbox) > iou_thr):
                    del_id.append(j)
 
        new_detections = []
        for i, x in enumerate(detections):
            if(i not in del_id):
                new_detections.append(x)

        detections_dictionary[filename] = new_detections
 
    return detections_dictionary

    # your code here /\
