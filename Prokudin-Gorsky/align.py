import numpy as np
# Слишком долго :(
# def cross_correlation(I1, I2):
#     return abs(np.sum(I1 * I2) / (np.sum(I1) * np.sum(I2)))

def MSE(I1, I2):
    return np.mean((I1 - I2) ** 2)

def shift(channel1, channel2, X, Y):
    best_metrik = 1000
    coord = [0, 0]
    for x in range(X[0], X[1]):
        for y in range(Y[0], Y[1]):
            move_channel1 = np.roll(channel1, (x, y), axis = (0, 1))
            metrik = MSE(channel2, move_channel1)
            if(metrik < best_metrik):
                best_metrik = metrik
                coord = (x, y)
    return coord


def pyramid(channel1, channel2):
    if((channel1.shape[0] < 500) or (channel1.shape[1] < 500)):
        return shift(channel1, channel2, (-15, 15), (-15, 15))

    # Для последовательного уменьшения изображения в 2 раза будем каждый раз брать только четные элементы, расположенные по порядку
    resize_channel1 = channel1[1::2, 1::2]
    resize_channel2 = channel2[1::2, 1::2]
    new_coord = pyramid(resize_channel1, resize_channel2)
    return shift(channel1, channel2, (new_coord[0]* 2 - 1, new_coord[0] * 2 + 2), (new_coord[1] * 2 - 1, new_coord[1] * 2 + 2))


def align(img, green_coord):
    #1
    img = img.astype("float64")
    row_g, col_g = green_coord
    H, W = img.shape
    
    # Разделение на 3 канала

    H = H - H % 3 # Cделаем высоту изображения кратной 3
    height = H // 3
    width = W
    B = img[ : height]
    G = img[height : 2 * height]
    R = img[2 * height : 3 * height]
    
    #2
    # Обрежем рамки изображений по 5% с каждой стороны
    k = 0.1

    crop_side_hight = int(height * k)
    crop_side_width = int(width * k)

    # Обновляем размеры изображений
    new_b = B[crop_side_hight:height - crop_side_hight, crop_side_width:width - crop_side_width]
    new_g = G[crop_side_hight:height - crop_side_hight, crop_side_width:width - crop_side_width]
    new_r = R[crop_side_hight:height - crop_side_hight, crop_side_width:width - crop_side_width]

    #3
    # Cовмещаем каналы 1)красный с зелёным, 2)синий с зелёным
    
    #1)
    x, y = pyramid(new_r, new_g)
    shift_r = np.roll(R, (x, y), axis = (0, 1))
    row_r = row_g - x + H // 3
    col_r = col_g - y

    #2)
    x, y = pyramid(new_b, new_g)
    shift_b = np.roll(B, (x, y), axis = (0, 1))
    row_b = row_g - x - H // 3
    col_b = col_g - y
               
    
    return(np.stack((shift_r, G, shift_b), axis = -1).astype("uint8"), (row_b, col_b), (row_r, col_r))