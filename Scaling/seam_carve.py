import numpy as np

def conversion_from_RGB(img): # яркость изображения
    
    k = [0.299, 0.587, 0.114]
    res = 0
    for i in range(0, 3):
        res += k[i] * img[:, :, i]
    return(res.astype("float64"))

def energy_img(img):
    
    Y = conversion_from_RGB(img)
    Y_row = Y.shape[0]
    Y_col = Y.shape[1]
    I_y, I_x = np.gradient(Y)
    # Возведём все элементы в квадрат кроме элементов с краю 
    I_y[1:Y_row - 1, :] *= 2 # производная для игреков считается в строчке как разность нижней и верхней строки
    I_x[:, 1:Y_col - 1] *= 2 # производная для иксов считается в колонке как разность двух соседних колонок 
    return(np.sqrt(I_x ** 2 + I_y ** 2).astype("float64"))

def new_energy_img(energy, mask):

    energy = energy + mask * (energy.shape[0] * energy.shape[1] * 256.0)
    print(energy.shape)
    return(energy)

def find_seam(img, mask):
    # a)

    arr = np.full((img.shape[0], img.shape[1]), np.inf, dtype='float64')
    energy = new_energy_img(energy_img(img), mask)
    arr[0, :] = energy[0, :]

    # b)
    shifts = np.zeros((img.shape[0], img.shape[1]), dtype = "int8")
    for y in range(1, energy.shape[0]):
        for x in range(0, energy.shape[1]):
            if(x - 1 < 0):
                shift = np.argmin(arr[y - 1, max(0, x - 1) : min(energy.shape[1], x + 2)])
            else:
                shift = np.argmin(arr[y - 1, max(0, x - 1) : min(energy.shape[1], x + 2)]) - 1
            arr[y][x] = energy[y][x] + arr[y - 1][x + shift]
            shifts[y, x] = shift
    # c)

    seam_mask = np.zeros((img.shape[0], img.shape[1]), dtype = "float64")
    # Поиск минимального левого элемента
    min_energy = arr[energy.shape[0] - 1, 0]
    min_index = 0
    for x in range(1, energy.shape[1]):
        if(min_energy > arr[energy.shape[0] - 1, x]):
            min_energy = arr[energy.shape[0] - 1, x]
            min_index = x   

    # заполняем массив единицами, если пиксель принадлежит шву
    t = energy.shape[0] - 1
    p = min_index
    y = energy.shape[0] - 1
    x = min_index
    seam_mask[y, x] = 1
    while(t > 0):
        t = t - 1
        p = p + shifts[y, x]
        y = t
        x = p
        seam_mask[y, x] = 1

    return seam_mask

def horizontal_shrink(img, mask):
    seam_mask = find_seam(img, mask)
    new_mask = np.zeros((img.shape[0], img.shape[1] - 1), dtype = "float64")
    final_img = np.zeros((img.shape[0], img.shape[1] - 1, img.shape[2]), dtype = "float64")
    for i in range(0, img.shape[0]):
        for j in range(0, img.shape[1]):
            if(seam_mask[i][j] == 1): # если пиксель принадлежит шву,то удаляем его
                final_img[i, :j] = img[i, :j]
                final_img[i, j:] = img[i, j+1:]
                new_mask[i, :j] = mask[i, :j]
                new_mask[i, j:] = mask[i, j+1:]
            break
    return(final_img, new_mask, seam_mask)

def vertical_shrink(img, mask):
    trans_img = (np.transpose(img[ :, :, 0]), np.transpose(img[ :, :, 1]), np.transpose(img[ :, :, 2]))
    final_img, new_mask, seam_mask = horizontal_shrink(np.dstack(trans_img), np.transpose(mask))
    trans_final_img = (np.transpose(final_img[ :, :, 0]), np.transpose(final_img[ :, :, 1]), np.transpose(final_img[ :, :, 2]))
    return(np.dstack(trans_final_img), np.transpose(new_mask), np.transpose(seam_mask))
    
def horizontal_expand(img, mask):
    seam_mask = find_seam(img, mask)
    new_mask = np.zeros((img.shape[0], img.shape[1] - 1), dtype = "float64")
    final_img = np.zeros((img.shape[0], img.shape[1] - 1, img.shape[2]), dtype = "float64")
    for i in range(0, img.shape[0]):
        for j in range(0, img.shape[1]):
            if(seam_mask[i][j] == 1): # если пиксель принадлежит шву,то удаляем его
                final_img[i, :j+1] = img[i, :j+1]
                final_img[i, j+2:] = img[i, j+3:]
                if(j + 1 > img.shape[1]): # то есть вышли за границу
                    final_img[i][j + 1] = (img[i][j] + img[i][j]) // 2
                else:
                    final_img[i][j + 1] = (img[i][j] + img[i][j + 1]) // 2
                new_mask[i, :j+1] = mask[i, :j+1]
                new_mask[i, j+2:] = mask[i, j+3:]
                new_mask[i][j+1] = mask[i][j]
            break
    return(final_img, new_mask, seam_mask)

def vertical_expand(img, mask):
    trans_img = (np.transpose(img[ :, :, 0]), np.transpose(img[ :, :, 1]), np.transpose(img[ :, :, 2]))
    final_img, new_mask, seam_mask = horizontal_expand(np.dstack(trans_img), np.transpose(mask))
    trans_final_img = (np.transpose(final_img[ :, :, 0]), np.transpose(final_img[ :, :, 1]), np.transpose(final_img[ :, :, 2]))
    return(np.dstack(trans_final_img), np.transpose(new_mask), np.transpose(seam_mask))

def seam_carve(img, mode, mask):
    if(mask is None):
        mask = np.zeros((img.shape[0], img.shape[1]), dtype='int8')
    orientation, direction = mode.split(' ')
    if(orientation == 'horizontal'):
        if(direction == 'shrink'):
            return(horizontal_shrink(img, mask))
        else:
            return(horizontal_expand(img, mask))
    elif(orientation == 'vertical'):
        if(direction == 'shrink'):
            return(vertical_shrink(img, mask))
        else:
            return(vertical_expand(img, mask))