from ctypes import c_bool
import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from scipy.ndimage.filters import gaussian_filter
from skimage.metrics import peak_signal_noise_ratio
# !Этих импортов достаточно для решения данного задания, нельзя использовать другие библиотеки!


def pca_compression(matrix, p):
    """ Сжатие изображения с помощью PCA
    Вход: двумерная матрица (одна цветовая компонента картинки), количество компонент
    Выход: собственные векторы и проекция матрицы на новое пр-во
    """
    
    # Your code here
    
    #1 Отцентруем каждую строчку матрицы
    # M_str = np.zeros(matrix.shape[0], dtype = "float64")
    M_mean = np.mean(matrix, axis=1)
    matrix = matrix - M_mean

    #2 Найдем матрицу ковариации
    covmat = np.cov(matrix)
    
    #3 Ищем собственные значения и собственные векторы матрицы ковариации, используйте linalg.eigh из numpy
    eig_val, eig_vec = np.linalg.eigh(covmat)

    #4 Посчитаем количество найденных собственных векторов
    count_eig_vec = eig_vec.shape[1]

    #5 Сортируем собственные значения в порядке убывания
    #6 Сортируем собственные векторы согласно отсортированным собственным значениям
    # !Это все для того, чтобы мы производили проекцию в направлении максимальной дисперсии!

    idx = np.argsort(eig_val)[::-1]   
    eig_val = eig_val[idx]
    eig_vec = eig_vec[:, idx]

    #7 Оставляем только p собственных векторов
    eig_vec = eig_vec[:, :p]

    #8 Проекция данных на новое пространство
    res = np.dot(np.transpose(eig_vec), matrix)
    return(eig_vec, res, M_mean)


def pca_decompression(compressed):
    """ Разжатие изображения
    Вход: список кортежей из собственных векторов и проекций для каждой цветовой компоненты
    Выход: разжатое изображение
    """
    result_img = []
    for i, comp in enumerate(compressed):
        # Матрично умножаем собственные векторы на проекции и прибавляем среднее значение по строкам исходной матрицы
        # !Это следует из описанного в самом начале примера!
        # Your code here
        k = np.dot(comp[0], comp[1]) + comp[2]
        size = k.shape[1]
        result_img.append(k)
    arr = np.zeros((size, size, 3), dtype = "float64")
    arr[..., 0] = result_img[0]
    arr[..., 1] = result_img[1]
    arr[..., 2] = result_img[2]
    #print(arr)
    return(np.clip(arr, 0, 255).astype('uint8'))

def pca_visualize():
    plt.clf()
    img = imread('cat.jpg')
    if len(img.shape) == 3:
        img = img[..., :3]
    fig, axes = plt.subplots(3, 3)
    fig.set_figwidth(12)
    fig.set_figheight(12)

    for i, p in enumerate([1, 5, 10, 20, 50, 100, 150, 200, 256]):
        compressed = []
        for j in range(0, 3):
            # Your code here
            compressed.append(pca_compression(img[..., j], p))
        axes[i // 3, i % 3].imshow(pca_decompression(compressed))
        axes[i // 3, i % 3].set_title('Компонент: {}'.format(p))

    fig.savefig("pca_visualization.png")

def rgb2ycbcr(img):
    """ Переход из пр-ва RGB в пр-во YCbCr
    Вход: RGB изображение
    Выход: YCbCr изображение
    """
    
    # Your code here
    R = img[..., 0]
    G = img[..., 1]
    B = img[..., 2]
    lst = []
    y  = 0.299 * R + 0.587 * G + 0.114 * B
    lst.append(y)
    cb = -0.1687 * R - 0.3313 * G + 0.5 * B + 128
    lst.append(cb)
    cr = 0.5 * R - 0.4187 * G - 0.0813 * B + 128
    lst.append(cr)
    arr = np.zeros((img.shape[0], img.shape[1], 3), dtype = "float64")
    arr[..., 0] = lst[0]
    arr[..., 1] = lst[1]
    arr[..., 2] = lst[2]
    return arr.astype('uint8')


def ycbcr2rgb(img):
    """ Переход из пр-ва YCbCr в пр-во RGB
    Вход: YCbCr изображение
    Выход: RGB изображение
    """
    
    # Your code here
    arr = np.array([[1, 0, 1.402], 
                    [1, -0.34414, -0.71414], 
                    [1, 1.772, 0]])
    img = img.astype("float64")
    img[:,:,[1,2]] -= 128
    res = img.dot(np.transpose(arr))
    return(np.clip(res, 0, 255).astype('uint8'))

def get_gauss_1():
    plt.clf()
    rgb_img = imread('Lenna.png')
    if len(rgb_img.shape) == 3:
        rgb_img = rgb_img[..., :3]

    # Your code here
    arr = rgb2ycbcr(rgb_img)
    lst = []
    Y = arr[..., 0]
    lst.append(Y)
    C_b = arr[..., 1]
    C_r = arr[..., 2]
    C_b_new = gaussian_filter(C_b, 10)
    lst.append(C_b_new)
    C_r_new = gaussian_filter(C_r, 10)
    lst.append(C_r_new)

    mas = np.zeros((rgb_img.shape[0], rgb_img.shape[1], 3), dtype = "float64")
    mas[..., 0] = lst[0]
    mas[..., 1] = lst[1]
    mas[..., 2] = lst[2]

    res = ycbcr2rgb(mas)
    plt.imshow(res)
    plt.savefig("gauss_1.png")

def get_gauss_2():
    plt.clf()
    rgb_img = imread('Lenna.png')
    if len(rgb_img.shape) == 3:
        rgb_img = rgb_img[..., :3]

    # Your code here
    arr = rgb2ycbcr(rgb_img)
    lst = []
    Y = arr[..., 0]
    Y_new = gaussian_filter(Y, 10)
    lst.append(Y_new)
    C_b = arr[..., 1]
    lst.append(C_b)
    C_r = arr[..., 2]
    lst.append(C_r)
    
    mas = np.zeros((rgb_img.shape[0], rgb_img.shape[1], 3), dtype = "float64")
    mas[..., 0] = lst[0]
    mas[..., 1] = lst[1]
    mas[..., 2] = lst[2]

    res = ycbcr2rgb(mas)
    plt.imshow(res)
    plt.savefig("gauss_2.png")

def downsampling(component):
    """Уменьшаем цветовые компоненты в 2 раза
    Вход: цветовая компонента размера [A, B, 1]
    Выход: цветовая компонента размера [A // 2, B // 2, 1]
    """
    
    # Your code here
    comp_new = gaussian_filter(component, 10)
    res = comp_new[0::2, 0::2]    
    return res


def dct(block):
    """Дискретное косинусное преобразование
    Вход: блок размера 8x8
    Выход: блок размера 8x8 после ДКП
    """

    # Your code here
    G = np.zeros((8, 8), dtype = "float64")
    for u in range(0, 8):
        if(u == 0):
            alpha_u = 1 / np.sqrt(2)
        else:
            alpha_u = 1
        for v in range(0, 8):
            if(v == 0):
                alpha_v = 1 / np.sqrt(2)
            else:
                alpha_v = 1
            res = 0
            for x in range(0, 8):
                for y in range(0, 8):
                    res = res + block[x][y] * np.cos((2 * x + 1) * u * np.pi / 16) * np.cos((2 * y + 1) * v * np.pi / 16)
            G[u][v] = 0.25 * alpha_u * alpha_v * res
    
    return G


# Матрица квантования яркости
y_quantization_matrix = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99]
])

# Матрица квантования цвета
color_quantization_matrix = np.array([
    [17, 18, 24, 47, 99, 99, 99, 99],
    [18, 21, 26, 66, 99, 99, 99, 99],
    [24, 26, 56, 99, 99, 99, 99, 99],
    [47, 66, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99]
])


def quantization(block, quantization_matrix):
    """Квантование
    Вход: блок размера 8x8 после применения ДКП; матрица квантования
    Выход: блок размера 8x8 после квантования. Округление осуществляем с помощью np.round
    """
    res = np.round(block / quantization_matrix)
    return(np.array(res, dtype = 'float64'))


def own_quantization_matrix(default_quantization_matrix, q):
    """Генерация матрицы квантования по Quality Factor
    Вход: "стандартная" матрица квантования; Quality Factor
    Выход: новая матрица квантования
    Hint: если после проделанных операций какие-то элементы обнулились, то замените их единицами
    """
    assert 1 <= q <= 100
    S = 1
    if q < 50:
        S = 5000 / q
        return np.array((50 + S * default_quantization_matrix) / 100, dtype = 'int')
    elif q <= 99:
        S = 200 - 2 * q
        return np.array((50 + S * default_quantization_matrix) / 100, dtype = 'int')
    else:
        S = 1
        return np.array((50 * 2 + S * default_quantization_matrix) / 100, dtype = 'int')
    

def zigzag(block):
    """Зигзаг-сканирование
    Вход: блок размера 8x8
    Выход: список из элементов входного блока, получаемый после его обхода зигзаг-сканированием
    """
    size = 8
    curX = 0
    curY = 0
    direction = "down"
    positions = []
    positions.append(block[curX, curY])
    while not (curX == size - 1 and curY == size - 1):
        if direction == "down":
            if curY == size - 1:
                curX += 1
            else:
                curY += 1
            positions.append(block[curX, curY])
            # Движемся по диагонали вверх и вправо
            while curX < size - 1 and curY > 0:
                curX += 1
                curY -= 1
                positions.append(block[curX, curY])
            direction = "right"
            continue
        else: #direction == "right"
            if curX == size - 1:
                curY += 1
            else:
                curX += 1
            positions.append(block[curX, curY])
            # Движемся по диагонали вниз и влево
            while curY < size - 1 and curX > 0:
                curX -= 1
                curY += 1
                positions.append(block[curX, curY])
            direction = "down"
            continue
    return positions


def compression(zigzag_list):
    """Сжатие последовательности после зигзаг-сканирования
    Вход: список после зигзаг-сканирования
    Выход: сжатый список в формате, который был приведен в качестве примера в самом начале данного пункта
    """
    lst = []
    count = 1
    i = 0

    while(i <= len(zigzag_list) - 1):

            
        while((i < len(zigzag_list) - 1) and (zigzag_list[i] == zigzag_list[i + 1] == 0.0)):
            count += 1
            i += 1  

        if count > 1:
            lst.append(0)
            lst.append(count)
            count = 1
        elif(zigzag_list[i] == 0.0 and zigzag_list[i] == -0.0 and count == 1):
            lst.append(0)
            lst.append(count)
        
        if(zigzag_list[i] != 0.0 and zigzag_list[i] != -0.0):
            lst.append(zigzag_list[i])
    
        i += 1

    return lst

def jpeg_compression(img, quantization_matrixes):
    """JPEG-сжатие
    Вход: цветная картинка, список из 2-ух матриц квантования
    Выход: список списков со сжатыми векторами: [[compressed_y1,...], [compressed_Cb1,...], [compressed_Cr1,...]]
    """

    # Your code here
    img1 = rgb2ycbcr(img)
    img2 = np.zeros((img1.shape[0] // 2, img1.shape[1] // 2, 2))
    img2[..., 0] = np.array(downsampling(img1[..., 1]))
    img2[..., 1] = np.array(downsampling(img1[..., 2]))

    windowsize_r = 8
    windowsize_c = 8
    res = []
    for r in range(0, img1.shape[0] - (windowsize_r - 1), windowsize_r):
        for c in range(0, img1.shape[1] - (windowsize_c - 1), windowsize_c):
            a = img1[r : r + windowsize_r, c : c + windowsize_c, 0].astype("float")
            a = a - 128
            res.extend(compression(zigzag(quantization(dct(a), quantization_matrixes[0]))))
    
    for r in range(0, img2.shape[0] - (windowsize_r - 1), windowsize_r):
        for c in range(0, img2.shape[1] - (windowsize_c - 1), windowsize_c):
            a = img2[r : r + windowsize_r, c : c + windowsize_c, 0].astype("float")
            a = a - 128
            res.extend(compression(zigzag(quantization(dct(a), quantization_matrixes[1]))))

    for r in range(0, img2.shape[0] - (windowsize_r - 1), windowsize_r):
        for c in range(0, img2.shape[1] - (windowsize_c - 1), windowsize_c):
            a = img2[r : r + windowsize_r, c : c + windowsize_c, 1].astype("float")
            a = a - 128
            res.extend(compression(zigzag(quantization(dct(a), quantization_matrixes[1]))))
    
    return res
    
def inverse_compression(compressed_list):
    """Разжатие последовательности
    Вход: сжатый список
    Выход: разжатый список
    """
    
    # Your code here
    lst = []
    j = 0
    i = 0
    while(i <= len(compressed_list) - 1):
        if(compressed_list[i] != 0.0):
            lst.append(compressed_list[i])

        if(compressed_list[i] == 0.0):
            while(j < compressed_list[i + 1]):
                lst.append(compressed_list[i])
                j += 1
            j = 0
            i += 1
    
        i += 1
    return lst

def inverse_zigzag(input):
    """Обратное зигзаг-сканирование
    Вход: список элементов
    Выход: блок размера 8x8 из элементов входного списка, расставленных в матрице в порядке их следования в зигзаг-сканировании
    """
    block = np.zeros((8, 8), dtype = "float64")
    size = 8
    curX = 0
    curY = 0
    c = 1
    direction = "down"
    block[curX, curY] = input[0]
    while not (curX == size - 1 and curY == size - 1):
        if direction == "down":
            if curY == size - 1:
                curX += 1
            else:
                curY += 1
            block[curX, curY] = input[c]
            c += 1
            # Движемся по диагонали вверх и вправо
            while curX < size - 1 and curY > 0:
                curX += 1
                curY -= 1
                block[curX, curY] = input[c]
                c += 1
            direction = "right"
            continue
        else: #direction == "right"
            if curX == size - 1:
                curY += 1
            else:
                curX += 1
            block[curX, curY] = input[c]
            c += 1
            # Движемся по диагонали вниз и влево
            while curY < size - 1 and curX > 0:
                curX -= 1
                curY += 1
                block[curX, curY] = input[c]
                c += 1
            direction = "down"
            continue
    return block

def inverse_quantization(block, quantization_matrix):
    """Обратное квантование
    Вход: блок размера 8x8 после применения обратного зигзаг-сканирования; матрица квантования
    Выход: блок размера 8x8 после квантования. Округление не производится
    """
    
    # Your code here
    res = block * quantization_matrix
    return(np.array(res, dtype = 'float64'))


def inverse_dct(block):
    """Обратное дискретное косинусное преобразование
    Вход: блок размера 8x8
    Выход: блок размера 8x8 после обратного ДКП. Округление осуществляем с помощью np.round
    """

    # Your code here
    f = np.zeros((8, 8), dtype = "float64")
    for x in range(0, 8):
        for y in range(0, 8):
            res = 0
            for u in range(0, 8):
                for v in range(0, 8):
                    if(u == 0):
                        alpha_u = 1 / np.sqrt(2)
                    else:
                        alpha_u = 1
                    if(v == 0):
                        alpha_v = 1 / np.sqrt(2)
                    else:
                        alpha_v = 1
                    res = res + alpha_u * alpha_v * block[u][v] * np.cos((2 * x + 1) * u * np.pi / 16) * np.cos((2 * y + 1) * v * np.pi / 16)
            f[x][y] = np.round(0.25 * res)
    return f


def upsampling(component):
    """Увеличиваем цветовые компоненты в 2 раза
    Вход: цветовая компонента размера [A, B, 1]
    Выход: цветовая компонента размера [2 * A, 2 * B, 1]
    """
    double = np.zeros((2 * component.shape[0], 2 * component.shape[1]), dtype = 'float64')
    for i in range (0, component.shape[0]):
        for j in range (0, component.shape[1]):
            double[2 * i][2 * j] = component[i][j]
            double[2 * i][2 * j + 1] = component[i][j]
            double[2 * i + 1][2 * j] = component[i][j]
            double[2 * i + 1][2 * j + 1] = component[i][j]
    
    return double

def jpeg_decompression(result, result_shape, quantization_matrixes):
    """Разжатие изображения
    Вход: result список сжатых данных, размер ответа, список из 2-ух матриц квантования
    Выход: разжатое изображение
    """
    result1 = inverse_compression(result)
    img = np.zeros((result_shape[0], result_shape[1], 3))
    img2 = np.zeros((result_shape[0] // 2, result_shape[1] // 2, 2))
    a = 0

    windowsize_r = 8
    windowsize_c = 8

    for r in range(0, img.shape[0] - (windowsize_r - 1), windowsize_r):
        for c in range(0, img.shape[1] - (windowsize_c - 1), windowsize_c):
            img[r : r + windowsize_r, c : c + windowsize_c, 0] = (inverse_dct(inverse_quantization(inverse_zigzag(result1[a:a+64]), quantization_matrixes[0]))).astype("float") + 128
            a += 64
    
    for r in range(0, img2.shape[0] - (windowsize_r - 1), windowsize_r):
        for c in range(0, img2.shape[1] - (windowsize_c - 1), windowsize_c):
            img2[r : r + windowsize_r, c : c + windowsize_c, 0] = (inverse_dct(inverse_quantization(inverse_zigzag(result1[a:a+64]), quantization_matrixes[1]))).astype("float") + 128
            a += 64

    for r in range(0, img2.shape[0] - (windowsize_r - 1), windowsize_r):
        for c in range(0, img2.shape[1] - (windowsize_c - 1), windowsize_c):
            img2[r : r + windowsize_r, c : c + windowsize_c, 1] = (inverse_dct(inverse_quantization(inverse_zigzag(result1[a:a+64]), quantization_matrixes[1]))).astype("float") + 128
            a += 64
            
    img[..., 1] = upsampling(img2[..., 0])
    img[..., 2] = upsampling(img2[..., 1])
    img = ycbcr2rgb(img)
    return img


def jpeg_visualize():
    plt.clf()
    img = imread('Lenna.png')
    if len(img.shape) == 3:
        img = img[..., :3]
    fig, axes = plt.subplots(2, 3)
    fig.set_figwidth(12)
    fig.set_figheight(12)

    for i, p in enumerate([1, 10, 20, 50, 80, 100]):
        # Your code here
        y_matrix = own_quantization_matrix(y_quantization_matrix, p)
        color_matrix = own_quantization_matrix(color_quantization_matrix, p)
        compressed = jpeg_compression(img, [y_matrix, color_matrix])
        axes[i // 3, i % 3].imshow(jpeg_decompression(compressed, img.shape, [y_matrix, color_matrix]))
        axes[i // 3, i % 3].set_title('Quality Factor: {}'.format(p))

    fig.savefig("jpeg_visualization.png")

def compression_pipeline(img, c_type, param=1):
    """Pipeline для PCA и JPEG
    Вход: исходное изображение; название метода - 'pca', 'jpeg'; 
    param - кол-во компонент в случае PCA, и Quality Factor для JPEG
    Выход: изображение; количество бит на пиксель
    """
    
    assert c_type.lower() == 'jpeg' or c_type.lower() == 'pca'
    
    if c_type.lower() == 'jpeg':
        y_quantization = own_quantization_matrix(y_quantization_matrix, param)
        color_quantization = own_quantization_matrix(color_quantization_matrix, param)
        matrixes = [y_quantization, color_quantization]
        
        compressed = jpeg_compression(img, matrixes)
        img = jpeg_decompression(compressed, img.shape, matrixes)
    elif c_type.lower() == 'pca':
        compressed = []
        for j in range(0, 3):
            compressed.append((pca_compression(img[:, :, j].astype(np.float64).copy(), param)))
            
        img = pca_decompression(compressed)
        compressed.extend([np.mean(img[:, :, 0], axis=1), np.mean(img[:, :, 1], axis=1), np.mean(img[:, :, 2], axis=1)])
        
    if 'tmp' not in os.listdir() or not os.path.isdir('tmp'):
        os.mkdir('tmp')
        
    np.savez_compressed(os.path.join('tmp', 'tmp.npz'), compressed)
    size = os.stat(os.path.join('tmp', 'tmp.npz')).st_size * 8
    os.remove(os.path.join('tmp', 'tmp.npz'))
        
    return img, size / (img.shape[0] * img.shape[1])


def calc_metrics(img_path, c_type, param_list):
    """Подсчет PSNR и Rate-Distortion для PCA и JPEG. Построение графиков
    Вход: пусть до изображения; тип сжатия; список параметров: кол-во компонент в случае PCA, и Quality Factor для JPEG
    """
    
    assert c_type.lower() == 'jpeg' or c_type.lower() == 'pca'
    
    img = imread(img_path)
    if len(img.shape) == 3:
        img = img[..., :3]
    
    outputs = []
    for param in param_list:
        outputs.append(compression_pipeline(img.copy(), c_type, param))
     
    psnr = [peak_signal_noise_ratio(img, output[0]) for output in outputs]
    rate = [output[1] for output in outputs]
    
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
    fig.set_figwidth(20)
    fig.set_figheight(5)
    
    ax1.set_title('PSNR for {}'.format(c_type.upper()))
    ax1.plot(param_list, psnr, 'tab:orange')
    ax1.set_xlabel('Quality Factor')
    ax1.set_ylabel('PSNR')
    
    ax2.set_title('Rate-Distortion for {}'.format(c_type.upper()))
    ax2.plot(psnr, rate, 'tab:red')
    ax2.set_xlabel('Distortion')
    ax2.set_ylabel('Rate')
    return fig


def get_pca_metrics_graph():
    plt.clf()
    fig = calc_metrics('Lenna.png', 'pca', [1, 5, 10, 20, 50, 100, 150, 200, 256])
    fig.savefig("pca_metrics_graph.png")


def get_jpeg_metrics_graph():
    plt.clf()
    fig = calc_metrics('Lenna.png', 'jpeg', [1, 10, 20, 50, 80, 100])
    fig.savefig("jpeg_metrics_graph.png")

# pca_visualize()
# get_gauss_1()
# get_gauss_2()
# jpeg_visualize()
# get_pca_metrics_graph()
# get_jpeg_metrics_graph()