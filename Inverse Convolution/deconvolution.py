import numpy as np
from scipy.fft import fft2, ifft2

def gaussian_kernel(size, sigma):
    """
    Построение ядра фильтра Гаусса.

    @param  size  int    размер фильтра (нечетный)
    @param  sigma float  параметр размытия
    @return numpy array  фильтр Гаусса размером size x size
    """
    g = np.zeros((size, size))
    x_0 = size // 2
    y_0 = size // 2
    for x in range(0, size):
        for y in range(0, size): 
            r = np.sqrt((x - x_0) ** 2 + (y - y_0) ** 2)
            h = np.exp(-(r ** 2) / (2 * sigma ** 2)) / (2 * np.pi * (sigma ** 2))
            g[x][y] = h

    # сумма всех элементов в итоговой матрице должна быть равна 1, поэтому каждый элемент делим на сумму всех элементов

    sum = np.sum(g)
    for x in range(0, size):
        for y in range(0, size):
            g[x][y] = g[x][y] / sum
    return g


def fourier_transform(h, shape):
    """
    Получение Фурье-образа искажающей функции

    @param  h            numpy array  искажающая функция h (ядро свертки)
    @param  shape        list         требуемый размер образа
    @return numpy array  H            Фурье-образ искажающей функции h
    """
    if(h.shape == shape):
        H = fft2(h)
        return H
        
    else:
        new_arr = np.zeros(shape)
        new_arr[:h.shape[0], :h.shape[1]] = h
        H = fft2(new_arr)
        return H
        


def inverse_kernel(H, threshold=1e-10):
    """
    Получение H_inv

    @param  H            numpy array    Фурье-образ искажающей функции h
    @param  threshold    float          порог отсечения для избежания деления на 0
    @return numpy array  H_inv
    """
    H_inv = np.zeros(H.shape).astype('csingle')
    for x in range(0, H.shape[0]):
        for y in range(0, H.shape[1]):
            if(np.abs(H[x][y]) <= threshold):
                H_inv[x][y] = 0
            else:
                H_inv[x][y] = 1 / H[x][y]
    return(H_inv)


def inverse_filtering(blurred_img, h, threshold=1e-10):
    """
    Метод инверсной фильтрации

    @param  blurred_img    numpy array  искаженное изображение
    @param  h              numpy array  искажающая функция
    @param  threshold      float        параметр получения H_inv
    @return numpy array                 восстановленное изображение
    """
    G = fft2(blurred_img)
    H = fourier_transform(h, blurred_img.shape)
    H_inv = inverse_kernel(H, threshold)
    F = G * H_inv
    f = ifft2(F) # обратное преобразование Фурье
    return(np.abs(f))

def wiener_filtering(blurred_img, h, K = 0.00006):
    """
    Винеровская фильтрация

    @param  blurred_img    numpy array  искаженное изображение
    @param  h              numpy array  искажающая функция
    @param  K              float        константа из выражения (8)
    @return numpy array                 восстановленное изображение
    """
    H = fourier_transform(h, blurred_img.shape)
    H_conj = np.conj(H)
    G = fft2(blurred_img)
    F = (H_conj * G)/ ((np.abs(H)) ** 2 + K)
    f = ifft2(F) # обратное преобразование Фурье
    return(np.abs(f))


def compute_psnr(img1, img2):
    """
    PSNR metric

    @param  img1    numpy array   оригинальное изображение
    @param  img2    numpy array   искаженное изображение
    @return float   PSNR(img1, img2)
    """
    MSE = np.mean((img1 - img2) ** 2)
    MAX_1 = 255
    PSNR = 20 * np.log10(MAX_1/ np.sqrt(MSE))
    return PSNR
