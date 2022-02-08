import numpy as np
from scipy import signal
#1

def get_bayer_masks(n_rows, n_cols):
    # задали координаты цветов в ячейке
    r = np.array([[False , True], [False, False]])
    g = np.array([[True, False], [False, True]])
    b = np.array([[False, False], [True, False]])

    new_r = np.tile(r, (n_rows // 2 + 1, n_cols // 2 + 1))
    new_g = np.tile(g, (n_rows // 2 + 1, n_cols // 2 + 1))
    new_b = np.tile(b, (n_rows // 2 + 1, n_cols // 2 + 1))

    red = new_r[ :n_rows, :n_cols]
    green = new_g[ :n_rows, :n_cols]
    blue = new_b[ :n_rows, :n_cols]
    
    rgb_mask = np.stack([red, green, blue], axis = 2)

    return rgb_mask
# 2

def get_colored_img(raw_img):
    # передаем размеры картинки в функцию для получения маски
    n_rows = raw_img.shape[0]
    n_cols = raw_img.shape[1]
    mask = get_bayer_masks(n_rows, n_cols)

    red_img = mask[:, :, 0] * raw_img
    green_img = mask[:, :, 1] * raw_img
    blue_img = mask[:, :, 2] * raw_img

    res = np.stack([red_img, green_img, blue_img], axis = 2)

    return res

# 3

def bilinear_interpolation(colored_img):
    len1, len2  = colored_img.shape[:2]
    P = colored_img.astype('int32')
    res = np.copy(P)
    mask = get_bayer_masks(len1, len2)
    for x in np.arange(1, len1 - 1):
        for y in np.arange(1, len2 - 1):
            if (mask[x, y, 0] == True): # случай R в центре
                res[x, y, 1] = (P[x - 1, y, 1] + P[x, y + 1, 1] + P[x + 1, y, 1] + P[x, y - 1, 1]) // 4
                res[x, y, 2] = (P[x - 1, y - 1, 2] + P[x - 1, y + 1, 2] + P[x + 1, y + 1, 2] + P[x + 1, y - 1, 2]) // 4

            if(mask[x, y, 1] == True): # случай G в центре
                if(x % 2 != 0):
                    # 1 случай
                    res[x, y, 0] = (P[x - 1, y, 0] + P[x + 1, y, 0]) // 2
                    res[x, y, 2] = (P[x, y - 1, 2] + P[x, y + 1, 2]) // 2
                else:
                    # 2 случай
                    res[x, y, 2] = (P[x - 1, y, 2] + P[x + 1, y, 2]) // 2
                    res[x, y, 0] = (P[x, y - 1, 0] + P[x, y + 1, 0]) // 2

            if(mask[x, y, 2] == True): # случай B в центре
                res[x, y, 0] = (P[x - 1, y - 1, 0] + P[x - 1, y + 1, 0] + P[x + 1, y + 1, 0] + P[x + 1, y - 1, 0]) // 4
                res[x, y, 1] = (P[x - 1, y, 1] + P[x, y + 1, 1] + P[x + 1, y, 1] + P[x, y - 1, 1]) // 4

    return(res.astype('uint8'))

#4

G_at_R = np.array([
        [0, 0, -1, 0, 0],
        [0, 0, 2, 0, 0],
        [-1, 2, 4, 2, -1],
        [0, 0, 2, 0, 0],
        [0, 0, -1, 0, 0]
        ])

R_at_G_1 = np.array([
        [0, 0, 0.5, 0, 0],
        [0, -1, 0, -1, 0],
        [-1, 4, 5, 4, -1],
        [0, -1, 0, -1, 0],
        [0, 0, 0.5, 0, 0]
        ])
 
R_at_G_2 = np.array([
        [0, 0, -1, 0, 0],
        [0, -1, 4, -1, 0],
        [0.5, 0, 5, 0, 0.5],
        [0, -1, 4, -1, 0],
        [0, 0, -1, 0, 0]
        ])
 
R_at_B = np.array([
        [0, 0, -1.5, 0, 0],
        [0, 2, 0, 2, 0],
        [-1.5, 0, 6, 0, -1.5],
        [0, 2, 0, 2, 0],
        [0, 0, -1.5, 0, 0]
        ])
        
def improved_interpolation(raw_img):
    H = raw_img.shape[0]
    W = raw_img.shape[1]
    mask = get_bayer_masks(H, W)
    image = get_colored_img(raw_img.astype("float64"))
    bilinear = bilinear_interpolation(image)
 
    g_b = bilinear[:, :, 1] * mask[:, :, 1]
    alpha =  signal.convolve(raw_img, G_at_R, "same") / np.sum(G_at_R) # G at R/B locations
    green = g_b + alpha * (~mask[:, :, 1])
 
    channel_1 = np.tile(np.array([[1, 0], [0, 0]]), (H // 2 + 1, W // 2 + 1))[ :H, :W]
    channel_2 = np.tile(np.array([[0, 0], [0, 1]]), (H // 2 + 1, W // 2 + 1))[ :H, :W]

    alpha = signal.convolve(raw_img, R_at_G_1, "same") / np.sum(R_at_G_1) # R at green in R row, B column ; B at green in B row, R column
    betta = signal.convolve(raw_img, R_at_G_2, "same") / np.sum(R_at_G_2) # R at green in B row, R column ; B at green in R row, B column
    gamma = signal.convolve(raw_img, R_at_B, "same") / np.sum(R_at_B) # R at blue in B row, B column ; B at red in R row, B column
 
    r_b = bilinear[:, :, 0] * mask[:, :, 0]
    red = r_b + (alpha * channel_1 + betta * channel_2 + gamma * mask[:, :, 2])
 
    r_b = bilinear[:, :, 2] * mask[:, :, 2]
    blue = r_b + (alpha * channel_2 + betta * channel_1 + gamma * mask[:, :, 0])
 
    return(np.clip(np.stack((red, green, blue), axis=2), 0, 255).astype('uint8'))
    
#5
def compute_psnr(img_pred, img_gt):
    H = img_pred.shape[0]
    W = img_pred.shape[1]
    C = 3
    I_pred = img_pred.astype("float64") 
    I_gt = img_gt.astype("float64")
    MSE = np.sum((I_pred - I_gt) ** 2) / (H * W * C)
    if(MSE == 0):
        raise ValueError
    PSNR = 10 * np.log10((np.max(I_gt) ** 2) / MSE)
    return PSNR