import random
import numpy as np
import paddle
import paddle.nn as nn
import cv2
import os
import math

def single2tensor3(img):
    return paddle.to_tensor(np.ascontiguousarray(img)).transpose([2, 0, 1])

def cubic(x):
    absx = paddle.abs(x)
    absx2 = absx**2
    absx3 = absx**3
    type = absx.dtype
    return (1.5*absx3 - 2.5*absx2 + 1) * (paddle.cast((absx <= 1), dtype=type)) + \
        (-0.5*absx3 + 2.5*absx2 - 4*absx + 2) * (paddle.cast(((absx > 1)*(absx <= 2)),dtype=type))

def calculate_weights_indices(in_length, out_length, scale, kernel, kernel_width, antialiasing):
    if (scale < 1) and (antialiasing):
        # Use a modified kernel to simultaneously interpolate and antialias- larger kernel width
        kernel_width = kernel_width / scale

    # Output-space coordinates
    x = paddle.linspace(1, out_length, out_length)

    # Input-space coordinates. Calculate the inverse mapping such that 0.5
    # in output space maps to 0.5 in input space, and 0.5+scale in output
    # space maps to 1.5 in input space.
    u = x / scale + 0.5 * (1 - 1 / scale)

    # What is the left-most pixel that can be involved in the computation?
    left = paddle.floor(u - kernel_width / 2)

    # What is the maximum number of pixels that can be involved in the
    # computation?  Note: it's OK to use an extra pixel here; if the
    # corresponding weights are all zero, it will be eliminated at the end
    # of this function.
    P = math.ceil(kernel_width) + 2

    # The indices of the input pixels involved in computing the k-th output
    # pixel are in row k of the indices matrix.
    indices = left.reshape([out_length, 1]).expand(out_length, P) + paddle.linspace(0, P - 1, P).reshape([
        1, P]).expand(out_length, P)

    # The weights used to compute the k-th output pixel are in row k of the
    # weights matrix.
    distance_to_center = u.paddle([out_length, 1]).expand(out_length, P) - indices
    # apply cubic kernel
    if (scale < 1) and (antialiasing):
        weights = scale * cubic(distance_to_center * scale)
    else:
        weights = cubic(distance_to_center)
    # Normalize the weights matrix so that each row sums to 1.
    weights_sum = paddle.sum(weights, 1).view(out_length, 1)
    weights = weights / weights_sum.expand(out_length, P)

    # If a column in weights is all zero, get rid of it. only consider the first and last column.
    weights_zero_tmp = paddle.sum((weights == 0), 0)
    if not math.isclose(weights_zero_tmp[0], 0, rel_tol=1e-6):
        indices = indices.narrow(1, 1, P - 2)
        weights = weights.narrow(1, 1, P - 2)
    if not math.isclose(weights_zero_tmp[-1], 0, rel_tol=1e-6):
        indices = indices.narrow(1, 0, P - 2)
        weights = weights.narrow(1, 0, P - 2)
    weights = weights
    indices = indices
    sym_len_s = -indices.min() + 1
    sym_len_e = indices.max() - in_length
    indices = indices + sym_len_s - 1
    return weights, indices, int(sym_len_s), int(sym_len_e)


def imresize_np(img, scale, antialiasing=True):
    # Now the scale should be the same for H and W
    # input: img: Numpy, HWC or HW [0,1]
    # output: HWC or HW [0,1] w/o round
    img = paddle.to_tensor(img)
    need_squeeze = True if img.dim() == 2 else False
    if need_squeeze:
        img.unsqueeze_(2)

    in_H, in_W, in_C = img.size()
    out_C, out_H, out_W = in_C, math.ceil(in_H * scale), math.ceil(in_W * scale)
    kernel_width = 4
    kernel = 'cubic'

    # Return the desired dimension order for performing the resize.  The
    # strategy is to perform the resize first along the dimension with the
    # smallest scale factor.
    # Now we do not support this.

    # get weights and indices
    weights_H, indices_H, sym_len_Hs, sym_len_He = calculate_weights_indices(
        in_H, out_H, scale, kernel, kernel_width, antialiasing)
    weights_W, indices_W, sym_len_Ws, sym_len_We = calculate_weights_indices(
        in_W, out_W, scale, kernel, kernel_width, antialiasing)
    # process H dimension
    # symmetric copying
    img_aug = paddle.zeros([in_H + sym_len_Hs + sym_len_He, in_W, in_C],dtype='float32')
    img_aug.narrow(0, sym_len_Hs, in_H).copy_(img)

    sym_patch = img[:sym_len_Hs, :, :]
    inv_idx = paddle.cast(paddle.arange(sym_patch.shape[0] - 1, -1, -1),dtype=paddle.int64)
    sym_patch_inv = sym_patch.index_select(0, inv_idx)
    img_aug.narrow(0, 0, sym_len_Hs).copy_(sym_patch_inv)

    sym_patch = img[-sym_len_He:, :, :]
    inv_idx = paddle.cast(paddle.arange(sym_patch.shape[0] - 1, -1, -1), dtype=paddle.int64)
    sym_patch_inv = sym_patch.index_select(0, inv_idx)
    img_aug.narrow(0, sym_len_Hs + in_H, sym_len_He).copy_(sym_patch_inv)
    out_1 = paddle.zeros([out_H, in_W, in_C], dtype='float32')

    kernel_width = weights_H.size(1)
    for i in range(out_H):
        idx = int(indices_H[i][0])
        for j in range(out_C):
            out_1[i, :, j] = img_aug[idx:idx + kernel_width, :, j].transpose(0, 1).mv(weights_H[i])

    # process W dimension
    # symmetric copying
    out_1_aug = paddle.zeros(out_H, in_W + sym_len_Ws + sym_len_We, in_C)
    out_1_aug.narrow(1, sym_len_Ws, in_W).copy_(out_1)

    sym_patch = out_1[:, :sym_len_Ws, :]
    inv_idx = paddle.arange(sym_patch.size(1) - 1, -1, -1).long()
    sym_patch_inv = sym_patch.index_select(1, inv_idx)
    out_1_aug.narrow(1, 0, sym_len_Ws).copy_(sym_patch_inv)

    sym_patch = out_1[:, -sym_len_We:, :]
    inv_idx = paddle.arange(sym_patch.size(1) - 1, -1, -1).long()
    sym_patch_inv = sym_patch.index_select(1, inv_idx)
    out_1_aug.narrow(1, sym_len_Ws + in_W, sym_len_We).copy_(sym_patch_inv)

    out_2 = paddle.zeros(out_H, out_W, in_C)
    kernel_width = weights_W.size(1)
    for i in range(out_W):
        idx = int(indices_W[i][0])
        for j in range(out_C):
            out_2[:, i, j] = out_1_aug[:, idx:idx + kernel_width, j].mv(weights_W[i])
    if need_squeeze:
        out_2.squeeze_()

    return out_2.numpy()

def _get_paths_from_images(path):
    assert os.path.isdir(path), '{:s} is not a valid directory'.format(path)
    images = []
    for dirpath, _, fnames in sorted(os.walk(path)):
        for fname in sorted(fnames):
            # print(fname, is_image_file(fname))
            if is_image_file(fname) and "checkpoint"  not in fname:
                img_path = os.path.join(dirpath, fname)
                images.append(img_path)
    assert images, '{:s} has no valid image file'.format(path)
    return images

IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.tif']


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def get_image_paths(dataroot):
    paths = None  # return None if dataroot is None
    if isinstance(dataroot, str):
        paths = sorted(_get_paths_from_images(dataroot))
    elif isinstance(dataroot, list):
        paths = []
        for i in dataroot:
            paths += sorted(_get_paths_from_images(i))
    return paths

def imread_uint(path, n_channels=3):
    #  input: path
    # output: HxWx3(RGB or GGG), or HxWx1 (G)
    if n_channels == 1:
        img = cv2.imread(path, 0)  # cv2.IMREAD_GRAYSCALE
        img = np.expand_dims(img, axis=2)  # HxWx1
    elif n_channels == 3:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)  # BGR or G
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)  # GGG
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # RGB
    return img


def uint2single(img):
    return np.float32(img/255.)


def modcrop(img_in, scale):
    # img_in: Numpy, HWC or HW
    img = np.copy(img_in)
    if img.ndim == 2:
        H, W = img.shape
        H_r, W_r = H % scale, W % scale
        img = img[:H - H_r, :W - W_r]
    elif img.ndim == 3:
        H, W, C = img.shape
        H_r, W_r = H % scale, W % scale
        img = img[:H - H_r, :W - W_r, :]
    else:
        raise ValueError('Wrong img ndim: [{:d}].'.format(img.ndim))
    return img

def augment_img(img, mode=0):
    '''Kai Zhang (github: https://github.com/cszn)
    '''
    if mode == 0:
        return img
    elif mode == 1:
        return np.flipud(np.rot90(img))
    elif mode == 2:
        return np.flipud(img)
    elif mode == 3:
        return np.rot90(img, k=3)
    elif mode == 4:
        return np.flipud(np.rot90(img, k=2))
    elif mode == 5:
        return np.rot90(img)
    elif mode == 6:
        return np.rot90(img, k=2)
    elif mode == 7:
        return np.flipud(np.rot90(img, k=3))

class DatasetSR(paddle.io.Dataset):

    def __init__(self, phrase, data_root_L, data_root_H,data_root_Hx4, split_ratio):
        super(DatasetSR, self).__init__()
        # self.opt = opt
        self.n_channels = 3
        self.sf = 2
        self.patch_size = 128
        self.L_size = self.patch_size // self.sf
        self.split_ratio = split_ratio

        self.paths_Hx4 = get_image_paths(data_root_Hx4)
        self.paths_H = get_image_paths(data_root_H)
        self.paths_L = get_image_paths(data_root_L)
        self.phrase ="train"

        assert self.paths_H, 'Error: H path is empty.'
        if self.paths_L and self.paths_H:
            assert len(self.paths_L) == len(self.paths_H) and len(self.paths_L)==len(self.paths_Hx4), 'L/H mismatch - {}, {}.'.format(len(self.paths_L), len(self.paths_H))
        len_path = len(self.paths_L)
        if phrase=='train':
               
                self.paths_L = self.paths_L[:int(len_path*self.split_ratio)]                
                self.paths_H = self.paths_H[:int(len_path*self.split_ratio)]
                self.paths_Hx4 = self.paths_Hx4[: int(len_path*self.split_ratio)]
        else:
                self.paths_L = self.paths_L[int(len_path*self.split_ratio):]
                self.paths_H = self.paths_H[int(len_path*self.split_ratio):]    
                self.paths_Hx4 = self.paths_Hx4[int(len_path*self.split_ratio):]      

    def __getitem__(self, index):

        L_path = None
        H_path = self.paths_H[index]
        img_H = imread_uint(H_path, self.n_channels)
        img_H = uint2single(img_H)

        Hx4_path = self.paths_Hx4[index]
        img_Hx4 = imread_uint(Hx4_path, self.n_channels)
        img_Hx4 = uint2single(img_Hx4)

        if self.paths_L:
            L_path = self.paths_L[index]
            img_L = imread_uint(L_path, self.n_channels)
            img_L = uint2single(img_L)
        else:
            H, W = img_H.shape[:2]

        if  self.phrase == 'train':

            H, W, C = img_L.shape
            mode = random.randint(0, 7)
            img_L, img_H, img_Hx4= augment_img(img_L, mode=mode), augment_img(img_H, mode=mode), augment_img(img_Hx4, mode=mode)

        img_H, img_Hx4, img_L = single2tensor3(img_H),single2tensor3(img_Hx4), single2tensor3(img_L)
    
        if L_path is None:
            L_path = H_path

        return {'L': img_L, 'H': img_H, 'Hx4': img_Hx4, 'L_path': L_path, 'H_path': H_path, 'Hx4_path': Hx4_path}

    def __len__(self):
        return len(self.paths_H)


class DatasetSR_Step3(paddle.io.Dataset):

    def __init__(self, phrase, data_root_L_list, data_root_H_list,data_root_Hx4_list, split_ratio):
        super(DatasetSR_Step3, self).__init__()
        # self.opt = opt
        self.n_channels = 3
        self.sf = 2
        self.patch_size = 128
        self.L_size = self.patch_size // self.sf
        self.split_ratio = split_ratio
        self.paths_L , self.paths_H , self.paths_Hx4 = [], [], []
        # idx = 0
        for data_root_L, data_root_H, data_root_Hx4 in zip(data_root_L_list, data_root_H_list, data_root_Hx4_list):
                self.paths_Hx4+=get_image_paths(data_root_Hx4)
                self.paths_H+=get_image_paths(data_root_H)
                self.paths_L+=get_image_paths(data_root_L)
                # idx+=1
                # print(f" Step {idx}: {len(get_image_paths(data_root_Hx4))}")

        self.phrase ="train"

        assert self.paths_H, 'Error: H path is empty.'
        if self.paths_L and self.paths_H:
            assert len(self.paths_L) == len(self.paths_H) and len(self.paths_L)==len(self.paths_Hx4), 'L/H mismatch - {}, {}.'.format(len(self.paths_L), len(self.paths_H))
        len_path = len(self.paths_L)
        if phrase=='train':
               
                self.paths_L = self.paths_L[:int(len_path*self.split_ratio)]                
                self.paths_H = self.paths_H[:int(len_path*self.split_ratio)]
                self.paths_Hx4 = self.paths_Hx4[: int(len_path*self.split_ratio)]
        else:
                self.paths_L = self.paths_L[int(len_path*self.split_ratio):]
                self.paths_H = self.paths_H[int(len_path*self.split_ratio):]    
                self.paths_Hx4 = self.paths_Hx4[int(len_path*self.split_ratio):]      

    def __getitem__(self, index):

        L_path = None
        # ------------------------------------
        # get H image
        # ------------------------------------
        H_path = self.paths_H[index]
        img_H = imread_uint(H_path, self.n_channels)
        img_H = uint2single(img_H)

        Hx4_path = self.paths_Hx4[index]
        img_Hx4 = imread_uint(Hx4_path, self.n_channels)
        img_Hx4 = uint2single(img_Hx4)

        # ------------------------------------
        if self.paths_L:
            L_path = self.paths_L[index]
            img_L = imread_uint(L_path, self.n_channels)
            img_L = uint2single(img_L)
        else:
            H, W = img_H.shape[:2]

        if  self.phrase == 'train':
            H, W, C = img_L.shape
            mode = random.randint(0, 7)
            img_L, img_H, img_Hx4= augment_img(img_L, mode=mode), augment_img(img_H, mode=mode), augment_img(img_Hx4, mode=mode)

        img_H, img_Hx4, img_L = single2tensor3(img_H),single2tensor3(img_Hx4), single2tensor3(img_L)
    
        if L_path is None:
            L_path = H_path

        return {'L': img_L, 'H': img_H, 'Hx4': img_Hx4, 'L_path': L_path, 'H_path': H_path, 'Hx4_path': Hx4_path}

    def __len__(self):
        return len(self.paths_H)
