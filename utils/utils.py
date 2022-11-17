import paddle
import paddle.vision
import paddle.fluid as fluid
import paddle.nn as nn
import paddle.nn.functional as F
from paddle_msssim import ms_ssim
import os
from PIL import Image
import numpy as np

def L1_Charbonier_loss(predict, gt):
    diff = paddle.add(predict, -gt)
    diff_sq = diff*diff 
    diff_sq_color = paddle.mean(diff_sq, 1, True)
    error = paddle.sqrt(diff_sq_color+1e-3*1e-3)
    loss = paddle.mean(error)
    return loss


def LossStep1(predit,gt):
    lambda2 =0.10
    L1_Char_loss = L1_Charbonier_loss(predit, gt)
    MS_SSIM_loss = 1-ms_ssim(predit,gt,data_range=1)

    total_loss = lambda2*MS_SSIM_loss + L1_Char_loss 
    return total_loss,  L1_Char_loss, MS_SSIM_loss

def LossStep2(predict,gt):

    l1_loss = nn.L1Loss()

    total_loss = l1_loss(predict, gt) 
    return total_loss

import math
def evaluation(predict, gt):
    mse = F.mse_loss(predict,gt)
    mse = paddle.mean(mse)
    PIXEL_MAX =1.0
    # print(mse)
    if mse ==0:
        PSNR= 30
    else:
        PSNR = 20* math.log10(PIXEL_MAX/math.sqrt(mse))
    predict = paddle.to_tensor(predict, dtype=paddle.float32)
    gt = paddle.to_tensor(gt, dtype = paddle.float32)
    SSIM = ms_ssim(predict,gt,data_range=1)
    SSIM = SSIM.numpy()
    return PSNR, SSIM


def sample_images(epoch, i, output, gt, input, save_path):
    output, gt, input = output*255, gt*255, input*255
    
    output = paddle.clip(output.detach(), 0, 255)

    output = output.cast('int64')
    gt = gt.cast('int64')
    input = input.cast('int64')
    h,w = output.shape[-2], output.shape[-1]
    img = np.zeros((h, 3*w, 3))
    for idx in range(0,1):
        row = idx * h
        tmplist = [input[idx], gt[idx], output[idx]]
        for k in range(3):
            col = k * w 
            tmp = np.transpose(tmplist[k], (1, 2, 0))
            img[row:row + h, col:col + w] = np.array(tmp)
    img = img.astype(np.uint8)
    img = Image.fromarray(img)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    img.save(os.path.join(save_path, '%03d_%06d.png'%(epoch,i)))