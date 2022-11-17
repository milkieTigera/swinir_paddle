import numpy as np
import cv2
import os
import os.path as osp
crop_size = 128
step = 128
thresh_size = 0

def prepare_patch_data(path, save_folder):

    save_x = os.path.join(save_folder,'x')
    save_x2 = os.path.join(save_folder,'x2')
    save_x4 = os.path.join(save_folder,'x4')
    if not os.path.exists(save_x):
        os.mkdir(save_x)
    if not os.path.exists(save_x2):
        os.mkdir(save_x2)
    if not os.path.exists(save_x4):
        os.mkdir(save_x4)
    # img_name, extension = osp.splitext(osp.basename(path))
    for img_name in os.listdir(path):
        file_path_x = os.path.join(path, img_name)
        file_path_x2 = file_path_x.replace("x","x2")
        file_path_x4 = file_path_x.replace("x","x4")
        print(file_path_x2,file_path_x4)
        save_patch(img_name, [file_path_x, file_path_x2, file_path_x4], [save_x, save_x2, save_x4], crop_size, step, thresh_size)

def cal_gradient(img):
    H, W = img.shape[:2]

    sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=3)
    sobelx = cv2.convertScaleAbs(sobelx)
    sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=3)
    sobely = cv2.convertScaleAbs(sobely)
    sobelxy = cv2.addWeighted(sobelx,0.5,sobely,0.5,0)
    return np.mean(sobelxy)

def img_config(file_path, img_name, crop_size,step):
    img = cv2.imread(file_path)
    h, w = img.shape[0:2]
    img_name = img_name.replace('.png',"")
    h_space = np.arange(0, h - crop_size + 1, step)
    if h - (h_space[-1] + crop_size) > thresh_size:
        h_space = np.append(h_space, h - crop_size)
    w_space = np.arange(0, w - crop_size + 1, step)
    if w - (w_space[-1] + crop_size) > thresh_size:
        w_space = np.append(w_space, w - crop_size)
    return img, h_space, w_space, img_name

def save_patch(img_name, file_path,save_x, crop_size, step, thresh_size):
    # print(img_name)
    img, h_space, w_space, img_name_x1 = img_config(file_path[0], img_name, crop_size, step)
    img_x2, h_space_x2, w_space_x2, img_name_x2 = img_config(file_path[1], img_name, crop_size*2, step*2)
    img_x4, h_space_x4, w_space_x4, img_name_x4 = img_config(file_path[2], img_name, crop_size*4, step*4)
    # print(h_space, h_space_x2, h_space_x4) 
    index = 0
    for x, x2, x4 in zip(h_space, h_space_x2, h_space_x4):
        for y, y2, y4 in zip(w_space, w_space_x2, w_space_x4):
            index += 1
            cropped_img = img[x:x + crop_size, y:y + crop_size, ...]
            gradient =  cal_gradient(cropped_img)
            if gradient <10:
                continue
           
            cropped_img_x2 = img_x2[x2:x2+crop_size*2, y2:y2+crop_size*2]
            cropped_img_x4 = img_x4[x4:x4+crop_size*4, y4:y4+crop_size*4]
            
            cropped_img = np.ascontiguousarray(cropped_img)
            cropped_img_x2 = np.ascontiguousarray(cropped_img_x2)
            cropped_img_x4 = np.ascontiguousarray(cropped_img_x4)
            

            save_path = osp.join(save_x[0], f'{img_name_x1}_s{index:03d}.png')
            # print(save_path, img_name,"1")
            cv2.imwrite(save_path, cropped_img)

            save_path_x2 = osp.join(save_x[1], f'{img_name_x2}_s{index:03d}.png')
            cv2.imwrite(save_path_x2, cropped_img_x2)

            save_path_x4 = osp.join(save_x[2], f'{img_name_x4}_s{index:03d}.png')
            cv2.imwrite(save_path_x4, cropped_img_x4)
        #     break
        # break
    # break

save_folder='/home/aistudio/patch_dataset1'
path = ['/home/aistudio/dataset/train/04/x/','/home/aistudio/dataset/train/05/x/',
'/home/aistudio/dataset/train/06/x/','/home/aistudio/dataset/train/07/x/',
'/home/aistudio/dataset/train/08/x/']
for p in path:
    prepare_patch_data(p, save_folder)
