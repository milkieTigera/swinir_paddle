import os
import sys
import glob
import cv2
import paddle
import numpy as np
from swin_IR import SwinIR

def single2tensor3(img):
    return paddle.to_tensor(np.ascontiguousarray(img)).transpose([2, 0, 1])


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

def concat_patch(b_patch):
    in_patch_1 = b_patch[0,:,:,:].unsqueeze(0)
    in_patch_2 = b_patch[1,:,:,:].unsqueeze(0)
    in_patch_3 = b_patch[2,:,:,:].unsqueeze(0)
    in_patch_4 = b_patch[3,:,:,:].unsqueeze(0)
    h_1 = paddle.concat([in_patch_1, in_patch_2], axis=2)
    h_2 = paddle.concat([in_patch_3, in_patch_4], axis=2)
    w = paddle.concat([h_1, h_2], axis=3)
    return w

def test(img_lq, model, window_size):
    is_tile = 256
    if is_tile is None:
        # test the image as a whole
        output = model(img_lq)
    else:
        b, c, h, w = img_lq.shape
        tile = min(is_tile, h, w)
        assert tile % window_size == 0, "tile size should be a multiple of window_size"
        tile_overlap = 5
        sf = 4

        stride = tile - tile_overlap
        h_idx_list = list(range(0, h-tile, stride)) + [h-tile]
        w_idx_list = list(range(0, w-tile, stride)) + [w-tile]
        type = img_lq.dtype
        E_x4 = paddle.zeros([b, c, h*sf, w*sf],dtype= type)
        E_x2 = paddle.zeros([b, c, h*(sf//2), w*(sf//2)],dtype= type)

        for h_idx in h_idx_list:
            for w_idx in w_idx_list:

                in_patch = img_lq[..., h_idx:h_idx+tile, w_idx:w_idx+tile]
                in_patch_1 = in_patch[:,:,:128,:128]
                in_patch_2 = in_patch[:,:,128:,:128]
                in_patch_3 = in_patch[:,:,:128,128:]
                in_patch_4 = in_patch[:,:,128:,128:]

                b_patch = paddle.concat([in_patch_1, in_patch_2, in_patch_3, in_patch_4], axis=0)
                out_patch_x2, out_patch_x4 = model(b_patch)
                out_patch_x2 = concat_patch(out_patch_x2)
                out_patch_x4 = concat_patch(out_patch_x4)
                E_x2[..., h_idx*(sf//2):(h_idx+tile)*(sf//2), w_idx*(sf//2):(w_idx+tile)*(sf//2)] = out_patch_x2
                E_x4[..., h_idx*sf:(h_idx+tile)*sf, w_idx*sf:(w_idx+tile)*sf] = out_patch_x4

    return E_x2, E_x4

def post_preprocess(output):
    out_img = paddle.clip(output, 0, 1)
    out_img = paddle.transpose(out_img, [1, 2, 0])
    out_img =  (out_img*255).numpy().astype(np.uint8)
    return out_img


def process(src_image_dir, pred_x2_dir, pred_x4_dir):
    window_size = 8
    model = SwinIR(upscale=4, in_chans=3, img_size=(128,128), window_size=8,
                        img_range=1., depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],
                        mlp_ratio=2, upsampler='nearest+conv', resi_connection='1conv')
    path ='model_Step3_515k.param'
    model.set_state_dict(paddle.load(path))
    model.eval()
    image_paths = glob.glob(os.path.join(src_image_dir, "*.png"))
    for image_path in image_paths:
        # do something
        img = imread_uint(image_path, 3)
        img = uint2single(img)
        img = single2tensor3(img)
        img = img.unsqueeze(0)
        
        with paddle.no_grad():
            _, _, h_old, w_old = img.shape
            h_pad = (h_old //window_size+1)*window_size-h_old
            w_pad = (w_old//window_size+1)*window_size-w_old
            img_lq = paddle.concat([img, paddle.flip(img,[2])], 2)[:,: ,:h_old+h_pad,:]
            img_lq = paddle.concat([img, paddle.flip(img,[3])], 3)[:,: ,:, :w_old+w_pad]
            # print(img_lq.shape)
            output_x2, output_x4 = test(img_lq, model, window_size)
            output_x2 = output_x2[..., :h_old * 2, :w_old * 2]
            output_x4 = output_x4[..., :h_old * 4, :w_old * 4]
            out_img_x2 = post_preprocess(output_x2[0])
            out_img_x4 = post_preprocess(output_x4[0])
        # 保存结果图片
        save_path_x2 = os.path.join(pred_x2_dir, os.path.basename(image_path))
        cv2.imwrite(save_path_x2, out_img_x2)
        save_path_x4 = os.path.join(pred_x4_dir, os.path.basename(image_path))
        cv2.imwrite(save_path_x4, out_img_x4)


if __name__ == "__main__":
    assert len(sys.argv) == 4

    src_image_dir = sys.argv[1]
    pred_x2_dir = sys.argv[2]
    pred_x4_dir = sys.argv[3]

    if not os.path.exists(pred_x2_dir):
        os.makedirs(pred_x2_dir)
    if not os.path.exists(pred_x4_dir):
        os.makedirs(pred_x4_dir)

    process(src_image_dir, pred_x2_dir, pred_x4_dir)
