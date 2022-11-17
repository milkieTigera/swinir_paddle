from dataset import DatasetSR_Step3
import paddle
import random
import numpy as np
from swin_IR import SwinIR
from utils import evaluation, LossStep1
import os
import logging
from visualdl import LogWriter

def post_preprocess(output):
    out_img = paddle.clip(output, 0, 1)
    out_img = paddle.transpose(out_img, [1, 2, 0])
    out_img =  (out_img*255).numpy().astype(np.uint8)
    return out_img

SEED = 2333
random.seed(SEED)
np.random.seed(SEED)
paddle.seed(SEED)
split_ratio = 0.90
data_root_Hx4= ['/home/aistudio/patch_dataset/x4',
               '/home/aistudio/patch_dataset2/x4']
data_root_H = ["/home/aistudio/patch_dataset/x2",
              "/home/aistudio/patch_dataset2/x2"]
data_root_L= ["/home/aistudio/patch_dataset/x",
            "/home/aistudio/patch_dataset2/x"]

logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)
handler = logging.FileHandler("train_log_Step3.txt")
handler.setLevel(logging.INFO)
formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] : %(message)s ", "%Y-%m-%d %H:%M:%S")
# logging.filemode='w'
handler.setFormatter(formatter)
logger.addHandler(handler)

visual_log = LogWriter("training_log_step3")

train_set = DatasetSR_Step3('train',data_root_L, data_root_H,data_root_Hx4,split_ratio)
train_loader = paddle.io.DataLoader(train_set, batch_size=4, num_workers=0, shuffle=True, return_list = True )
val_set = DatasetSR_Step3('val',data_root_L, data_root_H,data_root_Hx4,split_ratio )
val_loader = paddle.io.DataLoader(val_set, batch_size=2, num_workers=0, shuffle=False, return_list = True )
print("train {} val {}".format(len(train_set),len(val_set)))

model = SwinIR(upscale=4, in_chans=3, img_size=(128,128), window_size=8,
                    img_range=1., depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],
                    mlp_ratio=2, upsampler='nearest+conv', resi_connection='1conv')
base_lr = 0.25*1e-4  
max_score = 0
scheduler = paddle.optimizer.lr.MultiStepDecay(base_lr, milestones=[6000000,], gamma=0.5)
optimizer = paddle.optimizer.AdamW(learning_rate=scheduler, parameters=model.parameters())


resume_path ='/home/aistudio/work/best_model/model_Step2_190k.param'
start_iter = 0
if os.path.exists(resume_path):
        model.set_state_dict(paddle.load(resume_path))
        start_iter = 0
        for i in range(start_iter):
            scheduler.step()
        print(f"Resume model from {os.path.basename(resume_path)}")
        
        

iters_per_epoch = len(train_loader)
iter1 = start_iter
num_iters = 540000
sample_path ='/home/aistudio/work/sample'
model_path = '/home/aistudio/work/output'
while iter1<num_iters:
    for i, train_data in enumerate(train_loader):

            model.train()
            iter1+=1
            if iter1>num_iters:
                break
            L_img = train_data['L']
            H_x4_img = train_data['Hx4']
            H_x2_img = train_data['H']
            generate_x2, generate_x4 = model(L_img)

            # loss x4 0.6 x2 0.4
            total_loss_x2, MSE_loss_x2 , MS_SSIM_loss_x2  = LossStep1(generate_x2, H_x2_img)
            total_loss_x4, MSE_loss_x4 , MS_SSIM_loss_x4  = LossStep1(generate_x4, H_x4_img)            
            total_loss = 0.4*  total_loss_x2 + 0.6* total_loss_x4
            optimizer.clear_grad()
            total_loss.backward()
            optimizer.step()
            
            if iter1% 100==0:
                train_msg = f"epoch {(iter1 - 1) // (iters_per_epoch + 1)} {iter1}/{num_iters} total_loss {total_loss.item()}  lr: { np.array(scheduler.get_lr())} MSE_loss_x2: {MSE_loss_x2.item()} MS_MSSIM_Loss_x2: {MS_SSIM_loss_x2.item()} MSE_loss_x4: {MSE_loss_x4.item()} MS_MSSIM_Loss_x4: {MS_SSIM_loss_x4.item()} "
                logger.info(train_msg)
                visual_log.add_scalar(tag="train_loss", step=iter1, value = total_loss.item())
                
            scheduler.step()
            if iter1%500==0:
                paddle.save(model.state_dict(),model_path+'/model_temp.param')
            if iter1%5000==0:
                model.eval()
                PSNR_tot,SSIM_tot,num =0, 0, 0
                for i, data in enumerate(val_loader):
                        L_img = data['L']
                        H_x4_img = data['Hx4']
                        H_x2_img = data['H']
                      
                        with paddle.no_grad():
                            generate_x2, generate_x4 = model(L_img)
                        _, _, h1, w1 = L_img.shape
                        PSNR_x2,SSIM_x2 = evaluation(generate_x2,H_x2_img)
                        PSNR_x4,SSIM_x4 = evaluation(generate_x4,H_x4_img)
                        PSNR_tot+=(PSNR_x2*0.4+PSNR_x4*0.6)
                        SSIM_tot+=(SSIM_x2*0.4+SSIM_x4*0.6)
                        num+=1
                        if i% 100==0:
                            test_msg = "Testing {}/{}".format(num,len(val_loader))
                            # print(test_msg)
                            logger.info(test_msg)


                PSNR_tot = PSNR_tot/num
                SSIM_tot = SSIM_tot/num
                                    
                score = 0.5*PSNR_tot + 0.5* SSIM_tot*100
                eval_msg = "PSNR {} SSIM {} score {}".format( PSNR_tot, SSIM_tot, score)
                visual_log.add_scalar(tag="eval_psnr", step=iter1, value = PSNR_tot)

                visual_log.add_scalar(tag="eval_msssim", step=iter1, value = SSIM_tot)
                
                visual_log.add_scalar(tag="eval_score", step=iter1, value = score)
                
                logger.info(eval_msg)   
                if score>max_score:

                    paddle.save(model.state_dict(),model_path+'/model_best.param')
                    max_score= score
                    max_score_msg = f"iter: {iter1} max_score: {max_score}"
                    logger.info(max_score_msg)

