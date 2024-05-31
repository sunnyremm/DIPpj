# coding: utf-8
import os

import numpy as np
import time
import torch
from torch import nn
from torchvision import transforms
import datetime ##

from tools.config import TEST_SOTS_ROOT, OHAZE_ROOT, TEST_HAZERD_ROOT
from tools.utils import AvgMeter, check_mkdir, sliding_forward
from model import DM2FNet, DM2FNet_woPhy
from datasets import SotsDataset, OHazeDataset, HazeRDDataset
from torch.utils.data import DataLoader
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from skimage.metrics import mean_squared_error ##
from skimage import color ##
from colormath.color_objects import sRGBColor, LabColor ##
from colormath.color_conversions import convert_color ##
from colormath.color_diff import delta_e_cie2000 ##

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

torch.manual_seed(2018)
torch.cuda.set_device(0)

ckpt_path = './ckpt'
# exp_name = 'RESIDE_ITS'
exp_name = 'O-Haze'
# exp_name = 'HazeRD'

args = {
    # 'snapshot': 'iter_20000_loss_0.04733_lr_0.000000', ## O-Haze
    'snapshot': 'iter_4000_loss_0.04886_lr_0.000164', ## O-Haze_Loss_4000
    # 'snapshot': 'iter_5000_loss_0.04896_lr_0.000154', ## O-Haze_Loss_5000
    # 'snapshot': 'iter_20000_loss_0.04830_lr_0.000000', ## O-Haze_Loss_20000
    # 'snapshot': 'iter_20000_loss_0.04702_lr_0.000074', ## O-Haze_Loss2_20000
    # 'snapshot': 'iter_30000_loss_0.04648_lr_0.000000', ## O-Haze_Loss2_30000
    # 'snapshot': 'iter_10000_loss_0.04719_lr_0.000107', ## O-Haze2_10000
    # 'snapshot': 'iter_20000_loss_0.04646_lr_0.000000', ## O-Haze2


    # 'snapshot': 'iter_40000_loss_0.07562_lr_0.000000', ## RESIDE_ITS_batchsize=1
    # 'snapshot': 'iter_40000_loss_0.01393_lr_0.000000', ## RESIDE_ITS_batchsize=8
    # 'snapshot': 'iter_40000_loss_0.01207_lr_0.000000', ## RESIDE_ITS_batchsize=16
    # 'snapshot': 'iter_40000_loss_0.01053_lr_0.000000', ## RESIDE_ITS_batchsize=32
    # 'snapshot': 'iter_5000_loss_0.02794_lr_0.000443', ## RESIDE_ITS_Loss_5000
    # 'snapshot': 'iter_10000_loss_0.02366_lr_0.000386', ## RESIDE_ITS_Loss_10000
    # 'snapshot': 'iter_20000_loss_0.01610_lr_0.000268', ## RESIDE_ITS_Loss_20000
    # 'snapshot': 'iter_30000_loss_0.01388_lr_0.000144', ## RESIDE_ITS_Loss_30000
    # 'snapshot': 'iter_40000_loss_0.01330_lr_0.000000', ## RESIDE_ITS_Loss_40000
    # 'snapshot': 'iter_40000_loss_0.01333_lr_0.000000' ## RESIDE_ITS_Norm_batchsize=16
    # 'snapshot': 'iter_32000_loss_0.01230_lr_0.000117', ## RESIDE_ITS_Loss2_32000
    # 'snapshot': 'iter_40000_loss_0.01183_lr_0.000000', ## RESIDE_ITS_Loss2_40000
    # 'snapshot': 'iter_30000_loss_0.01348_lr_0.000144', ## RESIDE_ITS2_30000
    # 'snapshot': 'iter_40000_loss_0.01323_lr_0.000000', ## RESIDE_ITS2
}

to_test = {
    # 'SOTS': TEST_SOTS_ROOT,


    # 'O-Haze': OHAZE_ROOT,
    'O-Haze_Loss_4000': OHAZE_ROOT,
    # 'O-Haze_Loss_5000': OHAZE_ROOT,
    # 'O-Haze_Loss_20000': OHAZE_ROOT,
    # 'O-Haze_Loss2_20000': OHAZE_ROOT,
    # 'O-Haze_Loss2_30000': OHAZE_ROOT,
    # 'O-Haze2_10000': OHAZE_ROOT,
    # 'O-Haze2': OHAZE_ROOT,


    # 'HazeRD_batchsize=1': TEST_HAZERD_ROOT, 
    # 'HazeRD_batchsize=8': TEST_HAZERD_ROOT, 
    # 'HazeRD_batchsize=16': TEST_HAZERD_ROOT, 
    # 'HazeRD_batchsize=32': TEST_HAZERD_ROOT, 
    # 'HazeRD_Loss_5000': TEST_HAZERD_ROOT, 
    # 'HazeRD_Loss_10000': TEST_HAZERD_ROOT,
    # 'HazeRD_Loss_20000': TEST_HAZERD_ROOT,
    # 'HazeRD_Loss_30000': TEST_HAZERD_ROOT,
    # 'HazeRD_Loss_40000': TEST_HAZERD_ROOT,
    # 'HazeRD_Norm_batchsize=16': TEST_HAZERD_ROOT,
    # 'HazeRD_Loss2_32000': TEST_HAZERD_ROOT,
    # 'HazeRD_Loss2_40000': TEST_HAZERD_ROOT,
    # 'HazeRD2_30000': TEST_HAZERD_ROOT,
    # 'HazeRD2': TEST_HAZERD_ROOT,


}

to_pil = transforms.ToPILImage()

def check_mkdir2(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

def calculate_ciede2000_batch(rgb1, rgb2):
    lab1 = color.rgb2lab(rgb1)
    lab2 = color.rgb2lab(rgb2)
    delta_e = color.deltaE_ciede2000(lab1, lab2)
    return np.mean(delta_e)

def main():
    start_time = time.time()  # 记录程序开始时间
    with torch.no_grad():
        criterion = nn.L1Loss().cuda()

        for name, root in to_test.items():
            if 'SOTS' in name:
                net = DM2FNet().cuda()
                dataset = SotsDataset(root)
            elif 'O-Haze' in name:
                net = DM2FNet_woPhy().cuda()
                dataset = OHazeDataset(root, 'test')
            elif 'HazeRD' in name:
                net = DM2FNet().cuda()
                dataset = HazeRDDataset(root, 'data')
            else:
                raise NotImplementedError

            # net = nn.DataParallel(net)

            if len(args['snapshot']) > 0:
                print('load snapshot \'%s\' for testing' % args['snapshot'])
                net.load_state_dict(torch.load(os.path.join(ckpt_path, 'O-Haze_Loss', args['snapshot'] + '.pth')))

            net.eval()
            dataloader = DataLoader(dataset, batch_size=1)

            psnrs, ssims = [], []
            mses, ciede2000s = [], [] ##
            loss_record = AvgMeter()

            for idx, data in enumerate(dataloader):
                # haze_image, _, _, _, fs = data
                haze, gts, fs = data
                # print(haze.shape, gts.shape)

                # check_mkdir2(os.path.join(ckpt_path, exp_name,
                #                          '(%s) %s_%s' % (exp_name, name, args['snapshot'])))
                check_mkdir2(os.path.join(ckpt_path, exp_name, name,
                                         '(%s) %s_%s' % (exp_name, name, args['snapshot'])))

                haze = haze.cuda()

                if 'O-Haze' in name:
                    res = sliding_forward(net, haze).detach()
                else:
                    res = net(haze).detach()

                loss = criterion(res, gts.cuda())
                loss_record.update(loss.item(), haze.size(0))

                for i in range(len(fs)):
                    r = res[i].cpu().numpy().transpose([1, 2, 0])
                    gt = gts[i].cpu().numpy().transpose([1, 2, 0])
                    psnr = peak_signal_noise_ratio(gt, r)
                    psnrs.append(psnr)
                    ssim = structural_similarity(gt, r, data_range=1, multichannel=True,
                                                 gaussian_weights=True, sigma=1.5, use_sample_covariance=False, channel_axis=-1) ##
                    ssims.append(ssim)

                    # Calculate MSE
                    mse = mean_squared_error(gt, r)
                    mses.append(mse)

                    # Calculate CIEDE2000
                    ciede2000 = calculate_ciede2000_batch(r, gt)
                    ciede2000s.append(ciede2000)
                    
                    log = 'predicting for {} ({}/{}) [{}]: PSNR {:.4f}, SSIM {:.4f}, MSE {:.4f}, CIEDE2000 {:.4f}'.format(
                            name, idx + 1, len(dataloader), fs[i], psnr, ssim, mse, ciede2000) ##
                    print(log)
                    open(log_path, 'a').write(log + '\n') ##
                    
                    # print('predicting for {} ({}/{}) [{}]: PSNR {:.4f}, SSIM {:.4f}, CIEDE2000 {:.4f}'
                    #       .format(name, idx + 1, len(dataloader), fs[i], psnr, ssim, ciede2000))

                for r, f in zip(res.cpu(), fs):
                    to_pil(r).save(
                        os.path.join(ckpt_path, exp_name, name, ##
                                     '(%s) %s_%s' % (exp_name, name, args['snapshot']), '%s.png' % f))

            log = f"[{name}] L1: {loss_record.avg:.6f}, PSNR: {np.mean(psnrs):.6f}, SSIM: {np.mean(ssims):.6f}, MSE: {np.mean(mses):.6f}, CIEDE2000: {np.mean(ciede2000s):.6f}" ##
            print(log)
            open(log_path, 'a').write(log + '\n') ##
            # print(f"[{name}] L1: {loss_record.avg:.6f}, PSNR: {np.mean(psnrs):.6f}, SSIM: {np.mean(ssims):.6f}, CIEDE2000: {np.mean(ciede2000s):.6f}")
            
    end_time = time.time()  # 记录程序结束时间
    total_time = end_time - start_time  # 计算总的运行时间
    print(f"Total training time: {total_time:.2f} seconds")
    open(log_path, 'a').write(f"Total training time: {total_time:.2f} seconds\n")


if __name__ == '__main__':
    for name, root in to_test.items(): ##
        log_path = os.path.join(ckpt_path, exp_name, name, str(datetime.datetime.now()) + '.txt') ##
    main()
