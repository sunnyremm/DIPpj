# coding: utf-8
import argparse
import os
import datetime
from tqdm import tqdm
import time

import torch
from torch import nn
from torch import optim
from torch.backends import cudnn
from torch.utils.data import DataLoader
import torchvision.models as models
import torch.nn.functional as F 

from model import DM2FNet
from tools.config import TRAIN_ITS_ROOT, TEST_SOTS_ROOT
from datasets import ItsDataset, SotsDataset
from tools.utils import AvgMeter, check_mkdir

from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def parse_args():
    parser = argparse.ArgumentParser(description='Train a DM2FNet')
    parser.add_argument(
        '--gpus', type=str, default='0', help='gpus to use ')
    parser.add_argument('--ckpt-path', default='./ckpt', help='checkpoint path')
    parser.add_argument(
        '--exp-name',
        default='RESIDE_ITS_Loss2_lr=6e-4',
        help='experiment name.')
    args = parser.parse_args()

    return args


cfgs = {
    'use_physical': True,
    'iter_num': 40000,
    'train_batch_size': 16,
    'last_iter': 0,
    'lr': 6e-4,
    'lr_decay': 0.9,
    'weight_decay': 0,
    'momentum': 0.9,
    'snapshot': '',
    'val_freq': 2000,
    'crop_size': 256
}


def main():
    start_time = time.time()  # 记录程序开始时间

    net = DM2FNet().cuda().train()
    # net = nn.DataParallel(net)

    optimizer_G = optim.Adam([
        {'params': [param for name, param in net.named_parameters()
                    if name[-4:] == 'bias' and param.requires_grad],
         'lr': 2 * cfgs['lr']},
        {'params': [param for name, param in net.named_parameters()
                    if name[-4:] != 'bias' and param.requires_grad],
         'lr': cfgs['lr'], 'weight_decay': cfgs['weight_decay']}
    ])

    if len(cfgs['snapshot']) > 0:
        print('training resumes from \'%s\'' % cfgs['snapshot'])
        net.load_state_dict(torch.load(os.path.join(args.ckpt_path,
                                                    args.exp_name, cfgs['snapshot'] + '.pth')))
        optimizer_G.load_state_dict(torch.load(os.path.join(args.ckpt_path,
                                                          args.exp_name, cfgs['snapshot'] + '_optim.pth')))
        optimizer_G.param_groups[0]['lr'] = 2 * cfgs['lr']
        optimizer_G.param_groups[1]['lr'] = cfgs['lr']

    check_mkdir(args.ckpt_path)
    check_mkdir(os.path.join(args.ckpt_path, args.exp_name))
    open(log_path, 'w').write(str(cfgs) + '\n\n')

    discriminator = Discriminator().cuda().train()
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=1e-4)
    total_loss_fn = TotalLoss(discriminator)
    train(net, discriminator, optimizer_G, optimizer_D, total_loss_fn)

    end_time = time.time()  # 记录程序结束时间
    total_time = end_time - start_time  # 计算总的运行时间
    print(f"Total training time: {total_time:.2f} seconds")
    open(log_path, 'a').write(f"Total training time: {total_time:.2f} seconds\n")


class VGGFeatureExtractor(nn.Module):
    def __init__(self, layer_ids=[4, 9, 18, 27]):
        super(VGGFeatureExtractor, self).__init__()
        vgg = models.vgg19(pretrained=True).features
        self.features = nn.ModuleList([vgg[i] for i in layer_ids])

    def forward(self, x):
        outputs = []
        for layer in self.features:
            x = layer(x)
            outputs.append(x)
        return outputs

class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        self.feature_extractor = VGGFeatureExtractor()
        self.mse_loss = nn.MSELoss()

    def forward(self, input, target):
        input_features = self.feature_extractor(input)
        target_features = self.feature_extractor(target)
        loss = 0
        for input_feature, target_feature in zip(input_features, target_features):
            loss += self.mse_loss(input_feature, target_feature)
        return loss

class Discriminator(nn.Module):
    def __init__(self, input_channels=3):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=0)
        )

    def forward(self, x):
        return self.model(x)

class AdversarialLoss(nn.Module):
    def __init__(self, discriminator):
        super(AdversarialLoss, self).__init__()
        self.discriminator = discriminator
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, real, fake):
        real_output = self.discriminator(real)
        fake_output = self.discriminator(fake)
        
        real_label = torch.ones_like(real_output).cuda()
        fake_label = torch.zeros_like(fake_output).cuda()

        real_loss = self.bce_loss(real_output, real_label)
        fake_loss = self.bce_loss(fake_output, fake_label)
        return real_loss + fake_loss


class TotalLoss(nn.Module):
    def __init__(self, discriminator):
        super(TotalLoss, self).__init__()
        self.perceptual_loss = PerceptualLoss()
        self.adversarial_loss = AdversarialLoss(discriminator)
        self.l1_loss = nn.L1Loss()

    def forward(self, real, fake):
        perceptual_loss = self.perceptual_loss(fake, real)
        adversarial_loss = self.adversarial_loss(real, fake)
        l1_loss = self.l1_loss(fake, real)
        total_loss = perceptual_loss + 0.1 * adversarial_loss + l1_loss
        return total_loss
    

def train(generator, discriminator, optimizer_G, optimizer_D, total_loss_fn):
    curr_iter = cfgs['last_iter']

    while curr_iter <= cfgs['iter_num']:
        train_loss_record = AvgMeter()
        loss_x_jf_record, loss_x_j0_record = AvgMeter(), AvgMeter()
        loss_x_j1_record, loss_x_j2_record = AvgMeter(), AvgMeter()
        loss_x_j3_record, loss_x_j4_record = AvgMeter(), AvgMeter()
        loss_t_record, loss_a_record = AvgMeter(), AvgMeter()

        for data in train_loader:
            lr = cfgs['lr'] * (1 - float(curr_iter) / cfgs['iter_num']) ** cfgs['lr_decay']
            for param_group in optimizer_G.param_groups:
                param_group['lr'] = lr
            for param_group in optimizer_D.param_groups:
                param_group['lr'] = lr

            haze, gt_trans_map, gt_ato, gt, _ = data

            batch_size = haze.size(0)

            haze = haze.cuda()
            gt_trans_map = gt_trans_map.cuda()
            gt_ato = gt_ato.cuda()
            gt = gt.cuda()

            # ========== Train Discriminator ==========
            optimizer_D.zero_grad()
            x_jf, x_j0, x_j1, x_j2, x_j3, x_j4, t, a = generator(haze)
            fake_images = x_jf.detach()  # Detach to avoid gradient computation for generator

            gt_resized = F.interpolate(gt, size=fake_images.size()[2:], mode='bilinear', align_corners=True)
            d_loss = total_loss_fn.adversarial_loss(gt_resized, fake_images)
            d_loss.backward()
            optimizer_D.step()

            # ========== Train Generator ==========
            optimizer_G.zero_grad()
            g_loss = total_loss_fn(gt, x_jf)

            loss_x_jf = criterion(x_jf, gt)
            loss_x_j0 = criterion(x_j0, gt)
            loss_x_j1 = criterion(x_j1, gt)
            loss_x_j2 = criterion(x_j2, gt)
            loss_x_j3 = criterion(x_j3, gt)
            loss_x_j4 = criterion(x_j4, gt)

            loss_t = criterion(t, gt_trans_map)
            loss_a = criterion(a, gt_ato)

            loss = g_loss + loss_x_j0 + loss_x_j1 + loss_x_j2 + loss_x_j3 + loss_x_j4 \
                   + 10 * loss_t + loss_a
            loss.backward()

            optimizer_G.step()

            # update recorder
            train_loss_record.update(loss.item(), batch_size)

            loss_x_jf_record.update(loss_x_jf.item(), batch_size)
            loss_x_j0_record.update(loss_x_j0.item(), batch_size)
            loss_x_j1_record.update(loss_x_j1.item(), batch_size)
            loss_x_j2_record.update(loss_x_j2.item(), batch_size)
            loss_x_j3_record.update(loss_x_j3.item(), batch_size)
            loss_x_j4_record.update(loss_x_j4.item(), batch_size)

            loss_t_record.update(loss_t.item(), batch_size)
            loss_a_record.update(loss_a.item(), batch_size)

            curr_iter += 1

            log = '[iter %d], [train loss %.5f], [loss_x_fusion %.5f], [loss_x_phy %.5f], [loss_x_j1 %.5f], ' \
                  '[loss_x_j2 %.5f], [loss_x_j3 %.5f], [loss_x_j4 %.5f], [loss_t %.5f], [loss_a %.5f], ' \
                  '[lr %.13f]' % \
                  (curr_iter, train_loss_record.avg, loss_x_jf_record.avg, loss_x_j0_record.avg,
                   loss_x_j1_record.avg, loss_x_j2_record.avg, loss_x_j3_record.avg, loss_x_j4_record.avg,
                   loss_t_record.avg, loss_a_record.avg, optimizer_G.param_groups[0]['lr'])
            print(log)
            open(log_path, 'a').write(log + '\n')

            if (curr_iter + 1) % cfgs['val_freq'] == 0:
                validate(generator, curr_iter, optimizer_G)

            if curr_iter > cfgs['iter_num']:
                break



def validate(net, curr_iter, optimizer):
    print('validating...')
    net.eval()

    loss_record = AvgMeter()

    with torch.no_grad():
        for data in tqdm(val_loader):
            haze, gt, _ = data

            haze = haze.cuda()
            gt = gt.cuda()

            dehaze = net(haze)

            loss = criterion(dehaze, gt)
            loss_record.update(loss.item(), haze.size(0))

    snapshot_name = 'iter_%d_loss_%.5f_lr_%.6f' % (curr_iter + 1, loss_record.avg, optimizer.param_groups[0]['lr'])
    print('[validate]: [iter %d], [loss %.5f]' % (curr_iter + 1, loss_record.avg))
    torch.save(net.state_dict(),
               os.path.join(args.ckpt_path, args.exp_name, snapshot_name + '.pth'))
    torch.save(optimizer.state_dict(),
               os.path.join(args.ckpt_path, args.exp_name, snapshot_name + '_optim.pth'))

    net.train()


if __name__ == '__main__':
    args = parse_args()

    # os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    cudnn.benchmark = True
    torch.cuda.set_device(int(args.gpus))

    train_dataset = ItsDataset(TRAIN_ITS_ROOT, True, cfgs['crop_size'])
    train_loader = DataLoader(train_dataset, batch_size=cfgs['train_batch_size'], num_workers=4,
                              shuffle=True, drop_last=True)

    val_dataset = SotsDataset(TEST_SOTS_ROOT)
    val_loader = DataLoader(val_dataset, batch_size=8)

    criterion = nn.L1Loss().cuda()
    log_path = os.path.join(args.ckpt_path, args.exp_name, str(datetime.datetime.now()) + '.txt')

    main()
