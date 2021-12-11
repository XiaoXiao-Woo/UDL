# Adapted from https://github.com/Po-Hsun-Su/pytorch-ssim/blob/master/max_ssim.py

import torch
import torch.backends.cudnn as cudnn
import random
from torch.autograd import Variable
from torch import optim
from PIL import Image
import numpy as np
import sys, os
import matplotlib.pyplot as plt
import torch.nn.functional as F
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
from evaluate import analysis_accu
# from derain.cal_ssim import SSIM as cal_SSIM
# from derain.cal_ssim import ssim as cal_ssim
import time
class BCMSLoss(torch.nn.Module):

    def __init__(self, reduction='mean'):
        super(BCMSLoss, self).__init__()

        # self.l1 = torch.nn.L1Loss()
        self.reduction = reduction
        self.alpha = 0.25
        self.gamma = 2
        self.threasold = 0.1

    def forward(self, x, gt):

        #好
        a = torch.exp(gt)
        b = 1
        loss = a * x - (a+b) * torch.log(torch.exp(x)+1)
        # x = torch.sigmoid(x)
        # gt = torch.sigmoid(gt)
        # loss = self.l1(x, gt)
        # logits = torch.sigmoid(x)

        # a = torch.exp(gt)
        # b = 1
        # loss = (1-x) ** self.gamma * a * x - (a*(1-x) ** self.gamma + b*x ** self.gamma) * torch.log(torch.exp(x)+1)

        # loss = (1-x) ** self.gamma * a * x - (a*(1-x) ** self.gamma + b) * torch.log(torch.exp(x)+1)

        # x与sigmoid(x)接近，即loss小的时候,x接近1
        # loss = torch.where(loss < self.threasold, (x ** self.gamma) * loss, ((1-x) ** self.gamma) * loss)


        if self.reduction == 'none':
            return -loss
        if self.reduction == 'mean:':
            return -torch.mean(loss)

        return loss
import math
def PSNR(img1, img2, mse=None,inter=True):
    PIXEL_MAX = 1
    if inter:
        b, _, _, _ = img1.shape
        mse1 = np.mean((img1 - img2) ** 2)
        img1 = np.clip(img1 * 255, 0, 255) / 255.
        img2 = np.clip(img2 * 255, 0, 255) / 255.
        mse = np.mean((img1 - img2) ** 2)
        if mse == 0:
            return 100
        print(mse, mse1)
        return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
    else:
        if hasattr(mse, 'item'):
            mse = mse.item()
            if mse == 0:
                return 100
            return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

seed = 1
torch.cuda.empty_cache()
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
cudnn.deterministic = True
# npImg1 = np.array(Image.open("kodim10.png"))
# plt.imshow(npImg1)
# plt.show()
# img1 = torch.from_numpy(npImg1).float().unsqueeze(0).unsqueeze(0)/255.0
img1 = torch.from_numpy(np.array(Image.open("kodim10.png"))).permute(2, 0, 1).unsqueeze(0) / 255.0#.unsqueeze(-1)

img1 = Image.open('kodim10.png')
img1 = np.array(img1).astype(np.float32) / 255.
img_batch = []
img_noise_batch = []
for sigma in range(10, 110, 100):
    noise = sigma * np.random.rand(*img1.shape)
    # img_noise = (img1 + noise).astype(np.float32).clip(0, 1)
    img_torch = torch.from_numpy(img1).unsqueeze(0).permute(0, 3, 1, 2)  # 1, C, H, W
    # img_noise_torch = torch.from_numpy(img_noise).unsqueeze(0).permute(0, 3, 1, 2)
    img_batch.append(img_torch)
    # img_noise_batch.append(img_noise_torch)
img1 = torch.cat(img_batch, dim=0)
# img2 = torch.cat(img_noise_batch, dim=0)
img2 = torch.from_numpy(sigma * np.random.rand(*img1.shape)).float()
# img2 = torch.tensor(img2, requires_grad=True)
img2 = (img2).clamp(0, 1)
print(img1.size())
# img2 = torch.rand(img1.size())
# torch.nn.init.kaiming_uniform_(img2)
conv1 = torch.nn.Conv2d(3, 3, kernel_size=1).cuda()
if torch.cuda.is_available():
    img1 = img1.cuda()
    img2 = img2.cuda()

img1 = Variable( img1,  requires_grad=False)
img2 = Variable( img2,  requires_grad=True)
# img1[:,1,...] = img1[:,1,...] * 2
ssim_value = ssim(img1, img2, data_range=1).item()
print("Initial ssim:", ssim_value)

ssim_loss = SSIM(win_size=11, win_sigma=1.5, data_range=1, size_average=False, channel=3)
ms_ssim_loss = MS_SSIM(win_size=11, win_sigma=1.5, data_range=1, size_average=False, channel=3)
L1_loss = torch.nn.L1Loss(reduction='none').cuda()
L2_loss = torch.nn.MSELoss(reduction='none').cuda()#l2更适合作为正态分布的收敛目标
bce_loss = BCMSLoss(reduction='none').cuda()

weights = torch.rand(3, requires_grad=True)#torch.nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)#
pos = torch.nn.Parameter(torch.zeros(img1.size()), requires_grad=True)
torch.nn.init.trunc_normal_(pos)
print(weights.is_leaf)
print(img2.is_leaf)
relu = torch.nn.ReLU()
act = torch.nn.Softmax()
# weight = rs_w / (torch.sum(rs_w, dim=0) + self.epsilon)

optimizer = optim.Adam([img2, weights], lr=0.01)

# bce 206   10*rand 1611  zero_init 198 对应采用了bn\in\gn\ln的都存在标准正态假设
# l2 192    10*rand 2859  zero_init 183
# l1 inf
# 1-ssim 231
# 1-ssim+l2 177
# 1-ssim+l1 187 10:1, l1>1-ssim 222 100:1
# 1-ssim+l1+l2 149 1:0.1:1 | 172 0.01:1:1 | 127 1:0.1:10 10rand 1909
#              144
# 1-ssim+l1+l2+bce 127 1:0.1:10:1e-4
# 1:1:1 weights none->125 可学习的权值
iters = 0
starts = time.time()
# while ssim_value < 0.95:
for i in range(200):
    optimizer.zero_grad()
    weights = relu(weights.cuda())
    # weights = act(weights.cuda())
    # weights = weights / (torch.sum(weights, dim=0) + 1e-4)
    # with torch.no_grad():
    # img2 = img2 + pos.cuda()
    # out = conv1(img2)
    # img2 = torch.clamp(img2 * 255, 0, 255) / 255.
    pred = img2#torch.clamp(img2, 0, 1)
    l2_loss = L2_loss(img1, img2) #torch.mean((img1-pred) ** 2) * 0.5
    with torch.no_grad():
        l2 = L2_loss(img1, pred)
    '''
    1.9963338e-09 2.376446e-09
[200] ms_ssim_loss: 1.1930929133541213e-07 ssim_loss: 5.961464353276824e-07 ssim: 0.9999994039535522 l1_loss: 2.2356467525241897e-05 l2_loss 2.5767881162153117e-09
    '''
    # l2_loss = L2_loss(img1, img2)
    _ms_ssim_loss = torch.abs(1 - ms_ssim_loss(img1, pred))
    _ssim_loss = torch.abs(1 - ssim_loss(img1, pred))
    l1_loss = L1_loss(img1, pred)
    bce = bce_loss(pred, img1)
    '''
    4次
    [200] ms_ssim_loss: 2.384185791015625e-07 ssim_loss: 2.384185791015625e-07 ssim: 0.9999997019767761 l1_loss: 7.4748190854734275e-06 l2_loss 2.8102925542228263e-10 loss: 3.743565741842758e-07	PSNR: 98.29052798003323
    2个的3次方 
    ssim[200] ms_ssim_loss: 1.1920928955078125e-07 ssim_loss: 0.0 ssim: 1.0 l1_loss: 1.7920958271133713e-05 l2_loss 9.753723384520185e-10 loss: 0.0	PSNR: 90.95457810158179
    [200] ms_ssim_loss: 1.1920928955078125e-07 ssim_loss: 2.384185791015625e-07 ssim: 0.9999997615814209 l1_loss: 1.1696834917529486e-05 l2_loss 4.5147510729925955e-10 loss: 3.0081952218097285e-07	PSNR: 96.04296579824774
    abs 1
    [200] ms_ssim_loss: 1.1920928955078125e-07 ssim_loss: 0.0 ssim: 0.9999999403953552 l1_loss: 1.6058093024184927e-05 l2_loss 8.081534885739927e-10 loss: 0.0	PSNR: 92.37840540281498-》96 seed=2
    
    [200] ms_ssim_loss: 2.384185791015625e-07 ssim_loss: 2.384185791015625e-07 ssim: 0.9999999403953552 l1_loss: 9.250425136997364e-06 l2_loss 4.185984336935178e-10 loss: 7.152557373046875e-07	PSNR: 96.3441036779089
    '''
    # loss = l1_loss + l2_loss + _ssim_loss
    with torch.no_grad():
        w_1 = torch.mean(l1_loss) + 1e-10
        w_2 = torch.mean(l2) + 1e-10
        # t_l2_loss = l2_loss.reshape(l2_loss.shape[0]*l2_loss.shape[1], -1)
        # w_22 = torch.mean(t_l2_loss @ t_l2_loss.t()) + 1e-8
        w_ms = torch.mean(_ms_ssim_loss) + 1e-10
        w_s = torch.mean(_ssim_loss) + 1e-10
        w_bce = torch.mean(bce_loss(img2, img1)) + 1e-10
    #[1] ssim_loss: 0.9894499182701111 ssim: 0.01221519149839878 l1_loss: 0.27382731437683105 l2_loss 0.10724163055419922
    #[1077] ssim_loss: 0.00016695261001586914 ssim: 0.9999005198478699 l1_loss: 0.0005085289594717324 l2_loss 4.5317159447222366e-07
    # loss = l1_loss / w_1 * weights[0] + l2_loss / w_2 * weights[1] + _ssim_loss / w_s * weights[2]
    # loss = 0.1 * l1_loss + 10 * l2_loss + _ssim_loss
    # loss = (l1_loss/w_1 + _ssim_loss/w_s) * l2_loss + bce / w_bce * _ssim_loss#128
    # loss = _ssim_loss * (torch.abs(l1_loss/w_1) ** 1 + torch.abs(l2_loss/w_2) ** 1 + 1e-8) ** 1#119
    # loss = _ssim_loss * ((l1_loss / w_1) ** 1+ (l2_loss / w_2) ** 1) ** 1# 119 l1 69 l2 71 | l1+l2 92 | l1+l2+bce 96 | bce 37 | ssim_loss 37
    #k uniform l1 46.5 | l1+l2 42.7,对l1的解析域不友好 | 因此 new_loss + l1_loss 增强l1解析 64
    # print(_ssim_loss.reshape(-1,1,1,1).size())
    # loss = _ssim_loss.reshape(-1,1,1,1) * ((l1_loss / w_1) ** 1 + (l2_loss / w_2) ** 1 + (bce / w_bce) ** 1 + 1e-8) ** 1# 119
    loss = l2_loss * ((_ssim_loss.reshape(-1,1,1,1) / w_s) ** 1 + (l1_loss/ w_1) ** 1 + 1e-8) ** 1
    # loss = l2_loss * ((_ssim_loss.reshape(-1,1,1,1) / w_s) ** 1 + (l1_loss/ w_1) ** 1 + 1e-8) ** 1+ \
    #        _ssim_loss.reshape(-1,1,1,1) * ((l1_loss / w_1) ** 1+ (l2_loss / w_2) ** 1 + 1e-8) ** 1#让l2朝更多方向解析，却增强l1 82
    #img1+noise:  new_loss 79  l2 59 mixed 67 | noise: l2 59 new_loss 69 mixed 67
    # loss = _ssim_loss.reshape(-1,1,1,1) * ((l1_loss / w_1) ** 1+ (l2_loss / w_2) ** 1 + 1e-8) ** 1
    # loss = l2_loss
    '''
    [52] ms_ssim_loss: 0.020025253295898438 ssim_loss: 0.1038517951965332 ssim: 0.9051538705825806 l1_loss: 0.011928136460483074 l2_loss 0.0008697847370058298 loss: 0.16643451154232025	PSNR: 31.157792430757226
    [52] ms_ssim_loss: 0.019867420196533203 ssim_loss: 0.1033320426940918 ssim: 0.905561089515686 l1_loss: 0.011895130388438702 l2_loss 0.0008638759609311819 loss: 0.16570042073726654	PSNR: 31.184488657994113
    [75] ms_ssim_loss: 0.0018777847290039062 ssim_loss: 0.010007619857788086 ssim: 0.9909011721611023 l1_loss: 0.0036819204688072205 l2_loss 4.1384337237104774e-05 loss: 0.015481384471058846	PSNR: 44.333620208611784
    [75] ms_ssim_loss: 0.0018880963325500488 ssim_loss: 0.010046124458312988 ssim: 0.9908728003501892 l1_loss: 0.0036844678688794374 l2_loss 4.143712067161687e-05 loss: 0.015536490827798843	PSNR: 44.330917725115796
    
    3
    [58] ms_ssim_loss: 0.010408163070678711 ssim_loss: 0.050076305866241455 ssim: 0.9543293714523315 l1_loss: 0.009060760028660297 l2_loss 0.00043947529047727585 loss: 0.11061141639947891	PSNR: 34.13602822940696
    [118] ms_ssim_loss: 2.574920654296875e-05 ssim_loss: 0.00010502338409423828 ssim: 0.9999041557312012 l1_loss: 0.0004919272032566369 l2_loss 7.334001566050574e-07 loss: 0.00022112477745395154	PSNR: 62.25928483438019
    2
    [60] ms_ssim_loss: 0.009631514549255371 ssim_loss: 0.05235922336578369 ssim: 0.9528141021728516 l1_loss: 0.008544023148715496 l2_loss 0.0003161841304972768 loss: 0.08355379104614258	PSNR: 35.62479009262329
    [120] ms_ssim_loss: 2.1696090698242188e-05 ssim_loss: 0.00010788440704345703 ssim: 0.9999022483825684 l1_loss: 0.0003812575014308095 l2_loss 4.93686570735008e-07 loss: 0.00016859867901075631	PSNR: 63.769501386970255
    '''
    # loss = (l1_loss/w_1 + l2_loss/w_2) * _ssim_loss#121  (l1_loss/w_1) 121
    # loss = L2_loss(img1, img2)#bce_loss(img2, img1)
    loss = torch.abs(torch.mean(loss))
    loss.backward(retain_graph=True)
    optimizer.step()

    ssim_value = ssim(img1, pred, data_range=1).item()
    iters += 1
    print(f'[{iters}] ms_ssim_loss: {w_ms} ssim_loss: {w_s} ssim: {ssim_value} l1_loss: {w_1} l2_loss {w_2} bce {w_bce} loss: {loss}\t'
          f'PSNR: {PSNR(img1.data.cpu().numpy(), img2.data.cpu().numpy())}, PSNR_L: {PSNR(None, None, w_2, False)}')
    our_CC, our_PSNR, our_SSIM, our_SAM, our_ERGAS = analysis_accu(img2[0, ...], img1[0, ...], 4)
    print(f'our_CC: {our_CC}, our_PSNR: {our_PSNR}, '
          f'our_SSIM: {our_SSIM},\n'
          f'our_SAM: {our_SAM} our_ERGAS: {our_ERGAS}')

print("time:", time.time() - starts)
img2_ = (img2 * 255.0).squeeze(0)
np_img2 = img2_.permute(1, 2, 0).detach().cpu().numpy().astype(np.uint8)
plt.imshow(np_img2)
plt.show()
if np_img2.shape[-1] == 1:
    Image.fromarray(np_img2.squeeze()).save('results.png')
else:
    Image.fromarray(np_img2).save('results.png')
'''
[200] ms_ssim_loss: 0.0012507656356319785 ssim_loss: 0.00045981191215105355 ssim: 0.9995664358139038 l1_loss: 0.010951894335448742 l2_loss 0.00028425443451851606 loss: 0.0014714214485138655	PSNR: 35.73041617910202
'''