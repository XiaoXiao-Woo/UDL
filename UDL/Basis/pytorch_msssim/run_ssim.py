
def ssim_loss():
    import torch
    import numpy as np
    from PIL import Image
    from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
    # X: (N,3,H,W) a batch of non-negative RGB images (0~255)
    # Y: (N,3,H,W)
    X = torch.from_numpy(np.array(Image.open("tests/kodim10.png"))).permute(2,0,1).unsqueeze(0).float()
    Y = torch.rand(X.size())
    # calculate ssim & ms-ssim for each image
    ssim_val = ssim(X, Y, data_range=255, size_average=False)  # return (N,)
    ms_ssim_val = ms_ssim(X, Y, data_range=255, size_average=False)  # (N,)

    # set 'size_average=True' to get a scalar value as loss. see tests/tests_loss.py for more details
    ssim_loss = 1 - ssim(X, Y, data_range=255, size_average=True)  # return a scalar
    ms_ssim_loss = 1 - ms_ssim(X, Y, data_range=255, size_average=True)

    print(ssim_val, ms_ssim_val, ssim_loss, ms_ssim_loss)

    # reuse the gaussian kernel with SSIM & MS_SSIM.
    ssim_module = SSIM(data_range=255, size_average=True, channel=3)
    ms_ssim_module = MS_SSIM(data_range=255, size_average=True, channel=3)

    ssim_loss = 1 - ssim_module(X, Y)
    ms_ssim_loss = 1 - ms_ssim_module(X, Y)

    print(ssim_val, ms_ssim_val, ssim_loss, ms_ssim_loss)


if __name__ == "__main__":

    ssim_loss()