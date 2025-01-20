import torch
from torch import nn
class MLP(nn.Module):
    def __init__(self, num_spectral, num_channel):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels=num_spectral, out_channels=num_channel, kernel_size=3, stride=1, padding=1, bias=True),
            nn.Conv2d(in_channels=num_channel, out_channels=num_channel, kernel_size=3, stride=1, padding=1, bias=True),
            nn.Conv2d(in_channels=num_channel, out_channels=num_channel, kernel_size=3, stride=1, padding=1, bias=True),
            nn.Conv2d(in_channels=num_channel, out_channels=num_channel, kernel_size=3, stride=1, padding=1, bias=True),
            nn.Conv2d(in_channels=num_channel, out_channels=num_channel, kernel_size=3, stride=1, padding=1, bias=True),
            nn.Conv2d(in_channels=num_channel, out_channels=num_spectral, kernel_size=3, stride=1, padding=1, bias=True)
        )

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.xavier_normal_(m.weight)

    def forward(self, y1):
        return self.mlp(y1)

class Encoding_Block(torch.nn.Module):
    def __init__(self, c_in):
        super(Encoding_Block, self).__init__()

        self.conv1 = torch.nn.Conv2d(in_channels=c_in, out_channels=32, kernel_size=3, padding=3 // 2)
        self.conv2 = torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=3 // 2)
        self.conv3 = torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=3 // 2)
        self.conv4 = torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=3 // 2)
        self.conv5 = torch.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=2, padding=3 // 2)

        self.act = torch.nn.PReLU()

        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.xavier_normal_(m.weight)

    def forward(self, input):

        out1 = self.act(self.conv1(input))
        out2 = self.act(self.conv2(out1))
        out3 = self.act(self.conv3(out2))
        f_e = self.conv4(out3)
        down = self.act(self.conv5(f_e))
        return f_e, down


class Encoding_Block_End(torch.nn.Module):
    def __init__(self, c_in=64):
        super(Encoding_Block_End, self).__init__()

        self.conv1 = torch.nn.Conv2d(in_channels=c_in, out_channels=32, kernel_size=3, padding=3 // 2)
        self.conv2 = torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=3 // 2)
        self.conv3 = torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=3 // 2)
        self.conv4 = torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=3 // 2)
        self.act = torch.nn.PReLU()
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.xavier_normal_(m.weight)

    def forward(self, input):
        out1 = self.act(self.conv1(input))
        out2 = self.act(self.conv2(out1))
        out3 = self.act(self.conv3(out2))
        f_e = self.conv4(out3)
        return f_e


class Decoding_Block(torch.nn.Module):
    def __init__(self, c_in):
        super(Decoding_Block, self).__init__()
        self.conv0 = torch.nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=3 // 2)
        self.conv1 = torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=3 // 2)
        self.conv2 = torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=3 // 2)

        self.conv3 = torch.nn.Conv2d(in_channels=64, out_channels=256, kernel_size=1)
        self.up = torch.nn.ConvTranspose2d(c_in, 64, kernel_size=3, stride=2, padding=3 // 2)

        self.act = torch.nn.PReLU()
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.xavier_normal_(m.weight)

    def forward(self, input, map):

        up = self.up(input, output_size=[input.shape[0], input.shape[1], map.shape[2], map.shape[3]])
        cat = torch.cat((up, map), 1)
        cat = self.act(self.conv0(cat))
        out1 = self.act(self.conv1(cat))
        out2 = self.act(self.conv2(out1))

        out3 = self.conv3(out2)

        return out3


class Feature_Decoding_End(torch.nn.Module):
    def __init__(self, c_out):
        super(Feature_Decoding_End, self).__init__()
        self.conv0 = torch.nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=3 // 2)
        self.conv1 = torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=3 // 2)
        self.conv2 = torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=3 // 2)

        self.conv3 = torch.nn.Conv2d(in_channels=64, out_channels=c_out, kernel_size=3, padding=3 // 2)
        self.batch = 1
        self.up = torch.nn.ConvTranspose2d(256, 64, kernel_size=3, stride=2, padding=3 // 2)
        self.act = torch.nn.PReLU()
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.xavier_normal_(m.weight)

    def forward(self, input, map):

        up = self.up(input, output_size=[input.shape[0], input.shape[1], map.shape[2], map.shape[3]])
        cat = torch.cat((up, map), 1)
        cat = self.act(self.conv0(cat))
        out1 = self.act(self.conv1(cat))
        out2 = self.act(self.conv2(out1))

        out3 = self.conv3(out2)

        return out3


class ResConv(torch.nn.Module):
    def __init__(self, cin, mid, cout, kernel_size=3):
        super(ResConv, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=cin, out_channels=mid, kernel_size=kernel_size, padding="same")
        self.conv2 = torch.nn.Conv2d(in_channels=mid, out_channels=cout, kernel_size=1, padding="same")
        self.relu = torch.nn.PReLU()

    def forward(self, x):
        res = self.conv1(x)
        res = self.relu(res)
        res = self.conv2(res)
        out = x + res
        return out

class Unet(torch.nn.Module):
    def __init__(self, hs_channel, ms_channel):
        super(Unet, self).__init__()

        self.Encoding_block1 = Encoding_Block(hs_channel)
        self.Encoding_block2 = Encoding_Block(32)
        self.Encoding_block3 = Encoding_Block(32)
        self.Encoding_block4 = Encoding_Block(32)
        self.Encoding_block_end = Encoding_Block_End(32)

        self.Decoding_block1 = Decoding_Block(64)
        self.Decoding_block2 = Decoding_Block(256)
        self.Decoding_block3 = Decoding_Block(256)
        self.Decoding_block_End = Feature_Decoding_End(hs_channel)

        # self.act = torch.nn.PReLU()
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.xavier_normal_(m.weight)

    def forward(self, lms, pan):
        encode0, down0 = self.Encoding_block1(lms)
        encode1, down1 = self.Encoding_block2(down0)
        encode2, down2 = self.Encoding_block3(down1)
        encode3, down3 = self.Encoding_block4(down2)

        media_end = self.Encoding_block_end(down3)

        decode3 = self.Decoding_block1(media_end, encode3)
        decode2 = self.Decoding_block2(decode3, encode2)
        decode1 = self.Decoding_block3(decode2, encode1)
        decode0 = self.Decoding_block_End(decode1, encode0)

        return decode0
