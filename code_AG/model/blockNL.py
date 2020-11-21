import torch
import torch.nn as nn
import torch.nn.functional as F

class ExtractFea(torch.nn.Module):
    def __init__(self, channels):
        super(ExtractFea, self).__init__()
        self.channels = channels

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=self.channels, kernel_size=1, padding=0)
        self.conv2 = nn.Conv2d(in_channels=self.channels, out_channels=self.channels, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=self.channels, out_channels=self.channels, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=self.channels, out_channels=self.channels, kernel_size=3, padding=1)

    def forward(self, frame):
        f0 = F.relu(self.conv1(frame))
        f1 = F.relu(self.conv2(f0))
        f2 = F.relu(self.conv3(f1))
        out = self.conv4(f2)
        return out

class blockNL(torch.nn.Module):
    def __init__(self, channels, fs):
        super(blockNL, self).__init__()
        self.channels = channels
        self.fs = fs
        self.ExtractFea = ExtractFea(channels=self.channels)
        self.softmax = nn.Softmax(dim=-1)

        self.t = nn.Conv2d(in_channels=self.channels, out_channels=self.channels, kernel_size=1, stride=1, bias=False)
        self.p = nn.Conv2d(in_channels=self.channels, out_channels=self.channels, kernel_size=1, stride=1, bias=False)
        self.g = nn.Conv2d(in_channels=self.channels, out_channels=self.channels, kernel_size=1, stride=1, bias=False)
        self.w = nn.Conv2d(in_channels=self.channels, out_channels=self.channels, kernel_size=1, stride=1, bias=False)

    def forward(self, x):

        # x_fea = self.ExtractFea(x)

        x_fea = x

        theta = self.t(x_fea).permute(0, 2, 3, 1)#.contiguous()#[b, c, h, w]#[b, h, w,c]
        theta = torch.unsqueeze(theta, dim=-2)  # [b, h, w, 1, c]
        # print(theta.size())

        phi = self.p(x_fea)#[b, c, h, w]
        b, c, h, w = phi.size()
        phi_patches = F.unfold(phi, self.fs, padding=self.fs // 2)#[b, c*fs*fs, hw]
        phi_patches = phi_patches.view(b, c, self.fs * self.fs, -1)#[b, c, fs*fs, hw]
        phi_patches = phi_patches.view(b, c, self.fs * self.fs, h, w)  #[b, c, fs*fs, h, w]
        phi_patches = phi_patches.permute(0, 3, 4, 1, 2)#.contiguous()#[b, h, w, c, fs*fs]
        # print(phi_patches.size())

        att = torch.matmul(theta, phi_patches)# [b, h, w, 1, fs*fs]
        att = self.softmax(att)# [b, h, w, 1, fs*fs]
        # print(att.size())

        g = self.g(x_fea) #[b, 3, h, w]
        g_patches = F.unfold(g, self.fs, padding=self.fs // 2)#[b, 3*fs*fs, hw]
        g_patches = g_patches.view(b, 3, self.fs * self.fs, -1)#[b, 3, fs*fs, hw]
        g_patches = g_patches.view(b, 3, self.fs * self.fs, h, w)#[b, 3, fs*fs, h, w]
        g_patches = g_patches.permute(0, 3, 4, 2, 1)#.contiguous()#[b, h, w, fs*fs, 3]
        # print(g_patches.size())

        out_x = torch.matmul(att, g_patches)  # [1, h, w, 1, 3]
        out_x = torch.squeeze(out_x, dim=-2)# [1, h, w, 3]
        out_x = out_x.permute(0, 3, 1, 2)#.contiguous()
        # print(alignedframe.size())
        return self.w(out_x) + x