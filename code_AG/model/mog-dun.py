from model import common
import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
import random
from model import blockNL


def make_model(args, parent=False):
    return MoG_DUN(args)


class Encoding_Block(nn.Module):
    def __init__(self, c_in):
        super(Encoding_Block, self).__init__()

        self.down = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=3 // 2)

        self.act = nn.ReLU()
        body = [nn.Conv2d(in_channels=c_in, out_channels=64, kernel_size=3, padding=3 // 2),nn.ReLU(),
                common.ResBlock(common.default_conv, 64, 3),common.ResBlock(common.default_conv, 64, 3),
                nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=3 // 2)]
        self.body = nn.Sequential(*body)

    def forward(self, input):

        f_e = self.body(input)
        down = self.act(self.down(f_e))
        return f_e, down


class Encoding_Block_End(nn.Module):
    def __init__(self, c_in):
        super(Encoding_Block_End, self).__init__()

        self.down = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=3 // 2)
        self.act = nn.ReLU()
        head = [nn.Conv2d(in_channels=c_in, out_channels=64, kernel_size=3, padding=3 // 2), nn.ReLU()]
        body = [
                common.ResBlock(common.default_conv, 64, 3), common.ResBlock(common.default_conv, 64, 3),
                common.ResBlock(common.default_conv, 64, 3), common.ResBlock(common.default_conv, 64, 3),
                common.ResBlock(common.default_conv, 64, 3), common.ResBlock(common.default_conv, 64, 3),

                common.ResBlock(common.default_conv, 64, 3), common.ResBlock(common.default_conv, 64, 3),
                common.ResBlock(common.default_conv, 64, 3), common.ResBlock(common.default_conv, 64, 3),
                common.ResBlock(common.default_conv, 64, 3), common.ResBlock(common.default_conv, 64, 3),

                common.ResBlock(common.default_conv, 64, 3), common.ResBlock(common.default_conv, 64, 3),
                common.ResBlock(common.default_conv, 64, 3), common.ResBlock(common.default_conv, 64, 3),
                common.ResBlock(common.default_conv, 64, 3), common.ResBlock(common.default_conv, 64, 3),
                ]
        tail = [nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=3 // 2)]
        self.head = nn.Sequential(*head)
        self.body = nn.Sequential(*body)
        self.tail = nn.Sequential(*tail)
    def forward(self, input):
        out = self.head(input)
        f_e = self.body(out) + out
        f_e = self.tail(f_e)
        return f_e


class Decoding_Block(nn.Module):
    def __init__(self, c_in ):
        super(Decoding_Block, self).__init__()
        #self.up = torch.nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1)
        self.up = nn.ConvTranspose2d(128, 128, kernel_size=3, stride=2, padding=1)
        self.act = nn.ReLU()
        body = [nn.Conv2d(in_channels=c_in, out_channels=64, kernel_size=3, padding=3 // 2), nn.ReLU(),
                common.ResBlock(common.default_conv, 64, 3), common.ResBlock(common.default_conv, 64, 3),
                nn.Conv2d(in_channels=64, out_channels=128, kernel_size=1, padding=1 // 2) ]
        self.body = nn.Sequential(*body)


    def forward(self, input, map):

        up = self.act(self.up(input,output_size=[input.shape[0], input.shape[1], map.shape[2], map.shape[3]]))
        out = torch.cat((up, map), 1)
        out = self.body(out)

        return out


class Decoding_Block_End(nn.Module):
    def __init__(self, c_in):
        super(Decoding_Block_End, self).__init__()
        # self.up = torch.nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1)
        self.up = nn.ConvTranspose2d(128, 128, kernel_size=3, stride=2, padding=1)
        self.act = nn.ReLU()
        body = [nn.Conv2d(in_channels=c_in, out_channels=64, kernel_size=3, padding=3 // 2), nn.ReLU(),
                common.ResBlock(common.default_conv, 64, 3), common.ResBlock(common.default_conv, 64, 3),

                ]
        self.body = nn.Sequential(*body)



    def forward(self, input, map):
        up = self.act(self.up(input, output_size=[input.shape[0], input.shape[1], map.shape[2], map.shape[3]]))
        out = torch.cat((up, map), 1)
        out = self.body(out)
        return out


class Conv_up(nn.Module):
    def __init__(self, c_in,up_factor):
        super(Conv_up, self).__init__()

        body = [nn.Conv2d(in_channels=c_in, out_channels=64, kernel_size=3, padding=3 // 2), nn.ReLU(),
                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=3 // 2), nn.ReLU(),
                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=3 // 2), nn.ReLU(),
                ]
        self.body = nn.Sequential(*body)
        conv=common.default_conv
        ## x3 00
        ## x2 11
        if up_factor==2:
            modules_tail = [
                nn.ConvTranspose2d(64,64,kernel_size=3,stride=up_factor,padding=1,output_padding=1),
                conv(64, c_in, 3)]
        elif up_factor==3:
            modules_tail = [
                nn.ConvTranspose2d(64,64,kernel_size=3,stride=up_factor,padding=0,output_padding=0),
                conv(64, c_in, 3)]

        elif up_factor==4:
            modules_tail = [
                nn.ConvTranspose2d(64,64,kernel_size=3,stride=2,padding=1,output_padding=1),
                nn.ConvTranspose2d(64,64,kernel_size=3,stride=2,padding=1,output_padding=1),
                conv(64, c_in, 3)]
        self.tail = nn.Sequential(*modules_tail)    


    def forward(self, input):
        
        out = self.body(input)
        out = self.tail(out)
        return out



class Conv_down(nn.Module):
    def __init__(self, c_in,up_factor):
        super(Conv_down, self).__init__()

        body = [nn.Conv2d(in_channels=c_in, out_channels=64, kernel_size=3, padding=3 // 2), nn.ReLU(),
                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=3 // 2), nn.ReLU(),
                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=3 // 2), nn.ReLU(),
                ]
        self.body = nn.Sequential(*body)
        conv=common.default_conv
        if up_factor==4:
            modules_tail = [
                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1,stride=2),
                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1,stride=2),
                conv(64, c_in, 3)]
        elif up_factor==3:
            modules_tail = [
                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1,stride=up_factor),
                conv(64, c_in, 3)]
        elif up_factor==2:
            modules_tail = [
                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1,stride=up_factor),
                conv(64, c_in, 3)]                
        self.tail = nn.Sequential(*modules_tail)    



    def forward(self, input):
        
        out = self.body(input)
        out = self.tail(out)
        return out

class MoG_DUN(nn.Module):
    def __init__(self,args):
        super(MoG_DUN, self).__init__()

        self.channel0 = args.n_colors
        self.up_factor = args.scale[0]
        self.patch_size = args.patch_size
        self.batch_size = int(args.batch_size/args.n_GPUs)



        self.Encoding_block1 = Encoding_Block(64)
        self.Encoding_block2 = Encoding_Block(64)
        self.Encoding_block3 = Encoding_Block(64)
        self.Encoding_block4 = Encoding_Block(64)

        self.Encoding_block_end = Encoding_Block_End(64)

        self.Decoding_block1 = Decoding_Block(256)
        self.Decoding_block2 = Decoding_Block(256)
        self.Decoding_block3 = Decoding_Block(256)
        self.Decoding_block4 = Decoding_Block(256)

        self.feature_decoding_end = Decoding_Block_End(256)

        self.act =nn.ReLU()

        self.construction = nn.Conv2d(64, 3, 3, padding=1)

        G0 = 64
        kSize = 3
        T = 4
        self.Fe_e = nn.ModuleList([nn.Sequential(*[
            nn.Conv2d(3, G0, kSize, padding=(kSize - 1) // 2, stride=1),
            nn.Conv2d(G0, G0, kSize, padding=(kSize - 1) // 2, stride=1)
        ]) for _ in range(T)])



        self.RNNF = nn.ModuleList([nn.Sequential(*[
            nn.Conv2d((i+2) * G0, G0, 1, padding=0, stride=1),
            nn.Conv2d(G0, G0, kSize, padding=(kSize - 1) // 2, stride=1),
            self.act,
            nn.Conv2d(64, 3, 3, padding=1)

        ]) for i in range(T)])


        self.Fe_f = nn.ModuleList([nn.Sequential(*[nn.Conv2d((2*i+3) * G0, G0, 1, padding=0, stride=1)]) for i in range(T-1)])


        self.u = nn.ParameterList(
            [nn.Parameter(torch.tensor(0.5)) for _ in range(T)])
        self.eta = nn.ParameterList(
            [nn.Parameter(torch.tensor(0.5)) for _ in range(T)])
        self.gama = nn.ParameterList(
            [nn.Parameter(torch.tensor(0.5)) for _ in range(T)])
        self.delta = nn.ParameterList(
            [nn.Parameter(torch.tensor(0.1)) for _ in range(T)])
        self.gama1 = nn.ParameterList(
            [nn.Parameter(torch.tensor(0.5)) for _ in range(T)])
        self.delta1 = nn.ParameterList(
            [nn.Parameter(torch.tensor(0.1)) for _ in range(T)])
        self.u1 = nn.ParameterList(
            [nn.Parameter(torch.tensor(0.5)) for _ in range(T)])
        
       

        self.conv_up = Conv_up(3,self.up_factor)
        self.conv_down = Conv_down(3,self.up_factor)
       
        self.NLBlock = nn.ModuleList([blockNL.blockNL(3, 15) for _ in range(4)])
        




    def forward(self, y): # [batch_size ,3 ,7 ,270 ,480] ;
        Ay = torch.nn.functional.interpolate(y, scale_factor=self.up_factor, mode='bilinear',align_corners=False)
        x = Ay
        fea_list = []
        V_list = []
        outs = []
        for i in range(len(self.Fe_e)):
            fea = self.Fe_e[i](x)
            fea_list.append(fea)
            if i!=0:
                fea = self.Fe_f[i-1](torch.cat(fea_list, 1))
            encode0, down0 = self.Encoding_block1(fea)
            encode1, down1 = self.Encoding_block2(down0)
            encode2, down2 = self.Encoding_block3(down1)
            encode3, down3 = self.Encoding_block4(down2)

            media_end = self.Encoding_block_end(down3)

            decode3 = self.Decoding_block1(media_end, encode3)
            decode2 = self.Decoding_block2(decode3, encode2)
            decode1 = self.Decoding_block3(decode2, encode1)
            decode0 = self.feature_decoding_end(decode1, encode0)

            fea_list.append(decode0)
            V_list.append(decode0)
            if i==0:
                decode0 = self.construction(self.act(decode0))
            else:
                decode0 = self.RNNF[i-1](torch.cat(V_list, 1))
            conv_out = x + decode0

            NL = self.NLBlock[i](x)
            e = NL-x
            e = e - self.delta1[i]*(self.u1[i]*self.conv_up(self.conv_down(x+e)-y)-self.gama1[i]*(x+e-NL))
        
            x = x - self.delta[i]*(self.conv_up(self.conv_down(x)-y+self.u[i]*self.conv_down(x+e)-y)+self.eta[i]*(x-conv_out)+self.gama[i]*(x+e-NL))

            outs.append(x)

        return x



    
    def load_state_dict(self, state_dict, strict=True):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') == -1:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))

