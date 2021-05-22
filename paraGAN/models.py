import torch
import torch.nn as nn
import numpy as np
import math
import copy
import torch.nn.functional as F
from paraGAN.imresize import imresize, imresize_to_shape
import paraGAN.ACBlock as ACBlock
from paraGAN.espcn import ESPCN
import paraGAN.myglobal as myglobal
import paraGAN.MyACBlock as MyACBlock

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('Norm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def get_activation(opt):
    activations = {"lrelu": nn.LeakyReLU(opt.lrelu_alpha, inplace=True),
                   "elu": nn.ELU(alpha=1.0, inplace=True),
                   "prelu": nn.PReLU(num_parameters=1, init=0.25),
                   "selu": nn.SELU(inplace=True)
                   }
    return activations[opt.activation]


def upsample(x, size):
    x_up =  torch.nn.functional.interpolate(x, size=size, mode='bicubic', align_corners=True)
    return x_up


class ConvBlock(nn.Sequential):
    def __init__(self, in_channel, out_channel, ker_size, padd, opt,dialation=1,generator=False):
        super(ConvBlock,self).__init__()
        self.add_module('conv', nn.Conv2d(in_channel, out_channel, kernel_size=ker_size, stride=1,groups=opt.groups, padding=padd,dilation=dialation))
        if generator and opt.batch_norm:
            self.add_module('norm', nn.BatchNorm2d(out_channel))
        self.add_module(opt.activation, get_activation(opt))


class ConvEspcn(nn.Sequential):
    def __init__(self, in_channel, out_channel,opt, generator = False):
        super(ConvEspcn,self).__init__()
        self.add_module('deconv',ESPCN(in_channel,out_channel,2))
        if generator and opt.batch_norm:
            self.add_module('norm', nn.BatchNorm2d(out_channel))
        self.add_module(opt.activation, get_activation(opt))


class ConvACBlock(nn.Sequential):
    def __init__(self,in_channel, out_channel, ker_size, padd, opt,deploy=False,dilation = 1, generator=False):
        super(ConvACBlock,self).__init__()
        self.add_module('acbconv',ACBlock(in_channel,out_channel,kernel_size=ker_size,stride=1,padding=padd,deploy=deploy,dilation=dilation))
        if generator and opt.batch_norm:
            self.add_module('norm', nn.BatchNorm2d(out_channel))
        self.add_module(opt.activation, get_activation(opt))


class Discriminator(nn.Module):
    def __init__(self, opt):
        super(Discriminator, self).__init__()

        self.opt = opt
        opt.groups=1
        N = opt.nfc
        self.head = ConvBlock(opt.nc_im, N, 3, padd=opt.padd_size, opt=opt)
        self.body = nn.Sequential()
        for i in range(myglobal.currentdepth):
            dia = 1
            if myglobal.currentdepth ==1:
                dia = 2
            block = ConvBlock(N,N,3,padd= opt.padd_size,opt=opt,dialation=dia)
            self.body.add_module('block%d'%(i),block)
        # self.body = rbf.BasicRFB(N,N)
        # self.body1 = nn.Sequential()
        self.tail = nn.Conv2d(N, 1, kernel_size=3, padding=opt.padd_size)


    def forward(self,x):
        head = self.head(x)
        body = self.body(head)
        out = self.tail(body)
        return out


def calculateLayerList(x):
    a = []
    num=[3,2,1]
    index =0
    while x>0:
        t = math.ceil(x/num[index])
        for i in range(t):
            a.append(num[index])
        index = index+1
        x = x-t
    return a


class ChannelAttention(nn.Module):
    def __init__(self, in_planes,ratio = 16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class GrowingGenerator(nn.Module):
    def __init__(self, opt):
        super(GrowingGenerator, self).__init__()

        self.opt = opt
        N = int(opt.nfc)

        self._pad = nn.ZeroPad2d(1)
        self._pad_block = nn.ZeroPad2d(opt.num_layer-1) if opt.train_mode == "generation" \
                                                        else nn.ZeroPad2d(opt.num_layer)

        self.head = ConvBlock(opt.nc_im, N, opt.ker_size, opt.padd_size, opt, generator=True)


        # self.tmp = torch.nn.ModuleList([])
        # _tmp = nn.Sequential()
        # for i in range(5):
        #     block = ConvBlock(N, N, opt.ker_size, opt.padd_size, opt, generator=True)
        #     _tmp.add_module('block%d' % (i), block)
        # self.tmp.append(_tmp)

        self.body = torch.nn.ModuleList([])
        _first_stage = nn.Sequential()
        for i in range(opt.num_layer):
            block = ConvBlock(N, N, opt.ker_size, opt.padd_size, opt, generator=True)
            _first_stage.add_module('block%d' % (i), block)
        self.body.append(_first_stage)
        # self.body.append(GeneratorBlock(N,N,opt))
        # self.body.append(rbf.BasicRFB(N,N))
        self.tail = nn.Sequential(
            nn.Conv2d(N, opt.nc_im, kernel_size=3, padding=opt.padd_size),
            nn.Tanh())
        self.others = nn.Sequential()
        self.others.add_module('deblock', ConvEspcn(N, N, opt, True))

        # self.co = torch.nn.ModuleList([])
        # _co_stage = nn.Sequential()
        # for i in range(1):
        #     block = ChannelAttention(N)
        #     _co_stage.add_module('block%d' % (i), block)
        # self.co.append(_co_stage)

    def init_next_stage(self,current_scalenum):
        # if current_scalenum == myglobal.WhenEspcn:
        #     self.body.append(copy.deepcopy(self.others))
        # else:
        #     self.body.append(copy.deepcopy(self.body[-1]))
        self.body.append(copy.deepcopy(nn.Sequential(*list(self.body[-1].children())[:myglobal.num_layer_list[current_scalenum]])))
        # self.body.append(copy.deepcopy(self.body[-1]))
        # self.co.append(copy.deepcopy(self.co[-1]))

    def forward(self, noise, reals, noise_amp):
        real_shapes = [r.shape for r in reals]
        x = self.head(self._pad(noise[0]))

        if self.opt.train_mode == "generation":
            x = upsample(x, size=[x.shape[2] + 2, x.shape[3] + 2])
        x = self._pad_block(x)
        x_prev_out = self.body[0](x)
        for idx, block in enumerate(self.body[1:], 1):
            if idx <myglobal.WhenEspcn:
                x_prev_out_1 = upsample(x_prev_out, size=[real_shapes[idx][2], real_shapes[idx][3]])
                x_prev_out_2 = upsample(x_prev_out, size=[real_shapes[idx][2] + (myglobal.num_layer_list[idx]) * 2,
                                                          real_shapes[idx][3] + (myglobal.num_layer_list[idx]) * 2])
                x_prev = block(x_prev_out_2 + noise[idx] * noise_amp[idx])
                # x_prev = block(x_prev_out_2)
                x_prev_out = x_prev + x_prev_out_1
                # x_prev_out = self.co[idx](x_prev_out)*x_prev_out

            else:
                x_prev_out_1 = upsample(x_prev_out,size=[real_shapes[idx-1][2]+6, real_shapes[idx-1][3]+6])

                x_prev_out_2 = block(x_prev_out_1+noise[idx]*noise_amp[idx])
                if idx>=myglobal.WhenEspcn:
                    if int(real_shapes[idx][2]-1) in myglobal.updownload:
                        x_prev_out_3 = upsample(x_prev_out,size=[real_shapes[idx][2]-1,real_shapes[idx][3]-1])
                    else:
                        x_prev_out_3 = upsample(x_prev_out, size=[real_shapes[idx][2], real_shapes[idx][3]])
                    # x_prev_out = x_prev_out_2 + x_prev_out_3 * myglobal.priorStageFactor
                    x_prev_out = x_prev_out_2+x_prev_out_3
                else:
                    x_prev_out = x_prev_out_2
                if x_prev_out.shape[-1] in myglobal.updownload:
                    x_prev_out = upsample(x_prev_out,[x_prev_out.shape[2]+1,x_prev_out.shape[3]+1])

            # x_prev_out_2 = upsample(x_prev_out, size=[real_shapes[idx][2], real_shapes[idx][3]])
        out = self.tail(self._pad(x_prev_out))
        # out = self.tail(x_prev_out)
        return out
