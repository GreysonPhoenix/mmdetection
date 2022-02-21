import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import cv2
from torch.autograd import Variable
from PIL import Image


def init_weights(net, init_type='normal', gain=0.02):
    from torch.nn import init
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal(m.weight.data, 1.0, gain)
            init.constant(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)
def get_pad(in_,  ksize, stride, atrous=1):
    out_ = np.ceil(float(in_)/stride)
    return int(((out_ - 1) * stride + atrous*(ksize-1) + 1 - in_)/2)


class GatedConv2dWithActivation(torch.nn.Module):
    """
    Gated Convlution layer with activation (default activation:LeakyReLU)
    Params: same as conv2d
    Input: The feature from last layer "I"
    Output:\phi(f(I))*\sigmoid(g(I))
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True,batch_norm=True, activation=torch.nn.LeakyReLU(0.2, inplace=True)):
        super(GatedConv2dWithActivation, self).__init__()
        self.batch_norm = batch_norm
        self.activation = activation
        self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.mask_conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.batch_norm2d = torch.nn.BatchNorm2d(out_channels)
        self.sigmoid = torch.nn.Sigmoid()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
    def gated(self, mask):
        #return torch.clamp(mask, -1, 1)
        return self.sigmoid(mask)
    def forward(self, input):
        x = self.conv2d(input)
        mask = self.mask_conv2d(input)
        if self.activation is not None:
            x = self.activation(x) * self.gated(mask)
        else:
            x = x * self.gated(mask)
        if self.batch_norm:
            return self.batch_norm2d(x)
        else:
            return x


class DOAM(nn.Module):
    def __init__(self):

        super(DOAM, self).__init__()
        self.gamma = nn.Parameter(torch.zeros(1))

        self.sigmoid = torch.nn.Sequential()
        self.sigmoid.add_module('Sigmoid', nn.Sigmoid())

        self.conv2d_1_1 = torch.nn.Sequential()
        self.conv2d_1_1.add_module('conv2d_1_1', nn.Conv2d(8, 1, kernel_size=3, padding=1))

        self.AdaptiveAverPool_5 = torch.nn.Sequential()
        self.AdaptiveAverPool_5.add_module('AdaptiveAverPool_5', nn.AdaptiveAvgPool2d((60, 60)))

        self.AdaptiveAverPool_10 = torch.nn.Sequential()
        self.AdaptiveAverPool_10.add_module('AdaptiveAverPool_10', nn.AdaptiveAvgPool2d((30, 30)))

        self.AdaptiveAverPool_15 = torch.nn.Sequential()
        self.AdaptiveAverPool_15.add_module('AdaptiveAverPool_15', nn.AdaptiveAvgPool2d((20, 20)))

        #rgb convolution
        self.conv2d_1_rgb_attention = torch.nn.Sequential()
        self.conv2d_1_rgb_attention.add_module('conv2d_1_rgb_attention', nn.Conv2d(3, 8, kernel_size=3, padding=1))
        self.conv2d_2_rgb_attention = torch.nn.Sequential()
        self.conv2d_2_rgb_attention.add_module('conv2d_2_rgb_attention', nn.Conv2d(8, 16, kernel_size=3, padding=1))
        self.conv2d_3_rgb_attention = torch.nn.Sequential()
        self.conv2d_3_rgb_attention.add_module('conv2d_3_rgb_attention', nn.Conv2d(16, 32, kernel_size=3, padding=1))
        self.conv2d_4_rgb_attention = torch.nn.Sequential()
        self.conv2d_4_rgb_attention.add_module('conv2d_4_rgb_attention', nn.Conv2d(32, 16, kernel_size=3, padding=1))
        self.conv2d_5_rgb_attention = torch.nn.Sequential()
        self.conv2d_5_rgb_attention.add_module('conv2d_5_rgb_attention', nn.Conv2d(16, 8, kernel_size=3, padding=1))

        self.GatedConv2dWithActivation = GatedConv2dWithActivation(in_channels=(8 * 3),
                                                                   out_channels=8, kernel_size=3,
                                                                   stride=1, padding=1, activation=None)

        self.conv2d_1_rgb_red_concat_5 = torch.nn.Sequential()
        self.conv2d_1_rgb_red_concat_5.add_module('conv2d_1_rgb_red_concat_5',
                                                  nn.Conv2d(16, 8, kernel_size=3, padding=1))
        self.conv2d_1_rgb_red_concat_10 = torch.nn.Sequential()
        self.conv2d_1_rgb_red_concat_10.add_module('conv2d_1_rgb_red_concat_10',
                                                   nn.Conv2d(16, 8, kernel_size=3, padding=1))
        self.conv2d_1_rgb_red_concat_15 = torch.nn.Sequential()
        self.conv2d_1_rgb_red_concat_15.add_module('conv2d_1_rgb_red_concat_15',
                                                   nn.Conv2d(16, 8, kernel_size=3, padding=1))

    def RIA(self, x):
        x_shape = x.shape
        og_x = x
        refine_30 = []
        x_pooled_upsample_5 = torch.zeros((x.shape[0], 8, x.shape[2], x.shape[3])).cuda()
        x_pooled_upsample_10 = torch.zeros((x.shape[0], 8, x.shape[2], x.shape[3])).cuda()
        x_pooled_upsample_15 = torch.zeros((x.shape[0], 8, x.shape[2], x.shape[3])).cuda()
        # refined_x = torch.zeros((x.shape))
        # refined_x = torch.zeros((24,100,30*30))
        x_pooled_5 = self.AdaptiveAverPool_5(x)
        x_pooled_10 = self.AdaptiveAverPool_10(x)
        x_pooled_15 = self.AdaptiveAverPool_15(x)
        for i in range(60):
            for j in range(60):
                x_pooled_upsample_5[:, :, i * 5:(i + 1) * 5, j * 5:(j + 1) * 5] = x_pooled_5[:, :, i, j].unsqueeze(
                    -1).unsqueeze(-1)

        for i in range(30):
            for j in range(30):
                x_pooled_upsample_10[:, :, i * 10:(i + 1) * 10, j * 10:(j + 1) * 10] = x_pooled_10[:, :, i,
                                                                                       j].unsqueeze(-1).unsqueeze(-1)

        for i in range(20):
            for j in range(20):
                x_pooled_upsample_15[:, :, i * 15:(i + 1) * 15, j * 15:(j + 1) * 15] = x_pooled_15[:, :, i,
                                                                                       j].unsqueeze(-1).unsqueeze(-1)

        x_concat_5 = torch.cat((x, x_pooled_upsample_5), 1)
        x_concat_10 = torch.cat((x, x_pooled_upsample_10), 1)
        x_concat_15 = torch.cat((x, x_pooled_upsample_15), 1)

        x_concat_5_out = self.conv2d_1_rgb_red_concat_5(x_concat_5)
        x_concat_10_out = self.conv2d_1_rgb_red_concat_10(x_concat_10)
        x_concat_15_out = self.conv2d_1_rgb_red_concat_15(x_concat_15)

        x_gated_conv_input = torch.cat((x_concat_5_out, x_concat_10_out), 1)
        x_gated_conv_input = torch.cat((x_gated_conv_input, x_concat_15_out), 1)

        x_gated_conv_output = self.GatedConv2dWithActivation(x_gated_conv_input)
        return x_gated_conv_output

    def MaxMinNormalization(self, x):
        """[0,1] normaliaztion"""
        x = (x - x.min()) / (x.max() - x.min()) *255
        #x = x.astype(int)
        return x

    def forward(self, im):
        rgb_red = im

        rgb_conved = self.conv2d_1_rgb_attention(rgb_red)
        rgb_conved = self.conv2d_2_rgb_attention(rgb_conved)
        rgb_conved = self.conv2d_3_rgb_attention(rgb_conved)
        rgb_conved = self.conv2d_4_rgb_attention(rgb_conved)
        rgb_conved = self.conv2d_5_rgb_attention(rgb_conved)

        rgb_conved = self.RIA(rgb_conved)

        rgb_red_conved = rgb_conved
        rgb_red_conved = self.conv2d_1_1(rgb_red_conved)

        sigmoid_output = self.sigmoid(rgb_red_conved)

        rgb_red = self.gamma * (sigmoid_output * rgb_red) + (1 - self.gamma)*rgb_red

        # output = self.conv2d(rgb_red)

        # edge_detect = edge_detect.float().cpu().squeeze().detach().numpy()
        # edge_detect = cv2.cvtColor(edge_detect, cv2.COLOR_BGR2RGB)
        # img = Image.fromarray(edge_detect)# .convert('L')
        # img.save('edge.jpg')

        # softmax_output = softmax_output*255
        # softmax_output_im = softmax_output.float().cpu().squeeze().detach().numpy()
        # im = Image.fromarray(softmax_output_im).convert('L')
        # im.save('edge_sigmoid_output_im.jpg')
        # rgb_im = og_im*sigmoid_output
        # rgb_im = rgb_im.float().cpu().squeeze().detach().numpy()
        # sigmoid_output = sigmoid_output*255
        # sigmoid_output_im = sigmoid_output.float().cpu().squeeze().detach().numpy()
        # im = Image.fromarray(sigmoid_output_im).convert('L')
        # im.save('edge_sigmoid_output_im.jpg')
        # print(rgb_red.shape, sigmoid_output.shape)

        return rgb_red
