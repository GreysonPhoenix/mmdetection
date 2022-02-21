import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule, auto_fp16

from ..builder import NECKS

class Boundary_Aggregation(nn.Module):
    def __init__(self, in_channels):
        super(Boundary_Aggregation, self).__init__()
        self.conv = nn.Conv2d(in_channels * 5, in_channels, 1)

    def forward(self, x_batch: torch.tensor):
        in_channels, height, width = x_batch.size()[1:4]
        x_clk_rot = torch.rot90(x_batch, -1, [2, 3])
        x1 = self.up_to_bottom(x_batch, in_channels, height)
        x2 = self.bottom_to_up(x_batch, in_channels, height)
        x3 = self.left_to_right(x_clk_rot, in_channels, width)
        x4 = self.right_to_left(x_clk_rot, in_channels, width)
        x_con = torch.cat((x_batch, x1, x2, x3, x4), 1)
        x_merge = self.conv(x_con)
        return x_merge

    def left_to_right(self, x_clk_rot: torch.tensor, in_channels: int,
                      height: int):
        x = torch.clone(x_clk_rot)
        x = self.up_to_bottom(x, in_channels, height)
        x = torch.rot90(x, 1, [2, 3])
        return x

    def right_to_left(self, x_clk_rot: torch.tensor, in_channels: int,
                      height: int):
        x = torch.clone(x_clk_rot)
        x = self.bottom_to_up(x, in_channels, height)
        x = torch.rot90(x, 1, [2, 3])
        return x

    def bottom_to_up(self, x_raw: torch.tensor, in_channels: int, height: int):
        x = torch.clone(x_raw)
        for i in range(height - 1, -1, -1):
            x[:, :, i] = torch.max(x[:, :, i:], 2, True)[0].squeeze(2)
        return x

    def up_to_bottom(self, x_raw: torch.tensor, in_channels: int, height: int):
        x = torch.clone(x_raw)
        for i in range(height):
            x[:, :, i] = torch.max(x[:, :, :i + 1], 2, True)[0].squeeze(2)
        return x

@NECKS.register_module()
class DenseFeaturePyramidNetwork(BaseModule):
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 start_level=0,
                 end_level=-1,
                 add_extra_convs=False,
                 extra_convs_on_inputs=True,
                 relu_before_extra_convs=False,
                 no_norm_on_lateral=False,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None,
                 upsample_cfg=dict(mode='nearest'),
                 init_cfg=dict(
                     type='Xavier', layer='Conv2d', distribution='uniform')):
        super(DenseFeaturePyramidNetwork, self).__init__(init_cfg)
        C3_size, C4_size, C5_size, C6_size = in_channels
        feature_size = out_channels
      
        # upsample C5 to get P5 from the FPN paper
        self.P5_1 = nn.Conv2d(C5_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P5_1_dual = nn.Conv2d(C5_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P5_upsampled_1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.P5_upsampled_2 = nn.Upsample(scale_factor=2, mode='nearest')
        # self.P5_upsampled_3 = nn.Upsample(scale_factor=2, mode='nearest')
        self.P5_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # add P5 elementwise to C4
        self.P4_1 = nn.Conv2d(C4_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P4_1_dual = nn.Conv2d(C4_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P4_upsampled_1 = nn.Upsample(scale_factor=2, mode='nearest')
        # self.P4_upsampled_2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.P4_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # add P4 elementwise to C3
        self.P3_1 = nn.Conv2d(C3_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P3_1_dual = nn.Conv2d(C3_size, feature_size, kernel_size=1, stride=1, padding=0)
        #         self.P3_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P3_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # add P3 elementwise to C2
        # self.P2_1 = nn.Conv2d(C2_size, feature_size, kernel_size=1, stride=1, padding=0)
        # self.P2_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # "P6 is obtained via a 3x3 stride-2 conv on C5"
        self.P6 = nn.Conv2d(C5_size, feature_size, kernel_size=3, stride=2, padding=1)

        # "P7 is computed by applying ReLU followed by a 3x3 stride-2 conv on P6"
        self.P7_1 = nn.ReLU()
        self.P7_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=2, padding=1)

        # self.conv_to_1 = nn.Conv2d(feature_size, 16, kernel_size=1)
        # self.conv_resume = nn.Conv2d(16, feature_size, kernel_size=1)

        self.corner_proc_C3 = Boundary_Aggregation(256)
        self.corner_proc_C4 = Boundary_Aggregation(512)
        self.corner_proc_C5 = Boundary_Aggregation(1024)

        self.conv_p3_to_p4 = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1)
        self.conv_p4_to_p5 = nn.Conv2d(256, 256, kernel_size=2, stride=2, padding=0)
        self.conv_p3_to_p5 = nn.Conv2d(256, 256, kernel_size=2, stride=2, padding=0)

        # self.sig_3 = nn.Sigmoid()
        # self.sig_4 = nn.Sigmoid()
        # self.sig_5 = nn.Sigmoid()
        self.gamma_3 = nn.Parameter(torch.zeros(1))
        self.gamma_4 = nn.Parameter(torch.zeros(1))
        self.gamma_5 = nn.Parameter(torch.zeros(1))

        # self.conv_down_3 = nn.Conv2d(256, 1, kernel_size=1, stride=1, padding=0)
        # self.conv_down_4 = nn.Conv2d(256, 1, kernel_size=1, stride=1, padding=0)
        # self.conv_down_5 = nn.Conv2d(256, 1, kernel_size=1, stride=1, padding=0)

    def forward(self, inputs):
        global count
        C3, C4, C5, C6 = inputs  # 150，75,38,19

        C3_BA = self.corner_proc_C3(C3)
        C4_BA = self.corner_proc_C4(C4)
        C5_BA = self.corner_proc_C5(C5)

        P5_x = self.P5_1(C5)
        P5_upsampled_x_1 = self.P5_upsampled_1(P5_x)  # 38
        P5_upsampled_x_2 = self.P5_upsampled_2(P5_upsampled_x_1)  # 75
        # P5_upsampled_x_3 = self.P5_upsampled_3(P5_upsampled_x_2)#150
        P5_x = self.P5_2(P5_x)

        P4_x = self.P4_1(C4)
        P4_x = P4_x + P5_upsampled_x_1  # 38
        P4_upsampled_x_1 = self.P4_upsampled_1(P4_x)  # 75
        # P4_upsampled_x_2 = self.P4_upsampled_2(P4_upsampled_x_1)#150
        P4_x = self.P4_2(P4_x)

        P3_x = self.P3_1(C3)
        P3_x = P3_x + P4_upsampled_x_1 + P5_upsampled_x_2  # 75
        # P3_upsampled_x_1 = self.P3_upsampled(P3_x)  # 150
        P3_x = self.P3_2(P3_x)

        P3_dual = self.P3_1_dual(C3_BA)
        P3_downsample_x_1 = self.conv_p3_to_p4(P3_dual)
        P3_downsample_x_2 = self.conv_p3_to_p5(P3_downsample_x_1)

        P4_dual = self.P4_1_dual(C4_BA)
        P4_dual = P4_dual + P3_downsample_x_1
        P4_downsample_x_1 = self.conv_p4_to_p5(P4_dual)

        P5_dual = self.P5_1_dual(C5_BA)
        P5_dual = P5_dual + P4_downsample_x_1 + P3_downsample_x_2

        # P2_x = self.P2_1(C2)
        # P2_x = P2_x + P3_upsampled_x_1 +  P4_upsampled_x_2 + P5_upsampled_x_3# 75
        # P2_x = self.P2_2(P2_x)

        P6_x = self.P6(C5)  # 10

        P7_x = self.P7_1(P6_x)
        P7_x = self.P7_2(P7_x)  # 5

        # print(P3_dual.shape)
        # img = P3_dual[0][0].cpu().detach().numpy()
        # for i in range(1,P3_dual.shape[1]):
        #    img = img + P3_dual[0][i].cpu().detach().numpy()
        # print('img_shape',img.shape)
        # cv2.imwrite('test/test'+str(count)+'.jpg',img * 255)
        # time.sleep(10)
        O3_x = self.gamma_3 * P3_dual + (1 - self.gamma_3) * P3_x
        O4_x = self.gamma_4 * P4_dual + (1 - self.gamma_4) * P4_x
        O5_x = self.gamma_5 * P5_dual + (1 - self.gamma_5) * P5_x

        return [O3_x, O4_x, O5_x, P6_x, P7_x]