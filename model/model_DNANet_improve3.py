import torch
import torch.nn as nn


class VGG_CBAM_Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.ca = ChannelAttention(out_channels)
        self.sa = SpatialAttention()

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.ca(out) * out
        out = self.sa(out) * out
        out = self.relu(out)
        return out

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1   = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class Res_CBAM_block(nn.Module):
    def __init__(self, in_channels, out_channels, stride = 1):
        super(Res_CBAM_block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = stride, padding = 1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace = True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size = 3, padding = 1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if stride != 1 or out_channels != in_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size = 1, stride = stride),
                nn.BatchNorm2d(out_channels))
        else:
            self.shortcut = None

        self.ca = ChannelAttention(out_channels)
        self.sa = SpatialAttention()

    def forward(self, x):
        residual = x
        if self.shortcut is not None:
            residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.ca(out) * out
        out = self.sa(out) * out
        out += residual
        out = self.relu(out)
        return out

#搞一个即插即用的差异模块
class mqwNet(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(mqwNet, self).__init__()
        self.conv1_share = nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(inplace=True)
        self.conv2_share = nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.conv3_share = nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channel)
        self.conv4_share = nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1, bias=False)
        self.bn4 = nn.BatchNorm2d(out_channel)
        self.convchannel1 = nn.Conv2d(5, 2, kernel_size=1, stride=1, bias=False)
        self.convchannel2 = nn.Conv2d(5, 2, kernel_size=1, stride=1, bias=False)
        self.convchannel3 = nn.Conv2d(5, 2, kernel_size=1, stride=1, bias=False)
        self.convchannel4 = nn.Conv2d(5, 2, kernel_size=1, stride=1, bias=False)
        self.sigmoid1 = nn.Sigmoid()
        self.sigmoid2 = nn.Sigmoid()
        self.sigmoid3 = nn.Sigmoid()
        self.sigmoid4 = nn.Sigmoid()

    def forward(self, x1, x2, x3, x4, y1, y2, y3, y4):
        x1 = self.conv1_share(x1)
        # x1 = self.bn1(x1)
        # x1 = self.relu(x1)
        y1 = self.conv1_share(y1)
        x2 = self.conv1_share(x2)
        y2 = self.conv1_share(y2)
        x3 = self.conv1_share(x3)
        y3 = self.conv1_share(y3)
        x4 = self.conv1_share(x4)
        y4 = self.conv1_share(y4)

        F_tot1 = x1 + y1
        F_tot2 = x2 + y2
        F_tot3 = x3 + y3
        F_tot4 = x4 + y4

        F_sh1 = x1 * y1
        F_sh2 = x2 * y2
        F_sh3 = x3 * y3
        F_sh4 = x4 * y4

        F_diff1 = torch.abs(x1 - y1)
        F_diff2 = torch.abs(x2 - y2)
        F_diff3 = torch.abs(x3 - y3)
        F_diff4 = torch.abs(x4 - y4)

        F_rel1 = torch.cat([x1, F_tot1, F_sh1, F_diff1, y4], dim=1)
        F_rel2 = torch.cat([x2, F_tot2, F_sh2, F_diff2, y4], dim=1)
        F_rel3 = torch.cat([x3, F_tot3, F_sh3, F_diff3, y4], dim=1)
        F_rel4 = torch.cat([x4, F_tot4, F_sh4, F_diff4, y4], dim=1)

        x1_twochannel = self.convchannel1(F_rel1)
        x2_twochannel = self.convchannel2(F_rel2)
        x3_twochannel = self.convchannel3(F_rel3)
        x4_twochannel = self.convchannel4(F_rel4)

        x1_twochannel = self.sigmoid1(x1_twochannel)
        x2_twochannel = self.sigmoid2(x2_twochannel)
        x3_twochannel = self.sigmoid3(x3_twochannel)
        x4_twochannel = self.sigmoid4(x4_twochannel)

        w_I_1 = x1_twochannel[:,0:1,:,:]
        w_I255_1 = x1_twochannel[:, 0:2, :, :]
        w_I_2 = x2_twochannel[:,0:1,:,:]
        w_I255_2 = x2_twochannel[:, 0:2, :, :]
        w_I_3 = x3_twochannel[:,0:1,:,:]
        w_I255_3 = x3_twochannel[:, 0:2, :, :]
        w_I_4 = x4_twochannel[:,0:1,:,:]
        w_I255_4 = x4_twochannel[:, 0:2, :, :]

        xy1 = w_I_1 * x1 + w_I255_1 * y1
        xy2 = w_I_2 * x2 + w_I255_2 * y2
        xy3 = w_I_3 * x3 + w_I255_3 * y3
        xy4 = w_I_4 * x4 + w_I255_4 * y4

        return [xy1, xy2, xy3, xy4]

class mqwNet2(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(mqwNet2, self).__init__()
        self.conv1_share = nn.Conv2d(in_channel, 1, kernel_size=1, stride=1, bias=False)
        self.convdiff1 = nn.Conv2d(1, out_channel, kernel_size=1, stride=1, bias=False)
        self.convdiff2 = nn.Conv2d(1, out_channel, kernel_size=1, stride=1, bias=False)
        self.convdiff3 = nn.Conv2d(1, out_channel, kernel_size=1, stride=1, bias=False)
        self.convdiff4 = nn.Conv2d(1, out_channel, kernel_size=1, stride=1, bias=False)
        self.bndiff1 = nn.BatchNorm2d(out_channel)
        self.bndiff2 = nn.BatchNorm2d(out_channel)
        self.bndiff3 = nn.BatchNorm2d(out_channel)
        self.bndiff4 = nn.BatchNorm2d(out_channel)

        self.convmul1 = nn.Conv2d(1, out_channel, kernel_size=1, stride=1, bias=False)
        self.convmul2 = nn.Conv2d(1, out_channel, kernel_size=1, stride=1, bias=False)
        self.convmul3 = nn.Conv2d(1, out_channel, kernel_size=1, stride=1, bias=False)
        self.convmul4 = nn.Conv2d(1, out_channel, kernel_size=1, stride=1, bias=False)
        self.bnmul1 = nn.BatchNorm2d(out_channel)
        self.bnmul2 = nn.BatchNorm2d(out_channel)
        self.bnmul3 = nn.BatchNorm2d(out_channel)
        self.bnmul4 = nn.BatchNorm2d(out_channel)

        self.convadd1 = nn.Conv2d(1, out_channel, kernel_size=1, stride=1, bias=False)
        self.convadd2 = nn.Conv2d(1, out_channel, kernel_size=1, stride=1, bias=False)
        self.convadd3 = nn.Conv2d(1, out_channel, kernel_size=1, stride=1, bias=False)
        self.convadd4 = nn.Conv2d(1, out_channel, kernel_size=1, stride=1, bias=False)
        self.bnadd1 = nn.BatchNorm2d(out_channel)
        self.bnadd2 = nn.BatchNorm2d(out_channel)
        self.bnadd3 = nn.BatchNorm2d(out_channel)
        self.bnadd4 = nn.BatchNorm2d(out_channel)

        #self.mul1 = self.jump_layer(self.)

    def forward(self, x1, x2, x3, x4, y1, y2, y3, y4):
        x1_raw = x1
        # x1 = self.bn1(x1)
        # x1 = self.relu(x1)
        y1_raw = y1
        x2_raw = x2
        y2_raw = y2
        x3_raw = x3
        y3_raw = y3
        x4_raw = x4
        y4_raw = y4

        x1 = self.conv1_share(x1)
        # x1 = self.bn1(x1)
        # x1 = self.relu(x1)
        y1 = self.conv1_share(y1)
        x2 = self.conv1_share(x2)
        y2 = self.conv1_share(y2)
        x3 = self.conv1_share(x3)
        y3 = self.conv1_share(y3)
        x4 = self.conv1_share(x4)
        y4 = self.conv1_share(y4)
#
        diff = torch.abs(x1 - y1)
#
        F_diff1 = self.convdiff1(torch.abs(x1 - y1))
        F_diff1 = self.bndiff1(F_diff1)
        F_diff2 = self.convdiff2(torch.abs(x2 - y2))
        F_diff2 = self.bndiff2(F_diff2)
        F_diff3 = self.convdiff3(torch.abs(x3 - y3))
        F_diff3 = self.bndiff3(F_diff3)
        F_diff4 = self.convdiff4(torch.abs(x4 - y4))
        F_diff4 = self.bndiff4(F_diff4)
#
        share =    x1 * y1
#
        F_mul1 = self.convmul1(x1 * y1)
        F_mul1 = self.bnmul1(F_mul1)
        F_mul1_jump = self.jump_layer(F_mul1)
        F_mul2 = self.convmul2(x2 * y2)
        F_mul2 = self.bnmul2(F_mul2)
        F_mul2_jump = self.jump_layer(F_mul2)
        F_mul3 = self.convmul3(x3 * y3)
        F_mul3 = self.bnmul3(F_mul3)
        F_mul3_jump = self.jump_layer(F_mul3)
        F_mul4 = self.convmul4(x4 * y4)
        F_mul4 = self.bnmul4(F_mul4)
        F_mul4_jump = self.jump_layer(F_mul4)

        F_diff1_attention = F_diff1 * F_mul1_jump
        F_diff2_attention = F_diff2 * F_mul2_jump
        F_diff3_attention = F_diff3 * F_mul3_jump
        F_diff4_attention = F_diff4 * F_mul4_jump

        #或许应该注意力图和相加之后图的卷积图进行一个乘．再加上单纯的add，先试试对tensor进行ｃｂ的，如果不好用在直接用add
        F_add1_conv = self.convadd1(x1 + y1)
        F_add1_conv = self.bnadd1(F_add1_conv)
        F_add2_conv = self.convadd2(x2 + y2)
        F_add2_conv = self.bnadd2(F_add2_conv)
        F_add3_conv = self.convadd3(x3 + y3)
        F_add3_conv = self.bnadd3(F_add3_conv)
        F_add4_conv = self.convadd4(x4 + y4)
        F_add4_conv = self.bnadd4(F_add4_conv)

        F_diff1_attention_map = F_diff1_attention * F_add1_conv
        F_diff2_attention_map = F_diff2_attention * F_add2_conv
        F_diff3_attention_map = F_diff3_attention * F_add3_conv
        F_diff4_attention_map = F_diff4_attention * F_add4_conv


        F_add1 = x1_raw + y1_raw + F_diff1_attention_map
        F_add2 = x2_raw + y2_raw + F_diff2_attention_map
        F_add3 = x3_raw + y3_raw + F_diff3_attention_map
        F_add4 = x4_raw + y4_raw + F_diff4_attention_map

        return [F_add1, F_add2, F_add3, F_add4]





    def jump_layer(self, input):
        ones = torch.ones_like(input)
        ones_point_five = torch.ones_like(input) * 1.5
        result = torch.where(input > 0, ones, ones_point_five)
        return result













#搞一个即插即用的差异模块

class DNANet(nn.Module):
    def __init__(self, num_classes, input_channels, block, num_blocks, nb_filter,deep_supervision=False):
        super(DNANet, self).__init__()
        self.relu = nn.ReLU(inplace = True)
        self.deep_supervision = deep_supervision
        self.pool  = nn.MaxPool2d(2, 2)
        self.up    = nn.Upsample(scale_factor=2,   mode='bilinear', align_corners=True)
        self.down  = nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=True)

        self.up_4  = nn.Upsample(scale_factor=4,   mode='bilinear', align_corners=True)
        self.up_8  = nn.Upsample(scale_factor=8,   mode='bilinear', align_corners=True)
        self.up_16 = nn.Upsample(scale_factor=16,  mode='bilinear', align_corners=True)

        self.conv0_0 = self._make_layer(block, input_channels, nb_filter[0])
        self.conv1_0 = self._make_layer(block, nb_filter[0],  nb_filter[1], num_blocks[0])
        self.conv2_0 = self._make_layer(block, nb_filter[1],  nb_filter[2], num_blocks[1])
        self.conv3_0 = self._make_layer(block, nb_filter[2],  nb_filter[3], num_blocks[2])
        self.conv4_0 = self._make_layer(block, nb_filter[3],  nb_filter[4], num_blocks[3])

        #tianjia
        self.conv1_0_two = self._make_layer(block, nb_filter[0],  nb_filter[1], num_blocks[0])
        self.conv2_0_two = self._make_layer(block, nb_filter[1],  nb_filter[2], num_blocks[1])
        self.conv3_0_two = self._make_layer(block, nb_filter[2],  nb_filter[3], num_blocks[2])
        self.conv4_0_two = self._make_layer(block, nb_filter[3],  nb_filter[4], num_blocks[3])

        self.conv0_1_two = self._make_layer(block, nb_filter[0] + nb_filter[1],  nb_filter[0])
        self.conv1_1_two = self._make_layer(block, nb_filter[1] + nb_filter[2] + nb_filter[0],  nb_filter[1], num_blocks[0])
        self.conv2_1_two = self._make_layer(block, nb_filter[2] + nb_filter[3] + nb_filter[1],  nb_filter[2], num_blocks[1])
        self.conv3_1_two = self._make_layer(block, nb_filter[3] + nb_filter[4] + nb_filter[2],  nb_filter[3], num_blocks[2])

        self.conv0_2_two = self._make_layer(block, nb_filter[0]*2 + nb_filter[1], nb_filter[0])
        self.conv1_2_two = self._make_layer(block, nb_filter[1]*2 + nb_filter[2]+ nb_filter[0], nb_filter[1], num_blocks[0])
        self.conv2_2_two = self._make_layer(block, nb_filter[2]*2 + nb_filter[3]+ nb_filter[1], nb_filter[2], num_blocks[1])

        self.conv0_3_two = self._make_layer(block, nb_filter[0]*3 + nb_filter[1], nb_filter[0])
        self.conv1_3_two = self._make_layer(block, nb_filter[1]*3 + nb_filter[2]+ nb_filter[0], nb_filter[1], num_blocks[0])

        self.conv0_4_two = self._make_layer(block, nb_filter[0]*4 + nb_filter[1], nb_filter[0])

        self.conv0_4_final_two = self._make_layer(block, nb_filter[0]*5, nb_filter[0])

        self.conv0_4_1x1_two = nn.Conv2d(nb_filter[4], nb_filter[0], kernel_size=1, stride=1)
        self.conv0_3_1x1_two = nn.Conv2d(nb_filter[3], nb_filter[0], kernel_size=1, stride=1)
        self.conv0_2_1x1_two = nn.Conv2d(nb_filter[2], nb_filter[0], kernel_size=1, stride=1)
        self.conv0_1_1x1_two = nn.Conv2d(nb_filter[1], nb_filter[0], kernel_size=1, stride=1)
        #tianjia

        self.conv0_1 = self._make_layer(block, nb_filter[0] + nb_filter[1],  nb_filter[0])
        self.conv1_1 = self._make_layer(block, nb_filter[1] + nb_filter[2] + nb_filter[0],  nb_filter[1], num_blocks[0])
        self.conv2_1 = self._make_layer(block, nb_filter[2] + nb_filter[3] + nb_filter[1],  nb_filter[2], num_blocks[1])
        self.conv3_1 = self._make_layer(block, nb_filter[3] + nb_filter[4] + nb_filter[2],  nb_filter[3], num_blocks[2])

        self.conv0_2 = self._make_layer(block, nb_filter[0]*2 + nb_filter[1], nb_filter[0])
        self.conv1_2 = self._make_layer(block, nb_filter[1]*2 + nb_filter[2]+ nb_filter[0], nb_filter[1], num_blocks[0])
        self.conv2_2 = self._make_layer(block, nb_filter[2]*2 + nb_filter[3]+ nb_filter[1], nb_filter[2], num_blocks[1])

        self.conv0_3 = self._make_layer(block, nb_filter[0]*3 + nb_filter[1], nb_filter[0])
        self.conv1_3 = self._make_layer(block, nb_filter[1]*3 + nb_filter[2]+ nb_filter[0], nb_filter[1], num_blocks[0])

        self.conv0_4 = self._make_layer(block, nb_filter[0]*4 + nb_filter[1], nb_filter[0])

        self.conv0_4_final = self._make_layer(block, nb_filter[0]*5, nb_filter[0])

        self.conv0_4_1x1 = nn.Conv2d(nb_filter[4], nb_filter[0], kernel_size=1, stride=1)
        self.conv0_3_1x1 = nn.Conv2d(nb_filter[3], nb_filter[0], kernel_size=1, stride=1)
        self.conv0_2_1x1 = nn.Conv2d(nb_filter[2], nb_filter[0], kernel_size=1, stride=1)
        self.conv0_1_1x1 = nn.Conv2d(nb_filter[1], nb_filter[0], kernel_size=1, stride=1)

        self.mqw = mqwNet2(1, 1)

        if self.deep_supervision:
            self.final1 = nn.Conv2d (nb_filter[0], num_classes, kernel_size=1)
            self.final2 = nn.Conv2d (nb_filter[0], num_classes, kernel_size=1)
            self.final3 = nn.Conv2d (nb_filter[0], num_classes, kernel_size=1)
            self.final4 = nn.Conv2d (nb_filter[0], num_classes, kernel_size=1)

            # self.final1 = nn.Conv2d (2, num_classes, kernel_size=1)
            # self.final2 = nn.Conv2d (2, num_classes, kernel_size=1)
            # self.final3 = nn.Conv2d (2, num_classes, kernel_size=1)
            # self.final4 = nn.Conv2d (2, num_classes, kernel_size=1)
        else:
            self.final  = nn.Conv2d (nb_filter[0], num_classes, kernel_size=1)

    def _make_layer(self, block, input_channels,  output_channels, num_blocks=1):
        layers = []
        layers.append(block(input_channels, output_channels))
        for i in range(num_blocks-1):
            layers.append(block(output_channels, output_channels))
        return nn.Sequential(*layers)

    def forward(self, input, inputreverse):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0),self.down(x0_1)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0),self.down(x1_1)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1),self.down(x0_2)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0),self.down(x2_1)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1),self.down(x1_2)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2),self.down(x0_3)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))

        Final_x0_4 = self.conv0_4_final(
            torch.cat([self.up_16(self.conv0_4_1x1(x4_0)),self.up_8(self.conv0_3_1x1(x3_1)),
                       self.up_4 (self.conv0_2_1x1(x2_2)),self.up  (self.conv0_1_1x1(x1_3)), x0_4], 1)) #这个就是特征提取模块的输出经过特征融合模块之后的输出，其实这个结构就是特征金字塔
#####下面是添加的
        x0_0_two = self.conv0_0(inputreverse)
        x1_0_two = self.conv1_0_two(self.pool(x0_0_two))
        x0_1_two = self.conv0_1_two(torch.cat([x0_0_two, self.up(x1_0_two)], 1))

        x2_0_two = self.conv2_0_two(self.pool(x1_0_two))
        x1_1_two = self.conv1_1_two(torch.cat([x1_0_two, self.up(x2_0_two), self.down(x0_1_two)], 1))
        x0_2_two = self.conv0_2_two(torch.cat([x0_0_two, x0_1_two, self.up(x1_1_two)], 1))

        x3_0_two = self.conv3_0_two(self.pool(x2_0_two))
        x2_1_two = self.conv2_1_two(torch.cat([x2_0_two, self.up(x3_0_two), self.down(x1_1_two)], 1))
        x1_2_two = self.conv1_2_two(torch.cat([x1_0_two, x1_1_two, self.up(x2_1_two), self.down(x0_2_two)], 1))
        x0_3_two = self.conv0_3_two(torch.cat([x0_0_two, x0_1_two, x0_2_two, self.up(x1_2_two)], 1))

        x4_0_two = self.conv4_0_two(self.pool(x3_0_two))
        x3_1_two = self.conv3_1_two(torch.cat([x3_0_two, self.up(x4_0_two), self.down(x2_1_two)], 1))
        x2_2_two = self.conv2_2_two(torch.cat([x2_0_two, x2_1_two, self.up(x3_1_two), self.down(x1_2_two)], 1))
        x1_3_two = self.conv1_3_two(torch.cat([x1_0_two, x1_1_two, x1_2_two, self.up(x2_2_two), self.down(x0_3_two)], 1))
        x0_4_two = self.conv0_4_two(torch.cat([x0_0_two, x0_1_two, x0_2_two, x0_3_two, self.up(x1_3_two)], 1))
        Final_x0_4_two = self.conv0_4_final_two(
            torch.cat([self.up_16(self.conv0_4_1x1_two(x4_0_two)), self.up_8(self.conv0_3_1x1_two(x3_1_two)),
                       self.up_4(self.conv0_2_1x1_two(x2_2_two)), self.up(self.conv0_1_1x1_two(x1_3_two)), x0_4_two], 1))
        # mqwresult = self.mqw(x0_1, x0_2, x0_3, Final_x0_4,
        #                      x0_1_two, x0_2_two, x0_3_two, Final_x0_4_two)
###上面是添加的
        if self.deep_supervision:

            # output1 = self.final1(mqwresult[0])
            # output2 = self.final2(mqwresult[1])
            # output3 = self.final3(mqwresult[2])
            # output4 = self.final4(mqwresult[3])

            output1 = self.final1(x0_1)
            output2 = self.final2(x0_2)
            output3 = self.final3(x0_3)
            output4 = self.final4(Final_x0_4)
            outputfinalceshi = output4
          ##下面是添加的
            output1_two = self.final1(x0_1_two)
            output2_two = self.final2(x0_2_two)
            output3_two = self.final3(x0_3_two)
            output4_two = self.final4(Final_x0_4_two)
            output_twofinalceshi = output4_two

            mqwresult = self.mqw(output1, output2, output3, output4,
                                 output1_two, output2_two, output3_two, output4_two)




            outputsum1 = mqwresult[0]
            outputsum2 = mqwresult[1]
            outputsum3 = mqwresult[2]
            outputsum4 = mqwresult[3]

          ##上面是添加的
            return [outputsum1, outputsum2, outputsum3, outputsum4]
            #return [outputsum1, outputsum2, outputsum3, outputsum4, outputfinalceshi, output_twofinalceshi]
        else:
            output1 = self.final(Final_x0_4)
            output2 = self.final(Final_x0_4_two)
            output = output1+output2
            return output


