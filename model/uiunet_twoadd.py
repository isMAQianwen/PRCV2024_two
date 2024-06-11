import torch
import torch.nn as nn
import torch.nn.functional as F
#############################################
from .fusion_uiu import *
##################################################

class REBNCONV(nn.Module):
    def __init__(self,in_ch=3,out_ch=3,dirate=1):
        super(REBNCONV,self).__init__()

        self.conv_s1 = nn.Conv2d(in_ch,out_ch,3,padding=1*dirate,dilation=1*dirate)
        self.bn_s1 = nn.BatchNorm2d(out_ch)
        self.relu_s1 = nn.ReLU(inplace=True)

    def forward(self,x):

        hx = x
        xout = self.relu_s1(self.bn_s1(self.conv_s1(hx)))

        return xout

def _upsample_like(src,tar):

    src = F.upsample(src, size=tar.shape[2:], mode='bilinear')

    return src
def _upsample_like_y(src,tar):

    src = F.upsample(src, size=tar.shape[2:], mode='bilinear')

    return src


### RSU-7 ###
class RSU7(nn.Module):#UNet07DRES(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU7,self).__init__()

        self.rebnconvin = REBNCONV(in_ch,out_ch,dirate=1)

        self.rebnconv1 = REBNCONV(out_ch,mid_ch,dirate=1)
        self.pool1 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv2 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool2 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv3 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool3 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv4 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool4 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv5 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool5 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv6 = REBNCONV(mid_ch,mid_ch,dirate=1)

        self.rebnconv7 = REBNCONV(mid_ch,mid_ch,dirate=2)

        self.rebnconv6d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv5d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv4d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv3d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch*2,out_ch,dirate=1)

    def forward(self,x):

        hx = x
        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)

        hx4 = self.rebnconv4(hx)
        hx = self.pool4(hx4)

        hx5 = self.rebnconv5(hx)
        hx = self.pool5(hx5)

        hx6 = self.rebnconv6(hx)

        hx7 = self.rebnconv7(hx6)

        hx6d =  self.rebnconv6d(torch.cat((hx7,hx6),1))
        hx6dup = _upsample_like(hx6d,hx5)

        hx5d =  self.rebnconv5d(torch.cat((hx6dup,hx5),1))
        hx5dup = _upsample_like(hx5d,hx4)

        hx4d = self.rebnconv4d(torch.cat((hx5dup,hx4),1))
        hx4dup = _upsample_like(hx4d,hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4dup,hx3),1))
        hx3dup = _upsample_like(hx3d,hx2)

        hx2d = self.rebnconv2d(torch.cat((hx3dup,hx2),1))
        hx2dup = _upsample_like(hx2d,hx1)

        hx1d = self.rebnconv1d(torch.cat((hx2dup,hx1),1))

        return hx1d + hxin

### RSU-6 ###
class RSU6(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU6,self).__init__()

        self.rebnconvin = REBNCONV(in_ch,out_ch,dirate=1)

        self.rebnconv1 = REBNCONV(out_ch,mid_ch,dirate=1)
        self.pool1 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv2 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool2 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv3 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool3 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv4 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool4 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv5 = REBNCONV(mid_ch,mid_ch,dirate=1)

        self.rebnconv6 = REBNCONV(mid_ch,mid_ch,dirate=2)

        self.rebnconv5d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv4d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv3d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch*2,out_ch,dirate=1)

    def forward(self,x):

        hx = x

        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)

        hx4 = self.rebnconv4(hx)
        hx = self.pool4(hx4)

        hx5 = self.rebnconv5(hx)

        hx6 = self.rebnconv6(hx5)

        hx5d =  self.rebnconv5d(torch.cat((hx6,hx5),1))
        hx5dup = _upsample_like(hx5d,hx4)

        hx4d = self.rebnconv4d(torch.cat((hx5dup,hx4),1))
        hx4dup = _upsample_like(hx4d,hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4dup,hx3),1))
        hx3dup = _upsample_like(hx3d,hx2)

        hx2d = self.rebnconv2d(torch.cat((hx3dup,hx2),1))
        hx2dup = _upsample_like(hx2d,hx1)

        hx1d = self.rebnconv1d(torch.cat((hx2dup,hx1),1))

        return hx1d + hxin

### RSU-5 ###
class RSU5(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU5,self).__init__()

        self.rebnconvin = REBNCONV(in_ch,out_ch,dirate=1)

        self.rebnconv1 = REBNCONV(out_ch,mid_ch,dirate=1)
        self.pool1 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv2 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool2 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv3 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool3 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv4 = REBNCONV(mid_ch,mid_ch,dirate=1)

        self.rebnconv5 = REBNCONV(mid_ch,mid_ch,dirate=2)

        self.rebnconv4d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv3d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch*2,out_ch,dirate=1)

    def forward(self,x):

        hx = x

        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)

        hx4 = self.rebnconv4(hx)

        hx5 = self.rebnconv5(hx4)

        hx4d = self.rebnconv4d(torch.cat((hx5,hx4), 1))
        hx4dup = _upsample_like(hx4d, hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4dup, hx3), 1))
        hx3dup = _upsample_like(hx3d, hx2)

        hx2d = self.rebnconv2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = _upsample_like(hx2d, hx1)

        hx1d = self.rebnconv1d(torch.cat((hx2dup,hx1),1))

        return hx1d + hxin

### RSU-4 ###
class RSU4(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU4,self).__init__()

        self.rebnconvin = REBNCONV(in_ch,out_ch,dirate=1)

        self.rebnconv1 = REBNCONV(out_ch,mid_ch,dirate=1)
        self.pool1 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv2 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool2 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv3 = REBNCONV(mid_ch,mid_ch,dirate=1)

        self.rebnconv4 = REBNCONV(mid_ch,mid_ch,dirate=2)

        self.rebnconv3d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch*2,out_ch,dirate=1)

    def forward(self,x):

        hx = x

        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)

        hx4 = self.rebnconv4(hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4,hx3), 1))
        hx3dup = _upsample_like(hx3d, hx2)

        hx2d = self.rebnconv2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = _upsample_like(hx2d, hx1)

        hx1d = self.rebnconv1d(torch.cat((hx2dup,hx1),1))

        return hx1d + hxin

### RSU-4F ###
class RSU4F(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU4F,self).__init__()

        self.rebnconvin = REBNCONV(in_ch,out_ch,dirate=1)

        self.rebnconv1 = REBNCONV(out_ch,mid_ch,dirate=1)
        self.rebnconv2 = REBNCONV(mid_ch,mid_ch,dirate=2)
        self.rebnconv3 = REBNCONV(mid_ch,mid_ch,dirate=4)

        self.rebnconv4 = REBNCONV(mid_ch,mid_ch,dirate=8)

        self.rebnconv3d = REBNCONV(mid_ch*2,mid_ch,dirate=4)
        self.rebnconv2d = REBNCONV(mid_ch*2,mid_ch,dirate=2)
        self.rebnconv1d = REBNCONV(mid_ch*2,out_ch,dirate=1)

    def forward(self,x):

        hx = x

        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx2 = self.rebnconv2(hx1)
        hx3 = self.rebnconv3(hx2)

        hx4 = self.rebnconv4(hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4,hx3),1))
        hx2d = self.rebnconv2d(torch.cat((hx3d,hx2),1))
        hx1d = self.rebnconv1d(torch.cat((hx2d,hx1),1))

        return hx1d + hxin


##### UIU-net ####
class UIUNET(nn.Module):

    def __init__(self, in_ch=3, out_ch=1):
        super(UIUNET, self).__init__()

        self.stage1 = RSU7(in_ch,32,64)
        self.pool12 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.stage2 = RSU6(64,32,128)
        self.pool23 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.stage3 = RSU5(128,64,256)
        self.pool34 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.stage4 = RSU4(256,128,512)
        self.pool45 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.stage5 = RSU4F(512,256,512)
        self.pool56 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.stage6 = RSU4F(512,256,512)

        # decoder
        self.stage5d = RSU4F(1024,256,512)
        self.stage4d = RSU4(1024,128,256)
        self.stage3d = RSU5(512,64,128)
        self.stage2d = RSU6(256,32,64)
        self.stage1d = RSU7(128,16,64)

        self.side1 = nn.Conv2d(64,out_ch,3,padding=1)
        self.side2 = nn.Conv2d(64,out_ch,3,padding=1)
        self.side3 = nn.Conv2d(128,out_ch,3,padding=1)
        self.side4 = nn.Conv2d(256,out_ch,3,padding=1)
        self.side5 = nn.Conv2d(512,out_ch,3,padding=1)
        self.side6 = nn.Conv2d(512,out_ch,3,padding=1)

        self.outconv = nn.Conv2d(6*out_ch,out_ch,1)

        #self.fuse6 = self._fuse_layer(512, 512, 512, fuse_mode='AsymBi')
        self.fuse5 = self._fuse_layer(512, 512, 512, fuse_mode='AsymBi')
        self.fuse4 = self._fuse_layer(512, 512, 512, fuse_mode='AsymBi')
        self.fuse3 = self._fuse_layer(256, 256, 256, fuse_mode='AsymBi')
        self.fuse2 = self._fuse_layer(128, 128, 128, fuse_mode='AsymBi')

##下面是添加的
        self.pool12_y = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.stage2_y = RSU6(64,32,128)
        self.pool23_y = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.stage3_y = RSU5(128,64,256)
        self.pool34_y = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.stage4_y = RSU4(256,128,512)
        self.pool45_y = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.stage5_y = RSU4F(512,256,512)
        self.pool56_y = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.stage6_y = RSU4F(512,256,512)

        # decoder
        self.stage5d_y = RSU4F(1024,256,512)
        self.stage4d_y = RSU4(1024,128,256)
        self.stage3d_y = RSU5(512,64,128)
        self.stage2d_y = RSU6(256,32,64)
        self.stage1d_y = RSU7(128,16,64)

        self.side1_y = nn.Conv2d(64,out_ch,3,padding=1)
        self.side2_y = nn.Conv2d(64,out_ch,3,padding=1)
        self.side3_y = nn.Conv2d(128,out_ch,3,padding=1)
        self.side4_y = nn.Conv2d(256,out_ch,3,padding=1)
        self.side5_y = nn.Conv2d(512,out_ch,3,padding=1)
        self.side6_y = nn.Conv2d(512,out_ch,3,padding=1)

        self.outconv_y = nn.Conv2d(6*out_ch,out_ch,1)

        #self.fuse6 = self._fuse_layer(512, 512, 512, fuse_mode='AsymBi')
        self.fuse5_y = self._fuse_layer(512, 512, 512, fuse_mode='AsymBi')
        self.fuse4_y = self._fuse_layer(512, 512, 512, fuse_mode='AsymBi')
        self.fuse3_y = self._fuse_layer(256, 256, 256, fuse_mode='AsymBi')
        self.fuse2_y = self._fuse_layer(128, 128, 128, fuse_mode='AsymBi')



#上面是添加的


    def _fuse_layer(self, in_high_channels, in_low_channels, out_channels,fuse_mode='AsymBi'):#fuse_mode='AsymBi'
        # assert fuse_mode in ['BiLocal', 'AsymBi', 'BiGlobal']
        # if fuse_mode == 'BiLocal':
        #     fuse_layer = BiLocalChaFuseReduce(in_high_channels, in_low_channels, out_channels)
        # el
        if fuse_mode == 'AsymBi':
            fuse_layer = AsymBiChaFuseReduce(in_high_channels, in_low_channels, out_channels)
        # elif fuse_mode == 'BiGlobal':
        #     fuse_layer = BiGlobalChaFuseReduce(in_high_channels, in_low_channels, out_channels)
        else:
            NameError
        return fuse_layer


    def forward(self, x, y):

        hx = x
        hy = y

        #stage 1
        hx1 = self.stage1(hx)
        hx = self.pool12(hx1)


        #stage 2
        hx2 = self.stage2(hx)
        hx = self.pool23(hx2)

        #stage 3
        hx3 = self.stage3(hx)
        hx = self.pool34(hx3)

        #stage 4
        hx4 = self.stage4(hx)
        hx = self.pool45(hx4)

        #stage 5
        hx5 = self.stage5(hx)
        hx = self.pool56(hx5)

        #stage 6
        hx6 = self.stage6(hx)
        hx6up = _upsample_like(hx6,hx5)

        #-------------------- decoder --------------------

        fusec51,fusec52 = self.fuse5(hx6up, hx5)
        hx5d = self.stage5d(torch.cat((fusec51, fusec52),1))
        hx5dup = _upsample_like(hx5d,hx4)

        fusec41,fusec42 = self.fuse4(hx5dup, hx4)
        hx4d = self.stage4d(torch.cat((fusec41,fusec42),1))
        hx4dup = _upsample_like(hx4d,hx3)


        fusec31,fusec32 = self.fuse3(hx4dup, hx3)
        hx3d = self.stage3d(torch.cat((fusec31,fusec32),1))
        hx3dup = _upsample_like(hx3d,hx2)

        fusec21, fusec22 = self.fuse2(hx3dup, hx2)
        hx2d = self.stage2d(torch.cat((fusec21, fusec22), 1))
        hx2dup = _upsample_like(hx2d,hx1)


        hx1d = self.stage1d(torch.cat((hx2dup,hx1),1))


        #side output
        d1 = self.side1(hx1d)

        d22 = self.side2(hx2d)
        d2 = _upsample_like(d22,d1)

        d32 = self.side3(hx3d)
        d3 = _upsample_like(d32,d1)

        d42 = self.side4(hx4d)
        d4 = _upsample_like(d42,d1)

        d52 = self.side5(hx5d)
        d5 = _upsample_like(d52,d1)

        d62 = self.side6(hx6)
        d6 = _upsample_like(d62,d1)
        d0 = self.outconv(torch.cat((d1, d2, d3, d4, d5, d6), 1))
        #d0 = self.outconv(torch.cat((d1,d2,d3,d4,d5,d6),1))
###下面是添加的        #stage 1
        hy1 = self.stage1(hy)
        hy = self.pool12_y(hy1)


        #stage 2
        hy2 = self.stage2_y(hy)
        hy = self.pool23_y(hy2)

        #stage 3
        hy3 = self.stage3_y(hy)
        hy = self.pool34_y(hy3)

        #stage 4
        hy4 = self.stage4_y(hy)
        hy = self.pool45_y(hy4)

        #stage 5
        hy5 = self.stage5_y(hy)
        hy = self.pool56_y(hy5)

        #stage 6
        hy6 = self.stage6_y(hy)
        hy6up = _upsample_like_y(hy6,hy5)

        #-------------------- decoder --------------------

        fusec51_y,fusec52_y = self.fuse5_y(hy6up, hy5)
        hy5d = self.stage5d_y(torch.cat((fusec51_y, fusec52_y),1))
        hy5dup = _upsample_like_y(hy5d,hy4)

        fusec41_y,fusec42_y = self.fuse4_y(hy5dup, hy4)
        hy4d = self.stage4d_y(torch.cat((fusec41_y,fusec42_y),1))
        hy4dup = _upsample_like_y(hy4d,hy3)


        fusec31_y,fusec32_y = self.fuse3_y(hy4dup, hy3)
        hy3d = self.stage3d_y(torch.cat((fusec31_y,fusec32_y),1))
        hy3dup = _upsample_like_y(hy3d,hy2)

        fusec21_y, fusec22_y = self.fuse2_y(hy3dup, hy2)
        hy2d = self.stage2d_y(torch.cat((fusec21_y, fusec22_y), 1))
        hy2dup = _upsample_like_y(hy2d,hy1)


        hy1d = self.stage1d_y(torch.cat((hy2dup,hy1),1))

##################################################

        #side output
        d1_y = self.side1_y(hy1d)

        d22_y = self.side2_y(hy2d)
        d2_y = _upsample_like_y(d22_y,d1_y)

        d32_y = self.side3_y(hy3d)
        d3_y = _upsample_like_y(d32_y,d1_y)

        d42_y = self.side4_y(hy4d)
        d4_y = _upsample_like_y(d42_y,d1_y)

        d52_y = self.side5_y(hy5d)
        d5_y = _upsample_like_y(d52_y,d1_y)

        d62_y = self.side6_y(hy6)
        d6_y = _upsample_like_y(d62_y,d1_y)

        d0_y = self.outconv(torch.cat((d1_y, d2_y, d3_y, d4_y, d5_y, d6_y), 1))
        #####################################
        d1sum = d1 + d1_y
        d2sum = d2 + d2_y
        d3sum = d3 + d3_y
        d4sum = d4 + d4_y
        d5sum = d5 + d5_y
        d6sum = d6 + d6_y
        d0sum = d0 + d0_y
        #d0sum = self.outconv(d0sum)


        #d0_y = self.outconv(torch.cat((d1_y,d2_y,d3_y,d4_y,d5_y,d6_y),1))




#上面是添加的



        return [d0sum, d1sum, d2sum, d3sum, d4sum, d5sum, d6sum]

