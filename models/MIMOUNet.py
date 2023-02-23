import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import *


class EBlock(nn.Module):
    def __init__(self, out_channel, num_res=8):
        super(EBlock, self).__init__()

        layers = [ResBlock(out_channel, out_channel) for _ in range(num_res)]

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class DBlock(nn.Module):
    def __init__(self, channel, num_res=8):
        super(DBlock, self).__init__()

        layers = [ResBlock(channel, channel) for _ in range(num_res)]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class AFF(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(AFF, self).__init__()
        self.conv = nn.Sequential(
            BasicConv(in_channel, out_channel, kernel_size=1, stride=1, relu=True),
            BasicConv(out_channel, out_channel, kernel_size=3, stride=1, relu=False)
        )

    def forward(self, x1, x2, x4):
        x = torch.cat([x1, x2, x4], dim=1)
        return self.conv(x)


class SCM(nn.Module):
    def __init__(self, out_plane):
        super(SCM, self).__init__()
        self.main = nn.Sequential(
            BasicConv(3, out_plane//4, kernel_size=3, stride=1, relu=True),
            BasicConv(out_plane // 4, out_plane // 2, kernel_size=1, stride=1, relu=True),
            BasicConv(out_plane // 2, out_plane // 2, kernel_size=3, stride=1, relu=True),
            BasicConv(out_plane // 2, out_plane-3, kernel_size=1, stride=1, relu=True)
        )

        self.conv = BasicConv(out_plane, out_plane, kernel_size=1, stride=1, relu=False)

    def forward(self, x):
        x = torch.cat([x, self.main(x)], dim=1)
        return self.conv(x)


class FAM(nn.Module):
    def __init__(self, channel):
        super(FAM, self).__init__()
        self.merge = BasicConv(channel, channel, kernel_size=3, stride=1, relu=False)

    def forward(self, x1, x2):
        x = x1 * x2
        out = x1 + self.merge(x)
        return out


class MIMOUNet(nn.Module):
    def __init__(self, num_res=8):
        super(MIMOUNet, self).__init__()

        base_channel = 32

        self.Encoder = nn.ModuleList([
            EBlock(base_channel, num_res),
            EBlock(base_channel*2, num_res),
            EBlock(base_channel*4, num_res),
        ])

        self.feat_extract = nn.ModuleList([
            BasicConv(3, base_channel, kernel_size=3, relu=True, stride=1),
            BasicConv(base_channel, base_channel*2, kernel_size=3, relu=True, stride=2),
            BasicConv(base_channel*2, base_channel*4, kernel_size=3, relu=True, stride=2),
            BasicConv(base_channel*4, base_channel*2, kernel_size=4, relu=True, stride=2, transpose=True),
            BasicConv(base_channel*2, base_channel, kernel_size=4, relu=True, stride=2, transpose=True),
            BasicConv(base_channel, 3, kernel_size=3, relu=False, stride=1)
        ])

        self.Decoder = nn.ModuleList([
            DBlock(base_channel * 4, num_res),
            DBlock(base_channel * 2, num_res),
            DBlock(base_channel, num_res)
        ])

        self.Convs = nn.ModuleList([
            BasicConv(base_channel * 4, base_channel * 2, kernel_size=1, relu=True, stride=1),
            BasicConv(base_channel * 2, base_channel, kernel_size=1, relu=True, stride=1),
        ])

        self.ConvsOut = nn.ModuleList(
            [
                BasicConv(base_channel * 4, 3, kernel_size=3, relu=False, stride=1),
                BasicConv(base_channel * 2, 3, kernel_size=3, relu=False, stride=1),
            ]
        )

        self.AFFs = nn.ModuleList([
            AFF(base_channel * 7, base_channel*1),
            AFF(base_channel * 7, base_channel*2)
        ])

        self.FAM1 = FAM(base_channel * 4)
        self.SCM1 = SCM(base_channel * 4)
        self.FAM2 = FAM(base_channel * 2)
        self.SCM2 = SCM(base_channel * 2)

    def forward(self, x):
        x_2 = F.interpolate(x, scale_factor=0.5)
        x_4 = F.interpolate(x_2, scale_factor=0.5)
        z2 = self.SCM2(x_2)
        z4 = self.SCM1(x_4)

        outputs = list()

        x_ = self.feat_extract[0](x)
        res1 = self.Encoder[0](x_)

        z = self.feat_extract[1](res1)
        z = self.FAM2(z, z2)
        res2 = self.Encoder[1](z)

        z = self.feat_extract[2](res2)
        z = self.FAM1(z, z4)
        z = self.Encoder[2](z)

        z12 = F.interpolate(res1, scale_factor=0.5)
        z21 = F.interpolate(res2, scale_factor=2)
        z42 = F.interpolate(z, scale_factor=2)
        z41 = F.interpolate(z42, scale_factor=2)

        res2 = self.AFFs[1](z12, res2, z42)
        res1 = self.AFFs[0](res1, z21, z41)

        z = self.Decoder[0](z)
        z_ = self.ConvsOut[0](z)
        z = self.feat_extract[3](z)
        outputs.append(z_+x_4)

        z = torch.cat([z, res2], dim=1)
        z = self.Convs[0](z)
        z = self.Decoder[1](z)
        z_ = self.ConvsOut[1](z)
        z = self.feat_extract[4](z)
        outputs.append(z_+x_2)

        z = torch.cat([z, res1], dim=1)
        z = self.Convs[1](z)
        z = self.Decoder[2](z)
        z = self.feat_extract[5](z)
        outputs.append(z+x)

        return outputs


class MIMOUNetPlus(nn.Module):
    def __init__(self, num_res = 20):
        super(MIMOUNetPlus, self).__init__()
        base_channel = 32
        self.Encoder = nn.ModuleList([
            EBlock(base_channel, num_res),
            EBlock(base_channel*2, num_res),
            EBlock(base_channel*4, num_res),
        ])

        self.feat_extract = nn.ModuleList([
            BasicConv(3, base_channel, kernel_size=3, relu=True, stride=1),
            BasicConv(base_channel, base_channel*2, kernel_size=3, relu=True, stride=2),
            BasicConv(base_channel*2, base_channel*4, kernel_size=3, relu=True, stride=2),
            BasicConv(base_channel*4, base_channel*2, kernel_size=4, relu=True, stride=2, transpose=True),
            BasicConv(base_channel*2, base_channel, kernel_size=4, relu=True, stride=2, transpose=True),
            BasicConv(base_channel, 3, kernel_size=3, relu=False, stride=1)
        ])

        self.Decoder = nn.ModuleList([
            DBlock(base_channel * 4, num_res),
            DBlock(base_channel * 2, num_res),
            DBlock(base_channel, num_res)
        ])

        self.Convs = nn.ModuleList([
            BasicConv(base_channel * 4, base_channel * 2, kernel_size=1, relu=True, stride=1),
            BasicConv(base_channel * 2, base_channel, kernel_size=1, relu=True, stride=1),
        ])

        self.ConvsOut = nn.ModuleList(
            [
                BasicConv(base_channel * 4, 3, kernel_size=3, relu=False, stride=1),
                BasicConv(base_channel * 2, 3, kernel_size=3, relu=False, stride=1),
            ]
        )

        self.AFFs = nn.ModuleList([
            AFF(base_channel * 7, base_channel*1),
            AFF(base_channel * 7, base_channel*2)
        ])

        self.FAM1 = FAM(base_channel * 4)
        self.SCM1 = SCM(base_channel * 4)
        self.FAM2 = FAM(base_channel * 2)
        self.SCM2 = SCM(base_channel * 2)

        self.drop1 = nn.Dropout2d(0.1)
        self.drop2 = nn.Dropout2d(0.1)

    def forward(self, x):
        x_2 = F.interpolate(x, scale_factor=0.5)
        x_4 = F.interpolate(x_2, scale_factor=0.5)
        z2 = self.SCM2(x_2)
        z4 = self.SCM1(x_4)

        outputs = list()

        x_ = self.feat_extract[0](x)
        res1 = self.Encoder[0](x_)

        z = self.feat_extract[1](res1)
        z = self.FAM2(z, z2)
        res2 = self.Encoder[1](z)

        z = self.feat_extract[2](res2)
        z = self.FAM1(z, z4)
        z = self.Encoder[2](z)

        z12 = F.interpolate(res1, scale_factor=0.5)
        z21 = F.interpolate(res2, scale_factor=2)
        z42 = F.interpolate(z, scale_factor=2)
        z41 = F.interpolate(z42, scale_factor=2)

        res2 = self.AFFs[1](z12, res2, z42)
        res1 = self.AFFs[0](res1, z21, z41)

        res2 = self.drop2(res2)
        res1 = self.drop1(res1)

        z = self.Decoder[0](z)
        z_ = self.ConvsOut[0](z)
        z = self.feat_extract[3](z)
        outputs.append(z_+x_4)

        z = torch.cat([z, res2], dim=1)
        z = self.Convs[0](z)
        z = self.Decoder[1](z)
        z_ = self.ConvsOut[1](z)
        z = self.feat_extract[4](z)
        outputs.append(z_+x_2)

        z = torch.cat([z, res1], dim=1)
        z = self.Convs[1](z)
        z = self.Decoder[2](z)
        z = self.feat_extract[5](z)
        outputs.append(z+x)

        return outputs

    


class DWTL2(nn.Module):
    def __init__(self,num_res=20):
        super(DWTL2,self).__init__()
        base_channel = 32
        self.fam = FAM(base_channel*2)
        self.EB2 = EBlock(base_channel*2, num_res)
        self.aff2 = AFF(7*base_channel,base_channel*2)
        self.scm2 = SCM(out_plane=base_channel*2,in_plane=12)
        self.model_l3 = DWTL3()
        self.conv_trans = BasicConv(base_channel*4, base_channel*8, kernel_size=3, relu=True, stride=1)
        self.conv_trans2 = BasicConv(base_channel*4, base_channel*2, kernel_size=3, relu=True, stride=1)
        self.conv_trans3 = BasicConv(base_channel*2, 12, kernel_size=3, relu=True, stride=1)
        self.conv_trans4 = BasicConv(base_channel*8, base_channel*4, kernel_size=3, relu=True, stride=1)
        self.conv_trans5 = BasicConv(base_channel*4, base_channel*16, kernel_size=3, relu=True, stride=1)
        slef.DB2 = DBlock(base_channel * 2, num_res)
    
    def forward(self,b2,l1_fea,b1_aff,b3):
        x = self.scms(b2)
        if l1_fea is not None:
            x = self.fam(l1_fea,x)
        x_aff2 = self.EB2(x)
        x = dwt_init(x_aff)
        x = self.conv_trans4(x)
        x_fea3, x_dwt3, x_aff3 = self.DWTL3(b3,x)
        x = self.conv_trans(x_fea3)
        x = idwt_init(x)
        x_aff3_l2 = idwt_init(self.conv_trans5(x_aff3))
        if b1_aff is None:
            b1_aff = torch.zeros((x_aff2.size()[0],base_channel,x_aff.size()[2],x_aff.size()[3])).float().cuda()
        aff2_out = self.aff2(b1_aff,x_aff2,x_aff3_l2)
        x = torch.cat((x,aff2_out),1)
        x = self.conv_trans2(x)
        x_fea2 = self.DB2(x)
        x_dwt2 = self.conv_trans3(x_fea2)
        return x_fea2, x_dwt2, x_aff2, x_aff3_l2, x_dwt3


class DWTL3(nn.Module):
    def __init__(self,num_res=20):
        super(DWTL3,self).__init__()
        base_channel=32
        self.scm3 = SCM(out_plane=base_channel*4,in_plane=48)
        self.fam = FAM(base_channel*4)
        self.EB3 = EBlock(base_channel*4, num_res)
        self.DB3 = DBlock(base_channel * 4, num_res)
        self.conv_trans = BasicConv(base_channel*4, 48, kernel_size=3, relu=True, stride=1)

    def forward(self,b3,l2_fea):
        x = self.scm3(b3)
        if l2_fea is not None:
            x = self.fam(l2_fea, x)
        x_aff3 = self.EB3(x)
        x_fea3 = self.DB3(x_aff3)
        x_dwt3 = self.conv_trans(x_fea3)
        return x_fea3,x_dwt3,x_aff3


class MIMOUNetPlusDWTv2(nn.Module):
    def __init__(self, num_res = 20):
        super(MIMOUNetPlusDWTv2, self).__init__()
        base_channel = 32
        self.extract_fea = BasicConv(3, base_channel, kernel_size=3, relu=True, stride=1)
        self.EB1 = EBlock(base_channel, num_res)
        self.conv_trans1 = BasicConv(base_channel*4, base_channel*2, kernel_size=3, relu=True, stride=1)
        self.conv_trans2 = BasicConv(base_channel*4, base_channel, kernel_size=3, relu=True, stride=1)
        self.conv_trans3 = BasicConv(base_channel*2, base_channel*8, kernel_size=3, relu=True, stride=1)
        self.conv_trans4 = BasicConv(base_channel*4, base_channel*16, kernel_size=3, relu=True, stride=1)
        self.conv_trans5 = BasicConv(base_channel*2, base_channel*4, kernel_size=3, relu=True, stride=1)
        self.conv_trans6 = BasicConv(base_channel*2, base_channel, kernel_size=3, relu=True, stride=1)
        self.conv_trans7 = BasicConv(base_channel, 3, kernel_size=3, relu=True, stride=1)
        self.model_l2 = DWTL2()
        self.aff1 = AFF(base_channel*7, base_channel)
        self.DB1 = DBlock(base_channel, num_res)


    def forward(self, x):
        b1 = x
        b2 = dwt_init(x)
        b3 = dwt_init(b2)
        x = self.extract_fea(b1)
        x_aff1 = self.EB1(x)
        x_dwt = dwt_init(x_aff1)
        x_aff1_l2 = self.conv_trans2(x_dwt)
        x = self.conv_trans1(x_aff1_l2)
        l2_out = self.model_l2(b2,x,x_aff1_l2,b3)
        x_fea2, x_dwt2, x_aff2, x_aff3_l2,x_dwt3 = l2_out
        x_ = self.conv_trans5(x_fea2)
        x_aff2_l1 = idwt_init(self.conv_trans3(x_aff2))
        x_aff3_l1 = idwt_init(self.conv_trans4(x_aff3_l2))
        x = self.aff1(x_aff1,x_aff2_l1,x_aff3_l1)
        x = torch.cat((x_,x))
        x = self.conv_trans6(x)
        x = self.DB1(x)
        x = self.conv_trans7(x)
        outputs.append(x_dwt3+b3)
        outputs.append(x_dwt2+b2)
        outputs.append(x+b1)

        return outputs

    

def build_net(model_name):
    class ModelError(Exception):
        def __init__(self, msg):
            self.msg = msg

        def __str__(self):
            return self.msg

    if model_name == "MIMO-UNetPlus":
        return MIMOUNetPlus()
    elif model_name == "MIMO-UNet":
        return MIMOUNet()
    raise ModelError('Wrong Model!\nYou should choose MIMO-UNetPlus or MIMO-UNet.')
