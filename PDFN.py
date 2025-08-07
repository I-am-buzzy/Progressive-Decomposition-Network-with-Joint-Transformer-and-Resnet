#参数少点，维度32
import numbers
import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from transformernet import *
from utils import *

up_kwargs = {'mode': 'bilinear', 'align_corners': True}


def conv1(in_chsnnels, out_channels):  # "1x1 convolution with padding"
    return nn.Conv2d(in_chsnnels, out_channels, kernel_size=1, stride=1, bias=False)


def conv3(in_chsnnels, out_channels):  # "3x3 convolution with padding"
    return nn.Conv2d(in_chsnnels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)


class FReLU(nn.Module):  # FReLU formulation. The funnel condition has a window size of kxk. (k=3 by default)
    def __init__(self, in_channels):
        super(FReLU, self).__init__()
        self.conv_frelu = nn.Conv2d(in_channels, in_channels, 3, 1, 1, groups=in_channels)
        self.bn_frelu = nn.BatchNorm2d(in_channels)

    def forward(self, x):
        x1 = self.conv_frelu(x)
        x1 = self.bn_frelu(x1)
        x2 = torch.stack([x, x1], dim=0)  # 拼接
        out, _ = torch.max(x2, dim=0)
        return out


class Feature_extract(nn.Module):     #特征提取模块
    def __init__(self, in_channels, out_channels):
        super(Feature_extract, self).__init__()
        self.SFEB1 = nn.Sequential(nn.Conv2d(in_channels, int(out_channels / 2), kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(int(out_channels / 2)), FReLU(int(out_channels / 2)),
                                   nn.Conv2d(int(out_channels / 2), int(out_channels / 2), kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(int(out_channels / 2)), FReLU(int(out_channels / 2)))
        self.SFEB2 = nn.Sequential(nn.Conv2d(int(out_channels / 2), out_channels, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(out_channels), FReLU(out_channels),
                                   nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1))
        self.SFEB3 = BaseFeatureExtraction(int(out_channels), num_heads=8)       #后面加了块归一化和ReLU操作

    def forward(self, x):
        high_x = self.SFEB1(x)
        x = self.SFEB2(high_x)
        x = self.SFEB3(x)
        return high_x, x


class S2M(nn.Module):  # Scene Specific Mask   S2M(32)
    def __init__(self, channels, r=4):  # channels=32
        super(S2M, self).__init__()
        inter_channels = int(channels // r)  # 32//4=8
        self.local_att = nn.Sequential(nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, groups=channels), nn.BatchNorm2d(channels), nn.ReLU(inplace=True),
                                       nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, groups=channels), nn.BatchNorm2d(channels), nn.ReLU(inplace=True),
                                       #nn.Conv2d(inter_channels, inter_channels, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(inter_channels), nn.ReLU(inplace=True),
                                       nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, groups=channels), nn.BatchNorm2d(channels))
                                       #nn.Sequential(Conv2d(32,32,3,1,1),BN,ReLU,Conv2d(32,32,3,1,1),BN,ReLU,Conv2d(32,32,3,1,1),BN)
        self.global_att = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0), nn.BatchNorm2d(inter_channels), nn.ReLU(inplace=True),
                                        nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0), nn.BatchNorm2d(channels))
                                        #nn.Sequential(AdaptiveAvgPool2d(1),Conv2d(32,8,1,1,0),BN,ReLu,Conv2d(8,32,1,1,0),BN(32))
        self.sigmoid = nn.Sigmoid()
        self.conv_block = nn.BatchNorm2d(channels)          #BatchNorm2d(32)

    def forward(self, x):                     #(4,32,256,256)
        # spatial attention
        local_w = self.local_att(x)           #local attention   (4,32,256,256)
        # channel attention
        global_w = self.global_att(x)          #(4,32,1,1)
        mask = self.sigmoid(local_w * global_w)           #(4,32,256,256)
        masked_feature = mask * x              #(4,32,256,256)
        output = self.conv_block(masked_feature)          #(4,32,256,256)
        return output


class Prediction_head(nn.Module):  # 自适应特征连接模块, 用于跳变连接的自适应连接 Adaptive_Connection
    def __init__(self, channels, img=False):
        super(Prediction_head, self).__init__()
        self.conv_block = nn.Sequential(nn.Conv2d(in_channels=channels[0], out_channels=channels[1], kernel_size=3, padding=1), nn.Tanh())

    def forward(self, x):
        return (self.conv_block(x) + 1) / 2


class SFP(nn.Module):              #Scene Fidelity Path    输入尺寸：(4,32,256,256)   SFP([32, 1])
    def __init__(self, channels, img=False):         #channels=[32, 1]
        super(SFP, self).__init__()
        self.mask = S2M(channels[0])  #S2M(32)
        self.conv_block = nn.Sequential(nn.Conv2d(in_channels=channels[0], out_channels=channels[1], kernel_size=3, padding=1), nn.Tanh())      # Conv2d(32,1,3,1), nn.Tanh()

    def forward(self, x):             #(4,32,256,256)
        x = self.mask(x)              #(4,32,256,256)
        return (self.conv_block(x) + 1) / 2           #(4,1,256,256)


class IFP(nn.Module):         # Scene Fidelity Path  输入尺寸：(4,32,256,256)   IFP([32, 1])
    def __init__(self, channels):
        super(IFP, self).__init__()
        self.conv_block = nn.Sequential(nn.Conv2d(in_channels=channels[0], out_channels=channels[1], kernel_size=3, padding=1), nn.Tanh())

    def forward(self, x):                       #输入尺寸：(4,32,256,256)
        return (self.conv_block(x) + 1) / 2           #(4,1,256,256)


class SIM(nn.Module):           # (4 64 128, 128)  (4 64 64,64)  SIM1(norm_nc=64, label_nc=64, nhidden=64)  输出(4 32 128, 128)
    def __init__(self, norm_nc, label_nc, nhidden):    #(4 32 256, 256)  (4 32 128, 128)    SIM1(norm_nc=32, label_nc=32, nhidden=32)  输出(4 32 256, 256)
        super(SIM, self).__init__()
        self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False, track_running_stats=False)               # 归一化
        # self.param_free_norm = nn.BatchNorm2d(norm_nc, affine=False)
        # The dimension of the intermediate embedding space. Yes, hardcoded.
        self.mlp_shared = nn.Sequential(nn.Conv2d(label_nc, nhidden, kernel_size=3, padding=1), nn.ReLU())      # Conv2d(64, 32, 3, 1), nn.ReLU()
        self.mlp_gamma = nn.Sequential(nn.Conv2d(nhidden, norm_nc, kernel_size=3, padding=1), nn.Sigmoid())     # Conv2d(32, 32, 3,1), nn.Sigmoid()
        self.mlp_beta = nn.Conv2d(nhidden, norm_nc, kernel_size=3, padding=1)  # Conv2d(64, 32, 3,1)
        self.bn = nn.BatchNorm2d(num_features=norm_nc)

    def forward(self, x, segmap):      #(4 32 128, 128)  (4 64 64,64)  SIM1(norm_nc=32, label_nc=64, nhidden=64)
        # Part 1. generate parameter-free normalized activations
        normalized = self.param_free_norm(x)      # (4 32 256, 256)
        # Part 2. produce scaling and bias conditioned on semantic map          上采样之后  (4 64 128, 128)
        segmap = F.interpolate(segmap, size=x.size()[2:], mode='bilinear')      #(4 64 256, 256)数组采样操作, 利用插值方法，对输入的张量数组进行上\下采样操作，换句话说就是科学合理地改变数组的尺寸大小，尽量保持数据完整。
        actv = self.mlp_shared(segmap)  # (4 32 256, 256)
        # actv = segmap
        gamma = self.mlp_gamma(actv)  # (4 32 256, 256)  多做了nn.Sigmoid()
        beta = self.mlp_beta(actv)    # (4 32 256, 256)
        # apply scale and bias
        out = self.bn(normalized * (1 + gamma)) + beta  # (4 32 256, 256)
        return out


class PDFN(nn.Module):
    def __init__(self, n_classes):
        global resnet_raw_model1, resnet_raw_model2
        super(PDFN, self).__init__()
        resnet_raw_model1 = models.resnet34(pretrained=True)
        resnet_raw_model2 = models.resnet34(pretrained=True)

        ########  Thermal ENCODER  ir  ########
        self.encoder_thermal_conv1 = Feature_extract(1, 64)
        #self.encoder_thermal_bn1 = resnet_raw_model1.bn1
        #self.encoder_thermal_relu = resnet_raw_model1.relu
        self.encoder_thermal_maxpool = resnet_raw_model1.maxpool
        self.encoder_thermal_layer3 = resnet_raw_model1.layer1
        self.encoder_thermal_layer4 = resnet_raw_model1.layer2
        self.encoder_thermal_layer5 = resnet_raw_model1.layer3
        self.encoder_thermal_layer6 = resnet_raw_model1.layer4

        ########  RGB ENCODER  vis  ########
        self.encoder_rgb_conv1 = Feature_extract(3, 64)
        #self.encoder_rgb_bn1 = resnet_raw_model2.bn1
        #self.encoder_rgb_relu = resnet_raw_model2.relu
        self.encoder_rgb_maxpool = resnet_raw_model2.maxpool
        self.encoder_rgb_layer3 = resnet_raw_model2.layer1
        self.encoder_rgb_layer4 = resnet_raw_model2.layer2
        self.encoder_rgb_layer5 = resnet_raw_model2.layer3
        self.encoder_rgb_layer6 = resnet_raw_model2.layer4

        self.baseFeature3 = BaseFeatureExtraction(dim=64, num_heads=8)
        self.baseFeature4 = BaseFeatureExtraction(dim=128, num_heads=8)

        self.detailFeature2 = DetailFeatureExtraction(dim=64, num_layers=1)       #64
        self.detailFeature3 = DetailFeatureExtraction(dim=64, num_layers=1)       #64
        self.detailFeature4 = DetailFeatureExtraction(dim=128, num_layers=1)      #128

        self.high_fuse6 = PSFM(512, 64, 128)
        self.high_fuse5 = PSFM(256, 64, 128)
        self.low_fuse4 = SDFM(128, 64)
        self.low_fuse3 = SDFM(64, 64)
        self.low_fuse2 = SDFM(64, 64)
        self.low_fuse1 = SDFM(32, 32)
        self.detailBNR = ConvBNReLU(128, 64, 3, 1, 1)
        self.rec_decoder = DSRM(64, 64)
        self.rec_decoder1 = DSRM(32, 32)

        self.SIM3 = SIM(norm_nc=64, label_nc=64, nhidden=64)     # (4 64 64, 64)  (4 64 32,32)       SIM(norm_nc=64, label_nc=64, nhidden=32)  输出(4 64 128, 128)
        self.SIM2 = SIM(norm_nc=64, label_nc=64, nhidden=64)      # (4 64 128, 128)  (4 64 64, 64)    SIM(norm_nc=64, label_nc=64, nhidden=32)  输出(4 64 256, 256)
        self.SIM1 = SIM(norm_nc=32, label_nc=64, nhidden=32)      # (4 32 256, 256)  (4 64 128, 128)    SIM(norm_nc=32, label_nc=64, nhidden=32)  输出(4 32 256, 256)
        self.up2x = nn.UpsamplingBilinear2d(scale_factor=2)
        self.up4x = nn.UpsamplingBilinear2d(scale_factor=4)
        self.seg_decoder = S2PM(64, 64)

        self.binary_conv1 = ConvBNReLU(64, 64 // 2, kernel_size=1)              # ConvBNReLU(64, 16, kernel_size=1)
        self.binary_conv2 = nn.Conv2d(64 // 2, 2, kernel_size=3, padding=1)     # Conv2d(16, 2, kernel_size=3, padding=1)
        self.semantic_conv1 = ConvBNReLU(64, 32, kernel_size=1)                 # ConvBNReLU(32, 32, kernel_size=1)
        self.semantic_conv2 = nn.Conv2d(32, 9, kernel_size=3, padding=1)        # nn.Conv2d(64, 9, kernel_size=3, padding=1)
        self.boundary_conv = nn.Sequential(nn.Conv2d(128, 64, kernel_size=1), nn.BatchNorm2d(64), nn.ReLU6(inplace=True), nn.Conv2d(64, 2, kernel_size=3, padding=1))

        self.pred_vi = SFP([32, 1])
        self.pred_ir = SFP([32, 1])
        self.fusion = Restormer_Decoder(dim=32, out_channels=1)

    def forward(self, rgb, depth):
        rgb = rgb
        thermal = depth[:, :1, ...]       #thermal = depth 这个跟前面的没区别
        vobose = False        #原来是False
        # encoder
        if vobose: print("rgb.size() original: ", rgb.size())                #rgb.size() original:  torch.Size([4, 3, 256, 256])      (480, 640)
        if vobose: print("thermal.size() original: ", thermal.size())        #thermal.size() original:  torch.Size([4, 1, 256, 256])       (480, 640)

        rgb1, rgb2base = self.encoder_rgb_conv1(rgb)           #SFEB1:(4 32 256, 256)  SFEB2:(4 64 128, 128)
        #rgb2 = self.encoder_rgb_bn1(rgb2)                     #(4 64 128, 128)
        #rgb2 = self.encoder_rgb_relu(rgb2)                    #(4 64 128, 128)

        thermal1, thermal2base = self.encoder_thermal_conv1(thermal)      #SFEB1:(4 32 256, 256)  SFEB2:(4 64 128, 128)
        #thermal2 = self.encoder_thermal_bn1(thermal2)                #(4 64 128, 128)
        #thermal2 = self.encoder_thermal_relu(thermal2)               #(4 64 128, 128)

        rgb3 = self.encoder_rgb_maxpool(rgb2base)                         #(4 64 64, 64)
        thermal3 = self.encoder_thermal_maxpool(thermal2base)             #(4 64 64, 64)
        rgb3 = self.encoder_rgb_layer3(rgb3)                          #(4 64 64, 64)   在这后面加Tranformer
        thermal3 = self.encoder_thermal_layer3(thermal3)              #(4 64 64, 64)
        rgb3base = self.baseFeature3(rgb3)                            #自己加的   输出不变(4 64 64, 64)
        thermal3base = self.baseFeature3(thermal3)                    #自己加的   输出不变(4 64 64, 64)

        rgb4 = self.encoder_rgb_layer4(rgb3base)                     # (4 128 32, 32)
        thermal4 = self.encoder_thermal_layer4(thermal3base)         # (4 128 32, 32)
        rgb4base = self.baseFeature4(rgb4)                           # 自己加的   输出不变(4 128 32, 32)
        thermal4base = self.baseFeature4(thermal4)                   # 自己加的   输出不变(4 128 32, 32)

        rgb5 = self.encoder_rgb_layer5(rgb4base)                    # (4 256 16, 16)
        thermal5 = self.encoder_thermal_layer5(thermal4base)        # (4 256 16, 16)

        rgb6 = self.encoder_rgb_layer6(rgb5)                       # (4 512,8, 8)
        thermal6 = self.encoder_thermal_layer6(thermal5)           # (4 512,8, 8)

        #细节特征提
        rgb2detail = self.detailFeature2(rgb2base)                   # (4 64 128, 128)
        thermal2detail = self.detailFeature2(thermal2base)           # (4 64 128, 128)
        rgb3detail = self.detailFeature3(rgb3base)                   # (4 64 64, 64)
        thermal3detail = self.detailFeature3(thermal3base)           # (4 64 64，64)
        rgb4detail = self.detailFeature4(rgb4base)                   # (4 128 32，32)
        thermal4detail = self.detailFeature4(thermal4base)           # (4 128 32, 32)

        #特征融合块
        fused_f6 = self.high_fuse6(rgb6, thermal6)               # PSFM(512, 64, 128)    (4 64 8, 8)
        fused_f5 = self.high_fuse5(rgb5, thermal5)               # PSFM(256, 64, 128)    (4 64 16, 16)

        fused_f4 = self.detailFeature4(rgb4detail + thermal4detail)       #(4 128 32, 32)
        fused_f3 = self.detailFeature3(rgb3detail + thermal3detail)       #(4 64 64, 64)
        fused_f2 = self.detailFeature2(rgb2detail + thermal2detail)       #(4 64 128, 128)

        base4 = self.low_fuse4(rgb4base, thermal4base)         # SDFM(128, 64)   (4 64 32, 32)
        base3 = self.low_fuse3(rgb3base, thermal3base)         # SDFM(64, 64)   (4 64 64, 64)
        base2 = self.low_fuse2(rgb2base, thermal2base)         # SDFM(64, 64)   (4 64 128, 128)
        base1 = self.low_fuse1(rgb1, thermal1)                 # SDFM(32, 32)    (4 32 256, 256)

        detail5 = self.detailBNR(torch.cat((fused_f5, self.up2x(fused_f6)), dim=1))      # (4 64 16, 16)
        detail4 = self.detailBNR(torch.cat((self.detailBNR(fused_f4), self.up2x(detail5)), dim=1))       # (4 64 32, 32)
        detail3 = self.detailBNR(torch.cat((fused_f3, self.up2x(detail4)), dim=1))      # (4 64 64, 64)
        detail2 = self.detailBNR(torch.cat((fused_f2, self.up2x(detail3)), dim=1))      # (4 64,128, 128)

        refused4 = self.rec_decoder(base4 + detail4)           #DSRM(64, 64)  输出(4 64 32,32)
        refused3 = self.rec_decoder(base3 + detail3)           #DSRM(64, 64)  输出(4 64 64, 64)
        refused2 = self.rec_decoder(base2 + detail2)           #DSRM(64, 64)  输出(4 64 128,128)

        rec_f3 = self.SIM3(refused3, refused4)          # (4 64 64, 64)  (4 64 32,32)       SIM1(norm_nc=64, label_nc=64, nhidden=64)  输出(4 64 64, 64)
        rec_f2 = self.SIM2(refused2, rec_f3)            # (4 64 128, 128)  (4 64 64, 64)    SIM1(norm_nc=64, label_nc=64, nhidden=64)  输出(4 64 128,128)
        rec_f1 = self.SIM1(base1, rec_f2)               # (4 32 256, 256)  (4 64 128, 128)    SIM1(norm_nc=32, label_nc=64, nhidden=32)  输出(4 32 256, 256)
        rec = self.rec_decoder1(rec_f1)                 #输出(4 32 256, 256)

        #语义分割
        seg_1 = self.seg_decoder(detail2)       #self.seg_decoder = S2PM(64, 64)  输出：(4 64 128,128)
        binary = self.binary_conv2(self.binary_conv1(seg_1))         #先(4 32 128，128)， 再(32 2 128，128)  输出(4 2 128,128)
        binary_out = self.up2x(binary)      #输出(4 2 256,256)

        weight = torch.exp(binary)            #torch.Size([4, 2, 128, 128])
        weight = weight[:, 1:2, :, :] / torch.sum(weight, dim=1, keepdim=True)  #torch.Size([4, 1, 128, 128])

        seg_2 = self.up2x(self.seg_decoder(detail3))             #(4 64 64, 64)  self.seg_decoder = S2PM(64, 64)  输出：(4 64 128, 128)
        feat_sematic = self.semantic_conv1(seg_2 * weight)       #(4 32 128,128)
        semantic_out = self.up2x(self.semantic_conv2(feat_sematic))         # (4 9 256，256)

        seg_3 = self.up4x(self.seg_decoder(detail4))             #(4 64 32,32) self.seg_decoder = S2PM(64, 64)  输出：(4 64 128, 128)
        feat_boundary = torch.cat((seg_2, seg_3), dim=1)         #(4 128 128,128)
        boundary_out = self.up2x(self.boundary_conv(feat_boundary))          #(4 2 256, 256)

        vi_img = self.pred_vi(rec)
        ir_img = self.pred_ir(rec)
        fused_img = self.fusion(rec)
        return semantic_out, binary_out, boundary_out, fused_img, vi_img, ir_img


class SDFM(nn.Module):
    def __init__(self, in_C, out_C):  # (4 64 64, 64)  SDFM(64, 64)   (4 64 128, 128)  SDFM(64, 32)   (4 32 256, 256)  SDFM(32, 32)
        super(SDFM, self).__init__()
        self.RGBobj1_1 = BBasicConv2d(in_C, out_C, 3, 1, 1)
        self.RGBobj1_2 = BBasicConv2d(out_C, out_C, 3, 1, 1)
        self.RGBspr = BBasicConv2d(out_C, out_C, 3, 1, 1)

        self.Infobj1_1 = BBasicConv2d(in_C, out_C, 3, 1, 1)
        self.Infobj1_2 = BBasicConv2d(out_C, out_C, 3, 1, 1)
        self.Infspr = BBasicConv2d(out_C, out_C, 3, 1, 1)
        self.obj_fuse = Fusion_module(channels=out_C)  # channels=64， 32， 32

    def forward(self, rgb, depth):
        rgb_sum = self.RGBobj1_2(self.RGBobj1_1(rgb))  # (4 64 64, 64)  (4 64 128, 128)   (4 32 256, 256)
        rgb_obj = self.RGBspr(rgb_sum)  # (4 64 64, 64)  (4 64 128, 128)   (4 32 256, 256)
        Inf_sum = self.Infobj1_2(self.Infobj1_1(depth))  # (4 64 64, 64)  (4 64 128, 128)   (4 32 256, 256)
        Inf_obj = self.Infspr(Inf_sum)  # (4 64 64, 64)  (4 64 128, 128)   (4 32 256, 256)
        out = self.obj_fuse(rgb_obj, Inf_obj)  # (4 64 64, 64)  (4 64 128, 128)   (4 32 256, 256)   channels=64， 32， 32
        return out


class Fusion_module(nn.Module):  # 基于注意力的自适应特征聚合 Fusion_Module
    def __init__(self, channels=64, r=4):  # channels=64，32，32
        super(Fusion_module, self).__init__()
        inter_channels = int(channels // r)  # 16  8  8
        self.Recalibrate = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(2 * channels, 2 * inter_channels, kernel_size=1, stride=1, padding=0), nn.BatchNorm2d(2 * inter_channels),
                                         nn.ReLU(inplace=True), nn.Conv2d(2 * inter_channels, 2 * channels, kernel_size=1, stride=1, padding=0), nn.BatchNorm2d(2 * channels), nn.Sigmoid())
        # nn.Sequential(nn.AdaptiveAvgPool2d(1), Conv2d(128, 32, 1, 1, 0),BatchNorm2d(32), ReLU(inplace=True),Conv2d(32, 128,1,1,0),BatchNorm2d(32),Sigmoid())
        self.channel_agg = nn.Sequential(nn.Conv2d(2 * channels, channels, kernel_size=1, stride=1, padding=0), nn.BatchNorm2d(channels), nn.ReLU(inplace=True))
        self.local_att = nn.Sequential(nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0), nn.BatchNorm2d(inter_channels), nn.ReLU(inplace=True),
                                       nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0), nn.BatchNorm2d(channels))
        # nn.Sequential(nn.Conv2d(64,16,1,1,0),BatchNorm2d(16s),.ReLU(inplace=True),Conv2d(16,64,1,1,0),BatchNorm2d(64))
        self.global_att = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0), nn.BatchNorm2d(inter_channels), nn.ReLU(inplace=True),
                                        nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0), nn.BatchNorm2d(channels))
        # nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(64,16,1,1,0), BatchNorm2d(16),ReLU(inplace=True),Conv2d(16,64,1,1,0),BatchNorm2d(64))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x1, x2):  # (4 64 64, 64)  (4 64 128, 128)   (4 32 256, 256)
        _, c, _, _ = x1.shape  # 64  32   32
        input = torch.cat([x1, x2], dim=1)  # (4 128 64, 64)  (4 128 128, 128)   (4 64 256, 256)  C拼接
        recal_w = self.Recalibrate(input)  # (4 128 1, 1)  (4 128，1, 1)   (4 64 1,1)
        recal_input = recal_w * input  # 先对特征进行一步自校正  #(4 128 64, 64)  (4 128 128, 128)   (4 64 256, 256)
        recal_input = recal_input + input  # (4 128 64, 64)  (4 128 128, 128)   (4 64 256, 256)
        x1, x2 = torch.split(recal_input, c, dim=1)  # (4 64 64, 64)  (4 64 128, 128)   (4 32 256, 256)
        agg_input = self.channel_agg(recal_input)  # 进行特征压缩 因为只计算一个特征的权重    (4 64 64, 64)  (4 64 128, 128)   (4 32 256, 256)
        local_w = self.local_att(agg_input)  # 局部注意力 即spatial attention   (4 64 64, 64)  (4 64 128, 128)   (4 32 256, 256)
        global_w = self.global_att(agg_input)  # 全局注意力 即channel attention  (4,64,1,1)  (4,64,1,1)   (4,32,1,1)
        w = self.sigmoid(local_w * global_w)  # 计算特征x1的权重  (4 64 64, 64)  (4 64 128, 128)   (4 32 256, 256)
        xo = w * x1 + (1 - w) * x2  # fusion results 特征聚合  (4 64 64, 64)  (4 64 128, 128)   (4 32 256, 256)
        return xo


class GEFM(nn.Module):
    def __init__(self, in_C, out_C):  # GEFM(128, 64)
        super(GEFM, self).__init__()
        self.RGB_K = BBasicConv2d(out_C, out_C, 3, 1, 1)
        self.RGB_V = BBasicConv2d(out_C, out_C, 3, 1, 1)
        self.Q = BBasicConv2d(in_C, out_C, 3, 1, 1)
        self.INF_K = BBasicConv2d(out_C, out_C, 3, 1, 1)
        self.INF_V = BBasicConv2d(out_C, out_C, 3, 1, 1)
        self.Second_reduce = BBasicConv2d(in_C, out_C, 3, 1, 1)
        self.gamma1 = nn.Parameter(torch.zeros(1))
        self.gamma2 = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, y):  # torch.Size([4, 64, 8, 8])  torch.Size([4, 64, 16, 16])   torch.Size([4, 64, 32, 32])
        Q = self.Q(torch.cat([x, y], dim=1))  # torch.Size([4, 64, 8, 8])  torch.Size([4, 64, 16, 16])   torch.Size([4, 64, 32, 32])  BBasicConv2d(128, 64, 3, 1, 1)
        RGB_K = self.RGB_K(x)  # BBasicConv2d(64, 64, 3, 1, 1)  torch.Size([4, 128, 8, 8])
        RGB_V = self.RGB_V(x)  # BBasicConv2d(64, 64, 3, 1, 1)  torch.Size([4, 128, 8, 8])
        m_batchsize, C, height, width = RGB_V.size()  # torch.Size([4, 64, 8, 8])  torch.Size([4, 64, 16, 16])  torch.Size([4, 64, 32, 32])

        RGB_V = RGB_V.view(m_batchsize, -1, width * height)  # torch.Size([4, 64, 64])   torch.Size([4, 64, 256])   torch.Size([4, 64, 1024])  view中一个参数定为 - 1，代表自动调整这个维度上的元素个数，以保证元素的总数不变。
        RGB_K = RGB_K.view(m_batchsize, -1, width * height).permute(0, 2,
                                                                    1)  # torch.Size([4, 64, 64]) torch.Size([4, 256，64])   torch.Size([4,  1024， 64])
        RGB_Q = Q.view(m_batchsize, -1, width * height)  # torch.Size([4, 64, 64])   torch.Size([4, 64, 256])   torch.Size([4, 64, 1024])
        RGB_mask = torch.bmm(RGB_K, RGB_Q)  # 矩阵乘法torch.Size([4, 64, 64])   torch.Size([4, 256, 256])   torch.Size([4, 1024, 1024])
        RGB_mask = self.softmax(RGB_mask)  # torch.Size([4, 64, 64])   torch.Size([4, 256, 256])   torch.Size([4, 1024, 1024])
        RGB_refine = torch.bmm(RGB_V, RGB_mask.permute(0, 2, 1))  # torch.Size([4, 64, 64])  torch.Size([4, 64, 256])  torch.Size([4, 64, 1024])
        RGB_refine = RGB_refine.view(m_batchsize, -1, height, width)  # torch.Size([4, 64, 8, 8])  torch.Size([4, 64, 16, 16])   torch.Size([4, 64, 32, 32])
        RGB_refine = self.gamma1 * RGB_refine + y

        INF_K = self.INF_K(y)
        INF_V = self.INF_V(y)
        INF_V = INF_V.view(m_batchsize, -1, width * height)
        INF_K = INF_K.view(m_batchsize, -1, width * height).permute(0, 2, 1)
        INF_Q = Q.view(m_batchsize, -1, width * height)
        INF_mask = torch.bmm(INF_K, INF_Q)
        INF_mask = self.softmax(INF_mask)
        INF_refine = torch.bmm(INF_V, INF_mask.permute(0, 2, 1))
        INF_refine = INF_refine.view(m_batchsize, -1, height, width)          # torch.Size([4, 64, 8, 8])  torch.Size([4, 64, 16, 16])   torch.Size([4, 64, 32, 32])
        INF_refine = self.gamma2 * INF_refine + x

        out = self.Second_reduce(torch.cat([RGB_refine, INF_refine], dim=1))      # torch.Size([4, 64, 8, 8])  torch.Size([4, 64, 16, 16])   torch.Size([4, 64, 32, 32])
        return out


class PSFM(nn.Module):
    def __init__(self, in_C, out_C, cat_C):  # PSFM(512, 64, 128)   PSFM(64, 64, 128)
        super(PSFM, self).__init__()
        self.RGBobj = DenseLayer(in_C, out_C)  # DenseLayer(512, 64)
        self.Infobj = DenseLayer(in_C, out_C)
        self.obj_fuse = GEFM(cat_C, out_C)

    def forward(self, rgb, depth):
        rgb_sum = self.RGBobj(rgb)     # torch.Size([4, 64, 8, 8])  torch.Size([4, 64, 16, 16])   torch.Size([4, 64, 32, 32])
        Inf_sum = self.Infobj(depth)  # torch.Size([4, 64, 8, 8])  torch.Size([4, 64, 16, 16])   torch.Size([4, 64, 32, 32])
        out = self.obj_fuse(rgb_sum, Inf_sum)  # GEFM(128, 64)   torch.Size([4, 64, 8, 8])  torch.Size([4, 64, 16, 16])   torch.Size([4, 64, 32, 32])
        return out


class DenseLayer(nn.Module):  # 更像是DenseNet的Block，从而构造特征内的密集连接
    def __init__(self, in_C, out_C, down_factor=4, k=4):  # DenseLayer(512, 64)
        super(DenseLayer, self).__init__()
        self.k = k  # 4
        self.down_factor = down_factor  # 4
        mid_C = out_C // self.down_factor  # 64/4=16
        self.down = nn.Conv2d(in_C, mid_C, 1)  # Conv2d(512, 16, 1)
        self.denseblock = nn.ModuleList()
        for i in range(1, self.k + 1):  # BBasicConv2d(16,16, 3, 1, 1) BBasicConv2d(32, 16, 3, 1, 1)  BBasicConv2d(48, 16, 3, 1, 1)   BBasicConv2d(64, 16, 3, 1, 1)
            self.denseblock.append(BBasicConv2d(mid_C * i, mid_C, 3, 1, 1))

        self.fuse = BBasicConv2d(in_C + mid_C, out_C, kernel_size=3, stride=1, padding=1)  # BBasicConv2d(512 + 16, 64, 3, 1, 1)

    def forward(self, in_feat):  # (4 512,8, 8)
        down_feats = self.down(in_feat)  # torch.Size([4, 16, 8, 8])  torch.Size([4, 16, 16, 16])   torch.Size([4, 16, 32, 32])
        out_feats = []
        for i in self.denseblock:
            feats = i(torch.cat((*out_feats, down_feats), dim=1))  # 这句代码还是没搞懂！！！！
            out_feats.append(feats)
        # feats.shape: torch.Size([4, 16, 8, 8])
        feats = torch.cat((in_feat, feats), dim=1)  # torch.Size([4, 528, 8, 8])  ([4, 272, 16, 16])   ([4, 144, 32, 32])
        return self.fuse(feats)


class BBasicConv2d(nn.Module):  # CBR
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False):
        super(BBasicConv2d, self).__init__()
        self.basicconv = nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias), nn.BatchNorm2d(out_planes), nn.ReLU(inplace=True))

    def forward(self, x):
        return self.basicconv(x)


#########################################################################################################    Inception
class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
                 bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU6(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


#########################################################################################################     decoder
class decoder(nn.Module):
    def __init__(self, channel=64):
        super(decoder, self).__init__()
        self.block1 = nn.Sequential(BasicConv2d(channel, channel, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(channel), nn.ReLU(inplace=True))
        self.block2 = nn.Sequential(BasicConv2d(channel, channel, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(channel), nn.ReLU(inplace=True))
        self.block3 = nn.Sequential(BasicConv2d(channel, channel, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(channel), nn.ReLU(inplace=True))
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x):
        x1 = self.block1(x)
        x2 = self.block2(x1)
        x3 = self.block3(x2)
        out = x3 + x
        out = self.up2(out)
        return out


class S2PM(nn.Module):  # (4 256 64, 64)  S2PM(4 * 64, 64)
    def __init__(self, in_channel=64, out_channel=64):
        super(S2PM, self).__init__()
        self.block1 = nn.Sequential(BasicConv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1), nn.ReLU(inplace=True))
        self.block2 = nn.Sequential(BasicConv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1), nn.ReLU(inplace=True))
        self.block3 = nn.Sequential(BasicConv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1), nn.ReLU(inplace=True))

    def forward(self, x):  # (4 256 64, 64)
        x1 = self.block1(x)  # (4 64 64, 64)
        x2 = self.block2(x1)  # (4 64 64, 64)
        out = self.block3(x2)  # (4 64 64, 64)
        return out


'''
class DSRM3(nn.Module):    #  DSRM(128, 64)  (4 128 64, 64)
    def __init__(self, in_channel, out_channel):
        super(DSRM3, self).__init__()
        self.block1 = nn.Sequential(BasicConv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(out_channel), nn.ReLU(inplace=True))
        self.block2 = nn.Sequential(BasicConv2d(192, out_channel, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(out_channel), nn.ReLU(inplace=True))
        self.block3 = nn.Sequential(BasicConv2d(2 * in_channel, out_channel, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(out_channel), nn.ReLU(inplace=True))
        self.block4 = nn.Sequential(BasicConv2d(320, out_channel, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(out_channel), nn.ReLU(inplace=True))

    def forward(self, x):      #(4 128 64, 64)
        x1 = self.block1(x)      #(4 64 64, 64)
        x2 = self.block2(torch.cat([x, x1], dim=1))       #输入(4 192 64, 64)  输出：(4 64 64, 64)
        x3 = self.block3(torch.cat([x, x1, x2], dim=1))      #输入(4 256 64, 64)  输出：(4 64 64, 64)
        out = self.block4(torch.cat([x, x1, x2, x3], dim=1))        #(4 32 256, 256)
        return out

class DSRM2(nn.Module):    #  DSRM2(64, 32)  (4 64 256, 256)
    def __init__(self, in_channel, out_channel):
        super(DSRM2, self).__init__()
        self.block1 = nn.Sequential(BasicConv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(out_channel), nn.ReLU(inplace=True))
        self.block2 = nn.Sequential(BasicConv2d(96, out_channel, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(out_channel), nn.ReLU(inplace=True))
        self.block3 = nn.Sequential(BasicConv2d(2 * in_channel, out_channel, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(out_channel), nn.ReLU(inplace=True))
        self.block4 = nn.Sequential(BasicConv2d(160, out_channel, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(out_channel), nn.ReLU(inplace=True))

    def forward(self, x):      #(4 64 256, 256)
        x1 = self.block1(x)      #(4 32 256, 256)
        x2 = self.block2(torch.cat([x, x1], dim=1))       #输入(4 96 256, 256)  输出：(4 32 256, 256)
        x3 = self.block3(torch.cat([x, x1, x2], dim=1))      #输入(4 128 256, 256)  输出：(4 64 256, 256)
        out = self.block4(torch.cat([x, x1, x2, x3], dim=1))        #(4 32 256, 256)
        return out
'''


class DSRM(nn.Module):  # DSRM(64, 64)  (4 64 64, 64)
    def __init__(self, in_channel, out_channel):
        super(DSRM, self).__init__()
        #self.block1 = nn.Sequential(BasicConv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(out_channel), nn.ReLU(inplace=True))
        #self.block2 = nn.Sequential(BasicConv2d(2 * in_channel, out_channel, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(out_channel), nn.ReLU(inplace=True))
        #self.block3 = nn.Sequential(BasicConv2d(3 * in_channel, out_channel, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(out_channel), nn.ReLU(inplace=True))
        #self.block4 = nn.Sequential(BasicConv2d(4 * in_channel, out_channel, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(out_channel), nn.ReLU(inplace=True))
        self.block1 = nn.Sequential(BasicConv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1), nn.ReLU(inplace=True))
        self.block2 = nn.Sequential(BasicConv2d(2 * in_channel, out_channel, kernel_size=3, stride=1, padding=1), nn.ReLU(inplace=True))
        self.block3 = nn.Sequential(BasicConv2d(3 * in_channel, out_channel, kernel_size=3, stride=1, padding=1), nn.ReLU(inplace=True))
        self.block4 = nn.Sequential(BasicConv2d(4 * in_channel, out_channel, kernel_size=3, stride=1, padding=1), nn.ReLU(inplace=True))

    def forward(self, x):  # (4 64 64, 64)
        x1 = self.block1(x)  # (4 64 64, 64)
        x2 = self.block2(torch.cat([x, x1], dim=1))  # 输入(4 128 64, 64)  输出：(4 64 64, 64)
        x3 = self.block3(torch.cat([x, x1, x2], dim=1))  # 输入(4 192 64, 64)  输出：(4 64 64, 64)
        out = self.block4(torch.cat([x, x1, x2, x3], dim=1))  # (4 64 256, 256)
        return out


class S2P2(nn.Module):  # S2P2(feature=64, n_classes=9)语义分割 This path plays the role of a classifier and is responsible for predicting the results of semantic segmentation, binary segmentation and edge segmentation
    def __init__(self, feature=64, n_classes=9):
        super(S2P2, self).__init__()
        self.binary_conv1 = ConvBNReLU(feature, feature // 4, kernel_size=1)  # ConvBNReLU(64, 16, kernel_size=1)
        self.binary_conv2 = nn.Conv2d(feature // 4, 2, kernel_size=3, padding=1)  # Conv2d(16, 2, kernel_size=3, padding=1)

        self.semantic_conv1 = ConvBNReLU(feature, feature, kernel_size=1)  # ConvBNReLU(64, 64, kernel_size=1)
        self.semantic_conv2 = nn.Conv2d(feature, n_classes, kernel_size=3, padding=1)  # nn.Conv2d(64, 9, kernel_size=3, padding=1)

        self.boundary_conv1 = ConvBNReLU(feature * 2, feature, kernel_size=1)  # ConvBNReLU(64 * 2, 64, kernel_size=1)
        self.boundary_conv2 = nn.Conv2d(feature, 2, kernel_size=3, padding=1)  # nn.Conv2d(64, 2, kernel_size=3, padding=1)
        # nn.Sequential(nn.Conv2d(64 * 2, 64, kernel_size=1), nn.BatchNorm2d(64), nn.ReLU6(inplace=True), nn.Conv2d(64, 2, kernel_size=3, padding=1))
        self.boundary_conv = nn.Sequential(nn.Conv2d(feature * 2, feature, kernel_size=1), nn.BatchNorm2d(feature), nn.ReLU6(inplace=True), nn.Conv2d(feature, 2, kernel_size=3, padding=1))

        self.up4x = nn.UpsamplingBilinear2d(scale_factor=4)
        self.up2x = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, feat):      # (4 64 64, 64)
        binary = self.binary_conv2(self.binary_conv1(feat))     # (4 2 64, 64)
        binary_out = self.up4x(binary)                          # torch.Size([4, 2, 256, 256])

        weight = torch.exp(binary)          #作用是对输入的张量进行按元素指数运算。torch.Size([4, 2, 64, 64])
        weight = weight[:, 1:2, :, :] / torch.sum(weight, dim=1, keepdim=True)  # torch.Size([4, 1, 64, 64])

        feat_sematic = self.up2x(feat * weight)               # (4 64 128, 128)
        feat_sematic = self.semantic_conv1(feat_sematic)      # (4 64 128, 128)

        semantic_out = self.semantic_conv2(feat_sematic)      # (4 9 128, 128)
        semantic_out = self.up2x(semantic_out)                # (4 9 256, 256)

        feat_boundary = torch.cat([feat_sematic, self.up2x(feat)], dim=1)      # (4 128 128, 128)
        boundary_out = self.boundary_conv(feat_boundary)       # (4 2 128, 128)
        boundary_out = self.up2x(boundary_out)                 # (4 2 256, 256)
        return semantic_out, binary_out, boundary_out


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, dilation=1, groups=1, bn=True, relu=True):
        padding = ((kernel_size - 1) * dilation + 1) // 2          # padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, dilation=dilation, groups=groups, bias=False if bn else True)
        self.bn = bn
        if bn:
            self.bnop = nn.BatchNorm2d(out_planes)
        self.relu = relu
        if relu:
            self.reluop = nn.ReLU6(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.bn:
            x = self.bnop(x)
        if self.relu:
            x = self.reluop(x)
        return x


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        #self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x
