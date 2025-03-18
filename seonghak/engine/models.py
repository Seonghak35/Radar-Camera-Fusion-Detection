import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision.ops import DeformConv2d
from ultralytics import YOLO

# ### Reference yolov8 ###

# def autopad(k, p=None, d=1):  
#     if d > 1:
#         # actual kernel-size
#         k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
#     if p is None:
#         # auto-pad
#         p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
#     return p

# class SiLU(nn.Module):  
#     @staticmethod
#     def forward(x):
#         return x * torch.sigmoid(x)

# class Conv(nn.Module):
#     default_act = SiLU() 
#     def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
#         super().__init__()
#         self.conv   = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
#         self.bn     = nn.BatchNorm2d(c2, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
#         self.act    = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

#     def forward(self, x):
#         return self.act(self.bn(self.conv(x)))

#     def forward_fuse(self, x):
#         return self.act(self.conv(x))

# class Bottleneck(nn.Module):
#     def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
#         super().__init__()
#         c_ = int(c2 * e)  # hidden channels
#         self.cv1 = Conv(c1, c_, k[0], 1)
#         self.cv2 = Conv(c_, c2, k[1], 1, g=g)
#         self.add = shortcut and c1 == c2

#     def forward(self, x):
#         return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))    
    
# class C2f(nn.Module):
#     def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
#         super().__init__()
#         self.c      = int(c2 * e) 
#         self.cv1    = Conv(c1, 2 * self.c, 1, 1)
#         self.cv2    = Conv((2 + n) * self.c, c2, 1)
#         self.m      = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

#     def forward(self, x):
#         y = list(self.cv1(x).split((self.c, self.c), 1))
#         y.extend(m(y[-1]) for m in self.m)
#         return self.cv2(torch.cat(y, 1))
    
# class SPPF(nn.Module):
#     def __init__(self, c1, c2, k=5):
#         super().__init__()
#         c_          = c1 // 2
#         self.cv1    = Conv(c1, c_, 1, 1)
#         self.cv2    = Conv(c_ * 4, c2, 1, 1)
#         self.m      = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

#     def forward(self, x):
#         x = self.cv1(x)
#         y1 = self.m(x)
#         y2 = self.m(y1)
#         return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))

# class Backbone(nn.Module):
#     def __init__(self, base_channels, base_depth, deep_mul, phi, pretrained=False):
#         super().__init__()
#         # 3, 640, 640 => 32, 640, 640 => 64, 320, 320
#         self.stem = Conv(3, base_channels, 3, 2)
        
#         # 64, 320, 320 => 128, 160, 160 => 128, 160, 160
#         self.dark2 = nn.Sequential(
#             Conv(base_channels, base_channels * 2, 3, 2),
#             C2f(base_channels * 2, base_channels * 2, base_depth, True),
#         )
#         # 128, 160, 160 => 256, 80, 80 => 256, 80, 80
#         self.dark3 = nn.Sequential(
#             Conv(base_channels * 2, base_channels * 4, 3, 2),
#             C2f(base_channels * 4, base_channels * 4, base_depth * 2, True),
#         )
#         # 256, 80, 80 => 512, 40, 40 => 512, 40, 40
#         self.dark4 = nn.Sequential(
#             Conv(base_channels * 4, base_channels * 8, 3, 2),
#             C2f(base_channels * 8, base_channels * 8, base_depth * 2, True),
#         )
#         # 512, 40, 40 => 1024 * deep_mul, 20, 20 => 1024 * deep_mul, 20, 20
#         self.dark5 = nn.Sequential(
#             Conv(base_channels * 8, int(base_channels * 16 * deep_mul), 3, 2),
#             C2f(int(base_channels * 16 * deep_mul), int(base_channels * 16 * deep_mul), base_depth, True),
#             SPPF(int(base_channels * 16 * deep_mul), int(base_channels * 16 * deep_mul), k=5)
#         )
        
#         if pretrained:
#             url = {
#                 "n" : 'https://github.com/bubbliiiing/yolov8-pytorch/releases/download/v1.0/yolov8_n_backbone_weights.pth',
#                 "s" : 'https://github.com/bubbliiiing/yolov8-pytorch/releases/download/v1.0/yolov8_s_backbone_weights.pth',
#                 "m" : 'https://github.com/bubbliiiing/yolov8-pytorch/releases/download/v1.0/yolov8_m_backbone_weights.pth',
#                 "l" : 'https://github.com/bubbliiiing/yolov8-pytorch/releases/download/v1.0/yolov8_l_backbone_weights.pth',
#                 "x" : 'https://github.com/bubbliiiing/yolov8-pytorch/releases/download/v1.0/yolov8_x_backbone_weights.pth',
#             }[phi]
#             checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", model_dir="./model_data")
#             self.load_state_dict(checkpoint, strict=False)
#             print("Load weights from " + url.split('/')[-1])

#     def forward(self, x):
#         x = self.stem(x)
#         x = self.dark2(x)

#         x = self.dark3(x)
#         feat1 = x

#         x = self.dark4(x)
#         feat2 = x

#         x = self.dark5(x)
#         feat3 = x
#         return feat1, feat2, feat3


# ### Reference yolov8 ###


# ✅ CSP Block 정의
class CSPBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers=1, expansion=0.5, downsample=False):
        super(CSPBlock, self).__init__()
        hidden_channels = int(out_channels * expansion)

        self.conv1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=1, stride=2 if downsample else 1, bias=False)
        self.conv2 = nn.Conv2d(in_channels, hidden_channels, kernel_size=1, stride=2 if downsample else 1, bias=False)

        self.bottlenecks = nn.Sequential(*[
            nn.Sequential(
                nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(hidden_channels),
                nn.SiLU()
            ) for _ in range(num_layers)
        ])     

        self.final_conv = nn.Sequential(
            nn.Conv2d(hidden_channels * 2, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU()
        )

    def forward(self, x):
        y1 = self.conv1(x)
        y2 = self.conv2(x)
        y1 = self.bottlenecks(y1)
        y = torch.cat([y1, y2], dim=1)
        return self.final_conv(y)
    

# ✅ Shuffle Attention 정의
class ShuffleAttention(nn.Module):
    def __init__(self, channels, groups=2):
        super(ShuffleAttention, self).__init__()
        self.groups = groups

        # channel attention
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

        # spatial attention
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=7, padding=3, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.Sigmoid()
        )

    def channel_shuffle(self, x):
        B, C, H, W = x.shape
        x = x.view(B, self.groups, C // self.groups, H, W).transpose(1,2).reshape(B,C,H,W)
        return x

    def forward(self, x):
        x = self.channel_shuffle(x)
        channel_att = self.channel_attention(x)
        spatial_att = self.spatial_attention(x)
        return x * channel_att * spatial_att
    

class RadarCameraYOLO(nn.Module):
    def __init__(self, num_classes=7):
        super(RadarCameraYOLO, self).__init__()

        # Camera Feature Extractor (CSP)
        self.camera_stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.SiLU(),
            CSPBlock(64, 128, num_layers=3)
        )

        # Radar Feature Extractor
        self.radar_pool = nn.AvgPool2d(3, stride=1, padding=1)
        self.radar_deform_conv = DeformConv2d(4, 64, kernel_size=3, stride=2, padding=1)
        self.radar_bn = nn.BatchNorm2d(64)
        self.radar_attention = ShuffleAttention(64)

        # Adaptive fusion weight α
        self.alpha = nn.Parameter(torch.rand(1))

        # channel mathcing
        self.fusion_conv = nn.Sequential(
        nn.Conv2d(128 + 64, 128, kernel_size=3, stride=2, padding=1, bias=False),  
        nn.BatchNorm2d(128),
        nn.SiLU()
        )
        self.backbone_downsample = nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1, bias=False)

        # YOLO Backbone (CSPDarknet)
        self.yolo_backbone = nn.Sequential(
            CSPBlock(128, 256, num_layers=3, downsample=True),
            CSPBlock(256, 512, num_layers=3, downsample=False),
            CSPBlock(512, 1024, num_layers=1, downsample=False)
        )

        # FPN Neck (Feature Pyramid Network)
        self.yolo_neck = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=1, bias=False),
            nn.BatchNorm2d(512),
            nn.SiLU(),
            nn.Conv2d(512, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.SiLU()
        )

        # YOLO Decoupled Head
        self.yolo_head_cls = nn.Conv2d(256, 7, kernel_size=1)
        self.yolo_head_reg = nn.Conv2d(256, 4, kernel_size=1)
        self.yolo_head_obj = nn.Conv2d(256, 1, kernel_size=1)  # Objectness Score 추가

    def forward(self, camera, radar):
        # Camera Feature Extraction
        F_camera = self.camera_stem(camera)

        # Radar Feature Extration
        radar_pooled = self.radar_pool(radar)

        stride_factor = self.radar_deform_conv.stride[0] if isinstance(self.radar_deform_conv.stride, tuple) else self.radar_deform_conv.stride
        offset_h = radar.size(2) // stride_factor
        offset_w = radar.size(3) // stride_factor
        offset = torch.zeros((radar.size(0), 18, offset_h, offset_w), device=radar.device)

        radar_feature = torch.relu(self.radar_deform_conv(radar_pooled, offset))
        radar_feature = self.radar_bn(radar_feature)
        radar_feature = self.radar_attention(radar_feature)

        # Adaptive Fusion
        fusion_feature = torch.cat([F_camera, self.alpha * radar_feature], dim=1)
        fusion_feature = self.fusion_conv(fusion_feature)

        # YOLO Backbone
        yolo_feature = self.yolo_backbone(fusion_feature)
        neck_feature = self.yolo_neck(yolo_feature)

        # YOLO Head Outputs
        class_output = torch.sigmoid(self.yolo_head_cls(neck_feature))  # (B, num_classes, H, W)
        bbox_output = torch.sigmoid(self.yolo_head_reg(neck_feature))   # (B, 4, H, W)
        obj_output = torch.sigmoid(self.yolo_head_obj(neck_feature))    # (B, 1, H, W)

        # Shape 변환: (B, num_classes + 5, H, W) -> (B, N, num_classes + 5)
        B, C, H, W = class_output.shape
        N = H * W  # N: HxW 개수의 anchor points
        
        class_output = class_output.view(B, 7, N).permute(0, 2, 1)  # (B, N, num_classes)
        bbox_output = bbox_output.view(B, 4, N).permute(0, 2, 1)  # (B, N, 4)
        obj_output = obj_output.view(B, 1, N).permute(0, 2, 1)  # (B, N, 1)

        # 최종 YOLO 형식의 출력 (B, N, num_classes + 5)
        yolo_output = torch.cat([bbox_output, obj_output, class_output], dim=2)  # (B, N, num_classes + 5)

        return yolo_output
    

# ✅ Dynamic Collate Function (YOLO 바운딩 박스 개수 다름 문제 해결)
def yolo_collate_fn(batch):
    cameras = []
    radars = []
    labels = []

    for camera, radar, label in batch:
        cameras.append(camera)
        radars.append(radar)
        labels.append(label)  # ✅ 리스트로 유지 (Tensor 변환 X)

    # ✅ 이미지 및 레이더 데이터를 스택
    cameras = torch.stack(cameras, dim=0)
    radars = torch.stack(radars, dim=0)

    return cameras, radars, labels  # ✅ `labels`은 리스트로 유지


### Achelous3T ###

image_encoder_width = {
    'L': [40, 80, 192, 384],  # 26m 83.3% 6attn
    'S2': [32, 64, 144, 288],  # 12m 81.6% 4attn dp0.02
    'S1': [32, 48, 120, 224],  # 6.1m 79.0
    'S0': [32, 48, 96, 176],  # 75.0 75.7
}

class DWConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, bias=True):
        super().__init__()
        self.dconv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size,
                               stride=stride, groups=in_channels, padding=padding, dilation=dilation, bias=bias)
        self.pconv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, groups=1,
                               bias=bias)

    def forward(self, x):
        x = self.dconv(x)
        return self.pconv(x)
    
def get_activation(name="silu", inplace=True):
    if name == "silu":
        module = SiLU()
    elif name == "relu":
        module = nn.ReLU(inplace=inplace)
    elif name == "lrelu":
        module = nn.LeakyReLU(0.1, inplace=inplace)
    else:
        raise AttributeError("Unsupported act type: {}".format(name))
    return module

class BaseConv(nn.Module):
    def __init__(self, in_channels, out_channels, ksize, stride, groups=1, bias=False, act="relu", ds_conv=False):
        super().__init__()
        pad         = (ksize - 1) // 2
        if ds_conv is False:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=ksize, stride=stride, padding=pad,
                                  groups=groups, bias=bias)
        else:
            self.conv = DWConv(in_channels, out_channels, kernel_size=ksize, stride=stride, padding=pad, bias=bias)
        self.bn     = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.03)
        self.act    = get_activation(act, inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))

class RadarCamera(nn.Module):
    def __init__(self, num_det, num_seg, phi='SO', image_channels=3, radar_channels=3, resolution=320,
                 backbone='en', neck='gdf', pc_seg='pn', pc_channels=6, pc_classes=9, nano_head=True, spp=True):
        super(RadarCamera, self).__init__()

        self.num_det = num_det
        self.num_seg = num_seg
        self.resolution = resolution

        self.phi = phi
        self.image_channels = image_channels
        self.radar_channels = radar_channels

        self.image_radar_encoder = IREncoder(num_class_seg=num_seg, resolution=resolution, backbone=backbone, neck=neck,
                                             phi=phi, use_spp=spp)
        self.det_head = DecoupleHead(num_classes=num_det, phi=phi, nano_head=nano_head)

    def forward(self, x, x_radar):
        fpn_out, se_seg_output, lane_seg_output = self.image_radar_encoder(x, x_radar)
        det_output = self.det_head(fpn_out)
        return det_output, se_seg_output, lane_seg_output

class DecoupleHead(nn.Module):
    def __init__(self, num_classes, width=1.0, phi='S0', act="relu", depthwise=True, nano_head=True):
        super().__init__()
        Conv = BaseConv

        in_channels = [channels*5//4 for channels in image_encoder_width[phi][1:]]

        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.cls_preds = nn.ModuleList()
        self.reg_preds = nn.ModuleList()
        self.obj_preds = nn.ModuleList()
        self.stems = nn.ModuleList()

        if nano_head is True:
            base_num = 64
        else:
            base_num = 256

        for i in range(len(in_channels)):
            self.stems.append(
                Conv(in_channels=int(in_channels[i] * width), out_channels=int(base_num * width), ksize=1, stride=1,
                         act=act))
            self.cls_convs.append(nn.Sequential(*[
                Conv(in_channels=int(base_num * width), out_channels=int(base_num * width), ksize=5, stride=1, act=act, ds_conv=depthwise),
                Conv(in_channels=int(base_num * width), out_channels=int(base_num * width), ksize=5, stride=1, act=act, ds_conv=depthwise),
            ]))
            self.cls_preds.append(
                nn.Conv2d(in_channels=int(base_num * width), out_channels=num_classes, kernel_size=1, stride=1, padding=0)
            )

            self.reg_convs.append(nn.Sequential(*[
                Conv(in_channels=int(base_num * width), out_channels=int(base_num * width), ksize=5, stride=1, act=act, ds_conv=depthwise),
                Conv(in_channels=int(base_num * width), out_channels=int(base_num * width), ksize=5, stride=1, act=act, ds_conv=depthwise)
            ]))
            self.reg_preds.append(
                nn.Conv2d(in_channels=int(base_num * width), out_channels=4, kernel_size=1, stride=1, padding=0)
            )
            self.obj_preds.append(
                nn.Conv2d(in_channels=int(base_num * width), out_channels=1, kernel_size=1, stride=1, padding=0)
            )

    def forward(self, inputs):
        # ---------------------------------------------------#
        #   inputs
        #   P3_out  80, 80, 256
        #   P4_out  40, 40, 512
        #   P5_out  20, 20, 1024
        # ---------------------------------------------------#
        outputs = []
        for k, x in enumerate(inputs):
            x = self.stems[k](x)
            cls_feat = self.cls_convs[k](x)
            # ---------------------------------------------------#
            #   80, 80, num_classes
            #   40, 40, num_classes
            #   20, 20, num_classes
            # ---------------------------------------------------#
            cls_output = self.cls_preds[k](cls_feat)
            reg_feat = self.reg_convs[k](x)
            # ---------------------------------------------------#
            #   reg_pred 80, 80, 4
            #   reg_pred 40, 40, 4
            #   reg_pred 20, 20, 4
            # ---------------------------------------------------#
            reg_output = self.reg_preds[k](reg_feat)
            # ---------------------------------------------------#
            #   obj_pred 80, 80, 1
            #   obj_pred 40, 40, 1
            #   obj_pred 20, 20, 1
            # ---------------------------------------------------#
            obj_output = self.obj_preds[k](reg_feat)

            output = torch.cat([reg_output, obj_output, cls_output], 1)
            outputs.append(output)
        return outputs

