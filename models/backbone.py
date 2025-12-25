import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
import csv

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out




class ResNet(nn.Module):

    def __init__(self, args, block, layers, modality, num_classes=1000, pool='avgpool', zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        self.modality = modality
        self.pool = pool
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        if modality == 'audio':
            self.conv1 = nn.Conv2d(1, self.inplanes, kernel_size=7, stride=2, padding=3,
                                   bias=False)
        elif modality == 'visual':
            self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                                   bias=False)
        else:
            raise NotImplementedError('Incorrect modality, should be audio or visual but got {}'.format(modality))
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
     
        self.args = args

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.normal_(m.weight, mean=1, std=0.02)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):

        if self.modality == 'visual':

            (B, C, T, H, W) = x.size()
            x = x.permute(0, 2, 1, 3, 4).contiguous()
            x = x.view(B * T, C, H, W)

            x = self.conv1(x)

            # print("conv1 weight",self.conv1.weight)
            # print("visual,output",torch.sum(x))

            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)

            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            out = x


            out = out

            return out
        else:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)

            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            out = x

            # print(out.shape)

            out = out

            return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


def _resnet(arch, args, block, layers, modality, progress, **kwargs):
    model = ResNet(args, block, layers, modality, **kwargs)
    # if args.pretrain and args.modality == 'visual':
    #     state_dict = torch.load("resnet18-5c106cde.pth")
    #     model.load_state_dict(state_dict)
    return model


def resnet18(modality, args, progress=True, **kwargs):
    return _resnet('resnet18', args, BasicBlock, [2, 2, 2, 2], modality, progress,
                   **kwargs)


def resnet50(modality, args, progress=True, **kwargs):
    return _resnet('resnet50', args, BasicBlock, [3, 4, 6, 3], modality, progress,
                   **kwargs)

# 在 models/backbone.py 文件末尾添加这个类

class SimpleEEGNet(nn.Module):
    def __init__(self, args, output_dim=512):
        super(SimpleEEGNet, self).__init__()
        # 输入: (Batch, 1, 32, 384)
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(1, 64), stride=(1, 2), padding=(0, 32), bias=False),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.Dropout(0.5) # 强力 Dropout
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=(32, 1), groups=16, bias=False), # Depthwise conv 模拟空间滤波
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(0.5)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(1, 16), padding=(0, 8), bias=False),
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(0.5)
        )

        # 展平后的维度需要计算，这里简化处理，通过全连接层映射到 512
        self.flatten = nn.Flatten()
        
        # 假设经过上面层后，特征图大小约为 (Batch, 64, 1, 12) -> 768
        # 如果报错维度不匹配，请根据报错调整这里的输入维度
        self.fc = nn.Linear(768, output_dim) 

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

# models/backbone.py 末尾添加

class EEGNet(nn.Module):
    """
    标准的 EEGNet 实现 (Lawhern et al., 2018)
    输入: (Batch, 1, Channels, Time) -> 这里的 Channels=32, Time=384
    """
    def __init__(self, args, output_dim=512):
        super(EEGNet, self).__init__()
        
        # 根据你的数据配置
        nb_classes = output_dim # 这里我们不直接分类，而是映射到特征维度 (512)
        Chans = 32
        Samples = 384
        dropoutRate = 0.5
        kernLength = 64
        F1 = 8
        D = 2
        F2 = F1 * D

        self.block1 = nn.Sequential(
            # Padding=(0, 32) 为了保持时间维度，假设 kernel=64
            nn.Conv2d(1, F1, (1, kernLength), padding=(0, kernLength // 2), bias=False),
            nn.BatchNorm2d(F1, momentum=0.01, affine=True, eps=1e-3),
            
            # Depthwise Conv2D
            nn.Conv2d(F1, F1 * D, (Chans, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F1 * D, momentum=0.01, affine=True, eps=1e-3),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropoutRate)
        )

        self.block2 = nn.Sequential(
            # Separable Conv2D
            nn.Conv2d(F1 * D, F1 * D, (1, 16), padding=(0, 8), groups=F1 * D, bias=False),
            nn.Conv2d(F1 * D, F2, (1, 1), bias=False),
            nn.BatchNorm2d(F2, momentum=0.01, affine=True, eps=1e-3),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropoutRate)
        )

        self.flatten = nn.Flatten()
        
        # 计算全连接层输入维度
        # EEGNet 降采样非常厉害: 384 -> /4 -> 96 -> /8 -> 12
        # 输出特征图大小: (Batch, F2, 1, 12) -> F2=16
        # 所以 flatten 后大小 = 16 * 12 = 192
        # 建议让网络自动计算，或者手动算一下
        self.fc_input_dim = F2 * (Samples // 32) # 16 * 12 = 192
        
        self.fc = nn.Linear(self.fc_input_dim, output_dim)

    def forward(self, x):
        # EEGNet 期望输入 (Batch, 1, Channels, Time)
        # 你的数据是 (Batch, 1, 32, 384)，正好对应
        
        x = self.block1(x)
        x = self.block2(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x