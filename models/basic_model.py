import torch
import torch.nn as nn
import torch.nn.functional as F
# 【修改点1】导入 EEGNet
from .backbone import resnet18, EEGNet
from .fusion_modules import SumFusion, ConcatFusion, FiLM, GatedFusion, ConcatFusion_Swin, ConcatFusion_DGL, GatedFusion_DGL, SumFusion_DGL, FiLM_DGL, ConcatFusion_DGL_unimodal
import numpy as np
from models.swin_transformer import SwinTransformer


class AVClassifier_DGL(nn.Module):
    def __init__(self, args):
        super(AVClassifier_DGL, self).__init__()

        fusion = args.fusion_method
        if args.dataset == 'VGGSound':
            n_classes = 309
        elif args.dataset == 'KineticSound':
            n_classes = 34
        elif args.dataset == 'kinect400':
            n_classes = 400
        elif args.dataset == 'CREMAD':
            n_classes = 6
        elif args.dataset == 'AVE':
            n_classes = 28
        elif args.dataset == 'DEAP':
            n_classes = 4
        else:
            raise NotImplementedError('Incorrect dataset name {}'.format(args.dataset))

        if fusion == 'sum':
            self.fusion_module = SumFusion_DGL(output_dim=n_classes)
        elif fusion == 'concat':
            if args.dataset == 'kinect400':
                self.fusion_module = ConcatFusion_DGL(output_dim=n_classes, input_dim=1024)
            else:
                self.fusion_module = ConcatFusion_DGL(output_dim=n_classes)
        elif fusion == 'film':
            self.fusion_module = FiLM_DGL(output_dim=n_classes, x_film=True)
        elif fusion == 'gated':
            self.fusion_module = GatedFusion_DGL(output_dim=n_classes, x_gate=True)
        else:
            raise NotImplementedError('Incorrect fusion method: {}!'.format(fusion))

        if args.modality == 'full':
            # 【修改点2】音频分支改用 EEGNet
            self.audio_net = EEGNet(args)
            self.visual_net = resnet18(modality='visual', args=args)

        if args.modality == 'visual':
            if args.dataset == 'kinect400':
                self.visual_net = resnet18(modality='visual', args=args)
                self.visual_classifier = nn.Linear(512, n_classes)
            else:
                self.visual_net = resnet18(modality='visual', args=args)
                self.visual_classifier = nn.Linear(512, n_classes)
        
        if args.modality == 'audio':
            # 【修改点3】单音频也改用 EEGNet
            self.audio_net = EEGNet(args)
            self.audio_classifier = nn.Linear(512, n_classes)
            
        self.modality = args.modality
        self.args = args
        
        # 【修改点4】添加 Dropout
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, audio, visual):
        
        if self.modality == 'full':
            #EEGNet直接输出 (B, 512)
            a = self.audio_net(audio)  
            
            # ResNet18 输出特征图，需要处理
            v = self.visual_net(visual)
            (_, C, H, W) = v.size()
            B = a.size()[0]
            v = v.view(B, -1, C, H, W)
            v = v.permute(0, 2, 1, 3, 4)
            v = F.adaptive_avg_pool3d(v, 1)
            v = torch.flatten(v, 1)

            # 【修改点5】应用 Dropout
            a = self.dropout(a)
            v = self.dropout(v)

            # DGL 融合模块返回三个输出
            a_out, v_out, out = self.fusion_module(a, v)

            # 注意：DGL 主程序期望的返回顺序是 out, a_out, v_out
            return out, a_out, v_out
        
        elif self.modality == 'visual':
            v = self.visual_net(visual)
            (_, C, H, W) = v.size()
            B = self.args.batch_size
            v = v.view(B, -1, C, H, W)
            v = v.permute(0, 2, 1, 3, 4)
            v = F.adaptive_avg_pool3d(v, 1)
            v = torch.flatten(v, 1)

            v = self.dropout(v)
            out = self.visual_classifier(v)
            
            # 占位
            a = torch.zeros_like(v)
            return out, out, out

        elif self.modality == 'audio':
            a = self.audio_net(audio)
            a = self.dropout(a)
            out = self.audio_classifier(a)
            
            # 占位
            v = torch.zeros_like(a)
            return out, out, out
            
        else:
            return 0, 0, 0


class AVClassifier(nn.Module):
    def __init__(self, args):
        super(AVClassifier, self).__init__()

        fusion = args.fusion_method
        if args.dataset == 'VGGSound':
            n_classes = 309
        elif args.dataset == 'KineticSound':
            n_classes = 34
        elif args.dataset == 'CREMAD':
            n_classes = 6
        elif args.dataset == 'AVE':
            n_classes = 28
        elif args.dataset == 'DEAP':
            n_classes = 4
        else:
            raise NotImplementedError('Incorrect dataset name {}'.format(args.dataset))

        if fusion == 'sum':
            self.fusion_module = SumFusion(output_dim=n_classes)
        elif fusion == 'concat':
            self.fusion_module = ConcatFusion(output_dim=n_classes)
        elif fusion == 'film':
            self.fusion_module = FiLM(output_dim=n_classes, x_film=True)
        elif fusion == 'gated':
            self.fusion_module = GatedFusion(output_dim=n_classes, x_gate=True)
        else:
            raise NotImplementedError('Incorrect fusion method: {}!'.format(fusion))

        if args.modality == 'full':
            # 【修改点6】音频分支改用 EEGNet
            self.audio_net = EEGNet(args)
            self.visual_net = resnet18(modality='visual', args=args)

        if args.modality == 'visual':
            self.visual_net = resnet18(modality='visual', args=args)
            self.visual_classifier = nn.Linear(512, n_classes)
            
        if args.modality == 'audio':
            # 【修改点7】单音频也改用 EEGNet
            self.audio_net = EEGNet(args)
            self.audio_classifier = nn.Linear(512, n_classes)

        self.args = args
        # 【修改点8】添加 Dropout
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, audio, visual):

        if self.args.modality == 'full':
            # EEGNet 直接输出向量
            a = self.audio_net(audio)  
            
            # ResNet18 输出特征图
            v = self.visual_net(visual)
            (_, C, H, W) = v.size()
            B = a.size()[0]
            v = v.view(B, -1, C, H, W)
            v = v.permute(0, 2, 1, 3, 4)
            v = F.adaptive_avg_pool3d(v, 1)
            v = torch.flatten(v, 1)

            # 【修改点9】应用 Dropout
            a = self.dropout(a)
            v = self.dropout(v)

            # 融合
            a, v, out = self.fusion_module(a, v)

            return a, v, out

        elif self.args.modality == 'visual':
            v = self.visual_net(visual)
            (_, C, H, W) = v.size()
            B = self.args.batch_size
            v = v.view(B, -1, C, H, W)
            v = v.permute(0, 2, 1, 3, 4)
            v = F.adaptive_avg_pool3d(v, 1)
            v = torch.flatten(v, 1)

            v = self.dropout(v)
            out = self.visual_classifier(v)

            a = torch.zeros_like(v)
            return a, v, out

        elif self.args.modality == 'audio':
            a = self.audio_net(audio)
            a = self.dropout(a)
            out = self.audio_classifier(a)
            
            v = torch.zeros_like(a)
            return a, v, out
            
        else:
            return 0, 0, 0