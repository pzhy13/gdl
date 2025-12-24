import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class DEAPDataset(Dataset):
    def __init__(self, eeg_dir, face_dir, subject_id, mode='4class', transform=None):
        """
        Args:
            eeg_dir: EEG .npy 文件所在目录 (./EEGData)
            face_dir: 人脸图像所在目录 (./faces)
            subject_id: 被试 ID (如 1, 2...)
            mode: '4class' (四个象限), 'valence' (愉悦度2分类), 'arousal' (唤醒度2分类)
            transform: 图像预处理
        """
        self.subject_str = f's{subject_id:02}'
        self.eeg_path = os.path.join(eeg_dir, f'{self.subject_str}_eeg.npy')
        self.label_path = os.path.join(eeg_dir, f'{self.subject_str}_labels.npy')
        self.face_path = os.path.join(face_dir, f'{self.subject_str}')
        
        # 加载 EEG 和 标签
        # EEG shape: (800, 32, 384)
        self.eeg_data = np.load(self.eeg_path).astype(np.float32)
        self.labels = np.load(self.label_path).astype(np.longlong)
        
        # 处理标签 (2分类或4分类)
        if mode == 'valence':
            # 0(HVHA), 1(HVLA) -> 1 (High Valence)
            # 2(LVLA), 3(LVHA) -> 0 (Low Valence)
            self.labels = np.where(self.labels <= 1, 1, 0)
        elif mode == 'arousal':
            # 0(HVHA), 3(LVHA) -> 1 (High Arousal)
            # 1(HVLA), 2(LVLA) -> 0 (Low Arousal)
            self.labels = np.where((self.labels == 0) | (self.labels == 3), 1, 0)
        
        # 整理图像路径
        # 假设 faces/s01/ 下面是平铺的帧或者按Trial分文件夹
        # 根据你的脚本，文件名包含 video_filename_base，我们需要确保排序正确
        # 你的 EEG 是按 Trial 1-40 顺序排列的，图像也必须如此
        
        raw_files = sorted(os.listdir(self.face_path))
        # 过滤非图片文件
        self.image_files = [os.path.join(self.face_path, f) for f in raw_files if f.endswith('.png')]
        
        # 检查对齐
        # 理论上应该有 800个segment * 15帧 = 12000 张图 (如果所有视频都完整)
        # 实际情况可能有个别帧缺失，这里我们采用索引映射
        
        self.transform = transform
        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

        # 参数配置
        self.frames_per_segment = 15 
        
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # 1. 获取 EEG
        eeg_tensor = torch.tensor(self.eeg_data[idx]) # (32, 384)
        
        # 2. 获取 对应图像
        # 策略：从该 segment 对应的 15 帧中选 1 帧（通常选中间帧）或者多帧
        # 这里演示选中间帧 (第7帧)，如果你模型需要时序图像，可以改写这里返回 stack
        
        start_img_idx = idx * self.frames_per_segment
        mid_img_idx = start_img_idx + (self.frames_per_segment // 2)
        
        # 防止索引越界（如果有些视频帧数不够）
        if mid_img_idx >= len(self.image_files):
            mid_img_idx = len(self.image_files) - 1
            
        img_path = self.image_files[mid_img_idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image_tensor = self.transform(image)
            
        label = torch.tensor(self.labels[idx])
        
        return image_tensor, eeg_tensor, label