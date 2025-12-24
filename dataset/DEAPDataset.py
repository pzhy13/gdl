import os
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class DEAPDataset(Dataset):
    def __init__(self, args, mode='train'):
        self.mode = mode
        self.eeg_root = './EEGData'
        self.face_root = './faces'
        
        # 映射 mode 到 split.npy 的标记值
        mode_map = {'train': 0, 'val': 1, 'test': 2}
        if mode not in mode_map:
             raise ValueError("Mode must be 'train', 'val', or 'test'")
        target_split = mode_map[mode]
        
        self.samples = [] # 存储 (eeg_tensor, img_path, label)
        
        print(f"正在加载 DEAP {mode} 数据集 (混合 22 位被试)...")
        
        # 遍历所有被试
        for i in range(1, 23):
            subj_str = f's{i:02}'
            eeg_path = os.path.join(self.eeg_root, f'{subj_str}_eeg.npy')
            label_path = os.path.join(self.eeg_root, f'{subj_str}_labels.npy')
            split_path = os.path.join(self.eeg_root, f'{subj_str}_split.npy')
            face_dir = os.path.join(self.face_root, subj_str)
            
            if not os.path.exists(eeg_path) or not os.path.exists(split_path):
                continue
                
            # 加载数据
            eeg_data = np.load(eeg_path).astype(np.float32)   # (800, 32, 384)
            labels = np.load(label_path).astype(np.longlong)  # (800,)
            splits = np.load(split_path)                      # (800,)
            
            # 筛选当前 mode 的数据
            indices = np.where(splits == target_split)[0]
            
            if len(indices) == 0:
                continue

            # 准备图片列表 (确保排序)
            if os.path.exists(face_dir):
                all_imgs = sorted([
                    os.path.join(face_dir, f) 
                    for f in os.listdir(face_dir) 
                    if f.lower().endswith('.png')
                ])
            else:
                all_imgs = []
            
            # 逻辑：3秒切片 * 5fps = 15帧
            frames_per_seg = 15
            
            for idx in indices:
                # 提取 EEG
                eeg_sample = eeg_data[idx] 
                label_sample = labels[idx]
                
                # 提取 Image (取该切片对应 15 帧的中间那一帧)
                img_idx = idx * frames_per_seg + (frames_per_seg // 2)
                
                if img_idx < len(all_imgs):
                    img_path = all_imgs[img_idx]
                else:
                    # 容错：如果缺图，用最后一张图
                    img_path = all_imgs[-1] if len(all_imgs) > 0 else None
                
                if img_path:
                    self.samples.append((eeg_sample, img_path, label_sample))
        
        print(f"[{mode}] 加载完毕，总样本数: {len(self.samples)}")

        # 图像增强
        if mode == 'train':
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        eeg_arr, img_path, label = self.samples[idx]
        
        # 处理 EEG
        spec = torch.tensor(eeg_arr) 
        
        # 处理 Image
        try:
            image = Image.open(img_path).convert('RGB')
            image = self.transform(image) # 输出 shape: (3, 224, 224)
            
            # 【核心修改】增加时间维度 T=1
            # 变成 (3, 1, 224, 224) -> (Channels, Time, Height, Width)
            image = image.unsqueeze(1) 
            
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # 【核心修改】Fallback 也要保持维度一致 (3, 1, 224, 224)
            image = torch.zeros((3, 1, 224, 224))

        return spec, image, label