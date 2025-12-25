import numpy as np
import os

# --- 配置 ---
data_dir = './EEGData'
num_subjects = 22
total_samples = 800  # 每个被试 40 trials * 20 segments = 800 samples

# 划分数量 (按切片总数 800 计算)
# 80% 训练, 10% 验证, 10% 测试
n_train = 640
n_val = 80
n_test = 80

print(f"开始生成划分文件 (Segment-level Random Shuffle: {n_train}/{n_val}/{n_test})...")

for participant_id in range(1, num_subjects + 1):
    subject_str = f's{participant_id:02}'
    label_path = os.path.join(data_dir, f'{subject_str}_labels.npy')
    
    if not os.path.exists(label_path):
        print(f"跳过 {subject_str}: 文件未找到")
        continue

    # 1. 生成所有切片的索引 (0-799)
    indices = np.arange(total_samples)
    
    # 2. 【关键一步】完全随机打乱所有切片
    # 这会导致同一个试次的切片分散在训练集和测试集中
    np.random.shuffle(indices)
    
    # 3. 创建划分掩码 (初始化为 -1)
    split_mask = np.full(total_samples, -1, dtype=int)
    
    # 4. 分配索引
    train_indices = indices[:n_train]
    val_indices = indices[n_train : n_train + n_val]
    test_indices = indices[n_train + n_val :]
    
    # 0: Train, 1: Val, 2: Test
    split_mask[train_indices] = 0
    split_mask[val_indices] = 1
    split_mask[test_indices] = 2
    
    # 验证完整性
    assert not np.any(split_mask == -1), f"Error: {subject_str} 有样本未被划分"
    assert len(test_indices) == n_test
    
    # 5. 保存
    save_path = os.path.join(data_dir, f'{subject_str}_split.npy')
    np.save(save_path, split_mask)
    
    print(f"[{subject_str}] 划分完成. Train: {np.sum(split_mask==0)}, Val: {np.sum(split_mask==1)}, Test: {np.sum(split_mask==2)}")

print("所有被试划分完成 (随机切片模式)。")