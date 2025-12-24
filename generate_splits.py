import numpy as np
import os

# --- 配置 ---
data_dir = './EEGData'
num_subjects = 22
trials_per_subject = 40    # DEAP 原始有 40 个 Trial
segments_per_trial = 20    # 每个 Trial 切成了 20 个 Segment
total_samples = trials_per_subject * segments_per_trial # 800

# 按 Trial 数量划分
# 32个训练, 4个验证, 4个测试
n_train = 32
n_val = 4
n_test = 4

print(f"开始生成划分文件 (Trial-level Split: {n_train}/{n_val}/{n_test})...")

for participant_id in range(1, num_subjects + 1):
    subject_str = f's{participant_id:02}'
    label_path = os.path.join(data_dir, f'{subject_str}_labels.npy')
    
    if not os.path.exists(label_path):
        print(f"跳过 {subject_str}: 文件未找到")
        continue

    # 1. 生成 Trial 级索引 (0-39) 并随机打乱
    trial_indices = np.arange(trials_per_subject)
    np.random.shuffle(trial_indices)
    
    # 2. 分配 Trial
    train_trials = trial_indices[:n_train]
    val_trials = trial_indices[n_train : n_train + n_val]
    test_trials = trial_indices[n_train + n_val :]
    
    # 3. 将 Trial 映射到 Segment (0-799)
    # 0: Train, 1: Val, 2: Test
    split_mask = np.full(total_samples, -1, dtype=int)
    
    def mark_segments(trial_list, split_label):
        for t_idx in trial_list:
            # Trial t_idx 对应的切片范围是 [t*20, (t+1)*20)
            start_idx = t_idx * segments_per_trial
            end_idx = (t_idx + 1) * segments_per_trial
            split_mask[start_idx : end_idx] = split_label

    mark_segments(train_trials, 0)
    mark_segments(val_trials, 1)
    mark_segments(test_trials, 2)
    
    # 验证完整性
    assert not np.any(split_mask == -1), f"Error: {subject_str} 有样本未被划分"
    
    # 4. 保存
    save_path = os.path.join(data_dir, f'{subject_str}_split.npy')
    np.save(save_path, split_mask)
    
    print(f"[{subject_str}] 划分完成. Train Segs: {np.sum(split_mask==0)}, Val: {np.sum(split_mask==1)}, Test: {np.sum(split_mask==2)}")

print("所有被试划分完成。")