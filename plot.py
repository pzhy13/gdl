import numpy as np

# 文件路径
file_path = '/home/pzy/ICCV2025-GDL/EEGData/s01_labels.npy'

try:
    # 加载数据
    data = np.load(file_path)
    
    print(f"--- 文件信息: {file_path} ---")
    print(f"1. 数据形状 (Shape): {data.shape}")
    print(f"2. 数据类型 (Dtype): {data.dtype}")
    
    # 检查唯一值（看看有几类）
    unique_values = np.unique(data)
    print(f"3. 唯一值 (Unique Labels): {unique_values}")
    print(f"4. 类别数量 (Num Classes): {len(unique_values)}")
    
    # 打印前 20 个样本
    print(f"5. 前 20 个样本预览: {data[:20]}")
    
    # 简单统计
    if len(unique_values) < 20: # 如果类别少，打印每个类的样本数
        print("\n--- 类别分布统计 ---")
        for val in unique_values:
            count = np.sum(data == val)
            print(f"类别 {val}: {count} 个样本")

except FileNotFoundError:
    print(f"错误: 找不到文件 {file_path}")
except Exception as e:
    print(f"发生错误: {e}")