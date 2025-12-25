import torch
import torch.nn as nn
import torch.optim as optim
import sys
import os

# 将当前目录加入路径，确保能 import models
sys.path.append(os.getcwd())

# 尝试导入 EEGNet
# 注意：根据你的文件结构，这里可能需要调整。
# 如果你的 EEGNet 定义在 models/EEGNet.py 中，请改为 from models.EEGNet import EEGNet
try:
    from models.basic_model import EEGNet
    print("成功从 models.basic_model 导入 EEGNet")
except ImportError:
    try:
        from models.EEGNet import EEGNet
        print("成功从 models.EEGNet 导入 EEGNet")
    except ImportError:
        print("Error: 找不到 EEGNet。请确认 debug.py 在项目根目录，且 models 文件夹里有相应文件。")
        exit(1)

# --- 1. 模拟参数类 (Mock Args) ---
class MockArgs:
    def __init__(self):
        self.dataset = 'DEAP'
        # DEAP 数据集通常是 32 通道
        self.num_channels = 32  
        # DEAP 采样率通常是 128Hz
        self.sampling_rate = 128 
        # 【关键】卷积核长度。通常设为采样率的 1/2 或 1/4。
        # 如果是 128Hz，建议设为 32 或 16。如果是 64 会导致过大的时间窗。
        self.kernLength = 32  
        self.dropout = 0.25
        self.n_classes = 4

def test_eegnet():
    # 设备配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n--- 开始调试 (Device: {device}) ---")

    # 初始化参数和模型
    args = MockArgs()
    try:
        model = EEGNet(args).to(device)
        print("模型初始化成功。")
    except Exception as e:
        print(f"模型初始化失败: {e}")
        return

    # --- 2. 构造模拟数据 ---
    batch_size = 16
    channels = 32
    time_points = 128  # 假设输入是 1 秒的数据 (128个点)
    
    # 模拟输入数据 (Batch, Channels, Time)
    # 注意：这里先生成 3D 数据，稍后测试是否需要 4D
    dummy_input = torch.randn(batch_size, channels, time_points).to(device)
    
    # 模拟标签 (0-3)
    dummy_label = torch.randint(0, 4, (batch_size,)).to(device)

    # --- 3. 维度匹配测试 (Shape Check) ---
    print(f"\n[Step 1] 检查输入维度...")
    print(f"原始输入形状: {dummy_input.shape} (Batch, Channels, Time)")
    
    input_tensor = dummy_input
    
    # 尝试直接输入 (3D)
    try:
        output = model(input_tensor)
        print("Result: 模型接受 (B, C, T) 3D 输入。")
    except RuntimeError as e:
        if "Expected 4-dimensional input" in str(e) or "weight" in str(e):
            print("Catch: 模型报错，似乎需要 4D 输入。尝试 unsqueeze(1)...")
            # 变为 (Batch, 1, Channels, Time)
            input_tensor = dummy_input.unsqueeze(1)
            print(f"新输入形状: {input_tensor.shape}")
            try:
                output = model(input_tensor)
                print("Result: 模型接受 (B, 1, C, T) 4D 输入。")
            except Exception as e2:
                print(f"Fatal Error: 4D 输入也失败了。报错信息: {e2}")
                return
        else:
            print(f"Fatal Error: 前向传播失败。报错信息: {e}")
            return

    print(f"模型输出形状: {output.shape} (预期应为 [{batch_size}, 4])")

    # --- 4. 过拟合测试 (Sanity Check) ---
    print(f"\n[Step 2] 开始过拟合测试 (Sanity Check)...")
    print("目标：Loss 应该迅速下降，Acc 接近 100%。如果 Loss 不降，说明模型结构有问题。")
    
    criterion = nn.CrossEntropyLoss()
    # 使用较大的学习率，为了快速验证
    optimizer = optim.Adam(model.parameters(), lr=0.001) 

    model.train()
    
    for i in range(1, 101): # 跑 100 轮
        optimizer.zero_grad()
        
        # 前向传播
        out = model(input_tensor)
        loss = criterion(out, dummy_label)
        
        # 反向传播
        loss.backward()
        
        # --- 5. 梯度检查 (只在第一轮查) ---
        if i == 1:
            has_grad = False
            zero_grad_count = 0
            total_param_count = 0
            for name, param in model.named_parameters():
                total_param_count += 1
                if param.grad is not None:
                    has_grad = True
                    if param.grad.abs().sum() == 0:
                        zero_grad_count += 1
            
            if not has_grad:
                print("【严重警告】: 没有任何参数产生梯度！请检查 requires_grad 设置。")
            elif zero_grad_count == total_param_count:
                print("【严重警告】: 所有梯度均为 0！可能是输入数据全为0或死神经元。")
            else:
                print("梯度检查通过: 参数正在更新。")

        optimizer.step()
        
        # 计算准确率
        pred = torch.argmax(out, dim=1)
        acc = (pred == dummy_label).float().mean().item()
        
        if i % 10 == 0:
            print(f"Iter {i:03d} | Loss: {loss.item():.4f} | Acc: {acc*100:.1f}%")
            
        if acc == 1.0 and loss.item() < 0.01:
            print(f"\n成功！在 Iter {i} 达到完美拟合。模型结构有效！")
            break

    if loss.item() > 1.0:
        print("\n【结论】: 失败。模型无法拟合固定的随机噪声。")
        print("可能原因：")
        print("1. 卷积核 (KernLength) 太大，超过了输入数据的时间长度。")
        print("2. 缺少 BatchNorm 或激活函数。")
        print("3. 输入数据未归一化 (Input Normalization)。")
    else:
        print("\n【结论】: 成功。模型具备学习能力。请检查数据加载部分。")

if __name__ == "__main__":
    test_eegnet()