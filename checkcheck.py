import os
import torch
from models.topo_diffusion import TopoDiffusionUNet  # 替换为实际路径

# 基本加载方法
checkpoint = torch.load('D:\source\OBCR-D\checkpoints\\topo_model_step_999000.pt', map_location='cpu')


def process_checkpoint(checkpoint_path, device='cpu'):
    """完整的检查点处理流程"""
    # 检查文件是否存在
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"找不到检查点文件: {checkpoint_path}")

    print(f"处理检查点: {checkpoint_path}")

    try:
        # 加载检查点
        checkpoint = torch.load(checkpoint_path, map_location=device)

        # 检查检查点类型
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                print("检测到标准训练检查点格式")
                if 'epoch' in checkpoint:
                    print(f"此检查点来自训练的第 {checkpoint['epoch']} 轮")
                if 'global_step' in checkpoint:
                    print(f"此检查点来自训练的第 {checkpoint['global_step']} 步")

                # 创建模型并加载权重
                model = TopoDiffusionUNet(img_size=64, base_channels=64)
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                # 尝试将整个字典作为state_dict加载
                print("尝试将整个检查点作为state_dict加载")
                model = TopoDiffusionUNet(img_size=64, base_channels=64)
                model.load_state_dict(checkpoint)

        elif isinstance(checkpoint, torch.nn.Module):
            print("检查点包含完整模型实例")
            model = checkpoint

        else:
            # 假设是纯state_dict
            print("尝试将检查点作为纯state_dict加载")
            model = TopoDiffusionUNet(img_size=64, base_channels=64)
            model.load_state_dict(checkpoint)

        # 将模型移到设备并设为评估模式
        model = model.to(device)
        model.eval()

        print(f"成功加载模型!")
        return model

    except Exception as e:
        print(f"处理检查点时出错: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    import argparse

    # parser = argparse.ArgumentParser(description='处理PyTorch检查点')
    # parser.add_argument('--checkpoint', type=str, required=True, help='检查点文件路径')
    # parser.add_argument('--device', type=str, default='cuda', help='使用的设备')
    # args = parser.parse_args()

    print("已加载检查点")
    # 检查设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 处理检查点
    model = process_checkpoint("checkpoints/topo_model_step_999000.pt", device)

    # 打印模型摘要
    print("\n模型摘要:")
    print(model)

    # 打印参数数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型总参数: {total_params:,}")