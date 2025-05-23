import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
from torchvision.utils import save_image
import argparse
from tqdm import tqdm

from models.topo_diffusion import TopoDiffusionUNet  # 替换为您实际的模型类
from utils.scheduler import NoiseScheduler


class InferenceConfig:
    # 设备配置
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 模型参数（必须与训练时一致）
    img_size = 64
    base_channels = 64

    # 推理参数
    num_inference_steps = 1000  # 去噪步数
    use_topo_guidance = True  # 是否启用拓扑引导

    # 路径配置
    checkpoint_path = "./checkpoints/topo_model_step_999000.pt"
    oracle_dir = "./data/Oracle/images"
    output_dir = "./samples"


config = InferenceConfig()
# 检查配置文件中的路径设置

# 创建噪声调度器实例（参数需与训练时一致）
noise_scheduler = NoiseScheduler(
    num_timesteps=1000,       # 时间步数（必须与训练一致）
    beta_start=1e-4,          # 起始beta值
    beta_end=0.02             # 结束beta值
).to(config.device)                   # 将参数移至设备（如GPU）


def load_model(checkpoint_path, config):
    """加载预训练模型"""
    # 初始化模型
    model = TopoDiffusionUNet(
        img_size=config.img_size,
        base_channels=config.base_channels
    )

    # 加载检查点
    checkpoint = torch.load(checkpoint_path, map_location=config.device)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    # 设备迁移与评估模式
    model = model.to(config.device).eval()
    print(f"成功加载模型到 {config.device}")
    return model

def create_preprocess_pipeline():
    """创建图像预处理流程"""
    return transforms.Compose([
        transforms.Resize(config.img_size),
        transforms.CenterCrop(config.img_size),
        transforms.Grayscale(num_output_channels=1),  # 转为单通道
        transforms.ToTensor(),
        transforms.Lambda(lambda x: (x > 0.5).float()),  # 二值化
        transforms.Normalize(0.5, 0.5)  # 归一化到 [-1, 1]
    ])

preprocess = create_preprocess_pipeline()


@torch.no_grad()
def generate_single(model, oracle_img_path, config):
    """生成单个样本"""
    # 加载并预处理甲骨文图像
    oracle_img = Image.open(oracle_img_path).convert("L")
    oracle_tensor = preprocess(oracle_img).unsqueeze(0).to(config.device)

    # 生成过程
    generated = model.sample(
        oracle_img=oracle_tensor,
        noise_scheduler=noise_scheduler,
        num_inference_steps=config.num_inference_steps,
        use_topo_guidance=config.use_topo_guidance
    )

    # 后处理：反归一化并裁剪
    generated = (generated * 0.5) + 0.5  # [-1,1] => [0,1]
    generated = torch.clamp(generated, 0.0, 1.0)

    # 构造保存路径
    base_name = os.path.basename(oracle_img_path).split('.')[0]  # 提取文件名
    output_path = os.path.join(config.output_dir, f"gen_{base_name}.png")

    # 保存生成的图片
    save_image(generated, output_path)
    print(f"生成结果已保存到: {output_path}")
    return generated


def batch_inference(model, config):
    """批量处理甲骨文图像"""
    os.makedirs(config.output_dir, exist_ok=True)

    # 获取所有甲骨文图像
    oracle_files = [f for f in os.listdir(config.oracle_dir)
                    if f.endswith(('.png', '.jpg', '.jpeg'))]

    for filename in tqdm(oracle_files, desc="生成进度"):
        # 生成图像
        img_path = os.path.join(config.oracle_dir, filename)
        generated = generate_single(model, img_path, config)

        # 保存结果
        output_path = os.path.join(config.output_dir, f"gen_{filename}")
        save_image(generated, output_path)

        # 保存对比图
        oracle_img = preprocess(Image.open(img_path)).unsqueeze(0)
        comparison = torch.cat([oracle_img, generated.cpu()], dim=3)
        save_image(comparison, os.path.join(config.output_dir, f"comp_{filename}"))


if __name__ == "__main__":
    # 初始化组件
    model = load_model(config.checkpoint_path, config)
    noise_scheduler = noise_scheduler
    print(f"配置的输出目录: {config.output_dir}")

    # 检查目录是否真实存在
    print(f"目录是否存在: {os.path.exists(config.output_dir)}")

    #执行单次生成
    generate_single(model,"data/Oracle/images/3ac3_11.png", config)
    # 执行批量生成
    #batch_inference(model, config)
    print(f"生成完成！结果保存在 {config.output_dir}")