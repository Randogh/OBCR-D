import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import argparse
from datetime import datetime

# 导入自定义模块
from models.topo_diffusion import TopoDiffusionUNet, TopologicalLoss
from utils.scheduler import NoiseScheduler
from utils.dataset import OracleToModernDataset


def visualize_topo_samples(model, dataset, noise_scheduler, device, num_samples=4,
                           save_path=None, compare_with_without_topo=False):
    """生成样本并可视化，可选择对比有无拓扑引导的结果"""



def get_topo_weight(epoch, step, max_weight=0.1, warmup_epochs=10):
    """计算拓扑损失的动态权重"""
    # 前warmup_epochs个epoch不使用拓扑损失
    if epoch < warmup_epochs:
        return 0.0
    # 之后线性增加权重到max_weight
    return min(max_weight, (epoch - warmup_epochs) * (max_weight / 10))


def load_checkpoint_flexibly(model, checkpoint_path, device):
    """灵活加载checkpoint，允许部分加载"""
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint不存在: {checkpoint_path}")
        return None, 0, {}

    try:
        print(f"正在加载checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)

        # 尝试加载模型权重，允许部分匹配
        model_dict = model.state_dict()
        pretrained_dict = checkpoint['model']

        # 过滤掉不匹配的键
        matched_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and model_dict[k].shape == v.shape}

        # 检查匹配率
        match_percentage = len(matched_dict) / len(model_dict) * 100
        print(f"模型权重匹配率: {match_percentage:.1f}% ({len(matched_dict)}/{len(model_dict)})")

        if match_percentage < 50:
            print("警告: 匹配率低于50%，可能会导致意外行为")

        # 加载匹配的权重
        model_dict.update(matched_dict)
        model.load_state_dict(model_dict)

        return checkpoint.get('optimizer', None), checkpoint.get('epoch', 0), checkpoint.get('metrics', {})

    except Exception as e:
        import traceback
        print(f"加载checkpoint时出错: {e}")
        print(traceback.format_exc())
        return None, 0, {}


def verify_checkpoint(checkpoint_path):
    """验证checkpoint文件是否完整可用"""
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        # 检查必要的键是否存在
        required_keys = ['model', 'epoch']
        for key in required_keys:
            if key not in checkpoint:
                print(f"Checkpoint缺少关键部分: {key}")
                return False

        # 检查模型权重是否完整
        if len(checkpoint['model']) == 0:
            print("模型权重为空")
            return False

        print(f"Checkpoint验证通过，包含 {len(checkpoint['model'])} 个参数组，截至第 {checkpoint['epoch']} 个epoch")
        return True
    except Exception as e:
        print(f"Checkpoint验证失败: {e}")
        return False

def main(args):
    """主训练函数"""
    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 创建保存目录
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.sample_dir, exist_ok=True)

    # 初始化模型
    model = TopoDiffusionUNet(img_size=args.img_size, base_channels=args.base_channels).to(device)
    print("\n模型结构:")
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            if hasattr(module, 'weight'):
                print(f"{name}: {type(module).__name__}, 权重形状: {module.weight.shape}")

    # 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # 学习率调度、
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.num_epochs, eta_min=args.min_lr
    )

    # 噪声调度器
    noise_scheduler = NoiseScheduler(num_timesteps=args.timesteps)
    for attr in ['betas', 'alphas', 'alphas_cumprod', 'alphas_cumprod_prev',
                 'sqrt_alphas_cumprod', 'sqrt_one_minus_alphas_cumprod',
                 'sqrt_recip_alphas', 'posterior_variance']:
        setattr(noise_scheduler, attr, getattr(noise_scheduler, attr).to(device))

    # 数据集和数据加载器
    train_dataset = OracleToModernDataset(
        data_dir=args.data_dir,
        songti_dir=args.songti_dir,
        labels_path=args.labels_path,
        split="train",
        img_size=args.img_size
    )

    test_dataset = OracleToModernDataset(
        data_dir=args.data_dir,
        songti_dir=args.songti_dir,
        labels_path=args.labels_path,
        split="test",
        img_size=args.img_size
    )

    # 检查数据集是否有样本
    if len(train_dataset) == 0:
        print("错误: 训练数据集为空!")
        return

    # 测试第一个样本是否可以加载
    try:
        first_sample = train_dataset[0]
        if first_sample is None:
            print("错误: 无法加载第一个样本!")
            return

        print(
            f"第一个样本加载成功: Oracle shape: {first_sample['oracle'].shape}, Modern shape: {first_sample['modern'].shape}")
    except Exception as e:
        print(f"加载第一个样本时出错: {e}")
        return

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )

    # 初始化全局步数
    global_step = 0

    start_epoch = 0

    # 加载检查点(如果提供)
    if args.resume_from:
        checkpoint_dir = args.checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)

        checkpoints = []
        for filename in os.listdir(checkpoint_dir):
            if filename.startswith('checkpoint_') and filename.endswith('.pt'):
                try:
                    epoch = int(filename.split('_')[1].split('.')[0])
                    checkpoints.append((epoch, os.path.join(checkpoint_dir, filename)))
                except:
                    pass
        checkpoints.sort(reverse=True)
        loaded = False
        for epoch, checkpoint_path in checkpoints:
            if verify_checkpoint(checkpoint_path):
                print(f"尝试加载checkpoint: epoch {epoch}")
                optimizer_state, loaded_epoch, metrics = load_checkpoint_flexibly(model, checkpoint_path, args.device)
                if optimizer_state:
                    optimizer.load_state_dict(optimizer_state)
                    start_epoch = loaded_epoch + 1
                    best_metrics = metrics
                    loaded = True
                    print(f"成功恢复到epoch {loaded_epoch}，将从epoch {start_epoch}继续训练")
                    break
        if not loaded:
            start_epoch = 0


    # 训练循环
    for epoch in range(start_epoch, args.num_epochs):
        model.train()
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{args.num_epochs}")

        for step, batch in enumerate(progress_bar):
            # 准备数据
            oracle_imgs = batch["oracle"].to(device)
            modern_imgs = batch["modern"].to(device)

            # 当前拓扑权重
            current_topo_weight = get_topo_weight(
                epoch, global_step, max_weight=args.max_topo_weight,
                warmup_epochs=args.topo_warmup_epochs
            )

            # 计算损失
            loss = model.get_losses(
                modern_imgs, oracle_imgs, noise_scheduler,
                topo_weight=current_topo_weight
            )

            # 反向传播
            optimizer.zero_grad()
            loss.backward()

            # 梯度裁剪以稳定训练
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            optimizer.step()

            # 更新进度条
            progress_bar.set_postfix(loss=loss.item(), topo_w=current_topo_weight)

            # 生成样本
            if global_step % args.save_image_steps == 0:
                print(f"生成样本，步骤: {global_step}")
                sample_path = os.path.join(
                    args.sample_dir,
                    f"sample_step_{global_step}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                )
                visualize_topo_samples(
                    model, test_dataset, noise_scheduler, device,
                    num_samples=4, save_path=sample_path
                )

            # 保存模型
            if global_step % args.save_model_steps == 0:
                checkpoint_path = os.path.join(
                    args.checkpoint_dir,
                    f"topo_model_step_{global_step}.pt"
                )
                torch.save({
                    'epoch': epoch,
                    'global_step': global_step,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss.item(),
                }, checkpoint_path)

            global_step += 1

        # 更新学习率
        scheduler.step()

        # 每个epoch结束保存模型
        checkpoint_path = os.path.join(
            args.checkpoint_dir,
            f"topo_model_epoch_{epoch + 1}.pt"
        )
        torch.save({
            'epoch': epoch,
            'global_step': global_step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss.item(),
        }, checkpoint_path)

        # 每个epoch结束生成样本
        if epoch % args.compare_every_n_epochs == 0:
            # 每N个epoch对比一次有无拓扑引导的结果
            compare = True
        else:
            compare = False

        sample_path = os.path.join(
            args.sample_dir,
            f"sample_epoch_{epoch + 1}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        )
        visualize_topo_samples(
            model, test_dataset, noise_scheduler, device,
            num_samples=8, save_path=sample_path,
            compare_with_without_topo=compare
        )

    print("训练完成!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="拓扑增强扩散模型训练")

    # 数据参数
    parser.add_argument("--data_dir", type=str, default="./Oracle", help="甲骨文图像目录")
    parser.add_argument("--songti_dir", type=str, default="./songti", help="宋体汉字图像目录")
    parser.add_argument("--labels_path", type=str, default="./labels.csv", help="标签文件路径")
    parser.add_argument("--img_size", type=int, default=64, help="图像大小")

    # 模型参数
    parser.add_argument("--base_channels", type=int, default=64, help="模型基础通道数")
    parser.add_argument("--timesteps", type=int, default=1000, help="扩散时间步数")

    # 训练参数
    parser.add_argument("--batch_size", type=int, default=8, help="批次大小")
    parser.add_argument("--num_epochs", type=int, default=100, help="训练轮数")
    parser.add_argument("--lr", type=float, default=1e-4, help="初始学习率")
    parser.add_argument("--min_lr", type=float, default=1e-5, help="最小学习率")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="权重衰减")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪值")
    parser.add_argument("--num_workers", type=int, default=4, help="数据加载器工作线程数")
    parser.add_argument("--device", type=str, default="cuda", help="训练设备")

    # 拓扑参数
    parser.add_argument("--max_topo_weight", type=float, default=0.1, help="最大拓扑损失权重")
    parser.add_argument("--topo_warmup_epochs", type=int, default=10, help="拓扑损失预热轮数")

    # 保存参数
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints", help="检查点保存目录")
    parser.add_argument("--sample_dir", type=str, default="./samples", help="样本保存目录")
    parser.add_argument("--save_model_steps", type=int, default=1000, help="每多少步保存一次模型")
    parser.add_argument("--save_image_steps", type=int, default=500, help="每多少步生成一次样本")
    parser.add_argument("--compare_every_n_epochs", type=int, default=5,
                        help="每多少轮对比有无拓扑引导的结果")

    # 恢复训练
    parser.add_argument("--resume_from", type=str, default=None, help="从指定检查点恢复训练")

    args = parser.parse_args()

    main(args)