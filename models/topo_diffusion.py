import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gudhi as gd
from persim import wasserstein
from .blocks import SinusoidalPositionEmbeddings, AttentionBlock, DownBlock, UpBlock, TopoResBlock
from tqdm import tqdm


class TopologicalFeatureExtractor(nn.Module):
    """拓扑特征提取模块"""

    def __init__(self, filtration_levels=10):
        super().__init__()
        self.filtration_levels = filtration_levels

    def compute_persistence_diagram(self, image_tensor):
        """计算图像的持久性图(persistence diagram)"""
        # 转换为numpy进行处理
        image_np = image_tensor.detach().cpu().numpy().squeeze()

        # 二值化处理
        binary_image = (image_np > 0).astype(np.float32)

        # 创建立方复形
        cubical_complex = gd.CubicalComplex(
            dimensions=[binary_image.shape[0], binary_image.shape[1]],
            top_dimensional_cells=binary_image.flatten()
        )

        # 计算持久性
        cubical_complex.compute_persistence()

        # 获取持久性对
        persistence_pairs = cubical_complex.persistence_pairs()

        # 获取持久性图
        persistence_diagram = cubical_complex.persistence_intervals_in_dimension(1)  # 获取1维同调类(循环/洞)

        return persistence_diagram

    def compute_betti_curves(self, image_tensor):
        """计算贝蒂曲线(Betti curves)"""
        # 转换为numpy进行处理
        image_np = image_tensor.detach().cpu().numpy().squeeze()

        # 阈值序列
        thresholds = np.linspace(0, 1, self.filtration_levels)

        # 初始化贝蒂数数组
        betti_0 = np.zeros(self.filtration_levels)  # 0维贝蒂数(连通分量)
        betti_1 = np.zeros(self.filtration_levels)  # 1维贝蒂数(循环/洞)

        # 对每个阈值计算贝蒂数
        for i, threshold in enumerate(thresholds):
            # 二值化
            binary_image = (image_np > threshold).astype(np.float32)

            # 创建立方复形
            cubical_complex = gd.CubicalComplex(
                dimensions=[binary_image.shape[0], binary_image.shape[1]],
                top_dimensional_cells=binary_image.flatten()
            )

            # 计算持久性
            cubical_complex.compute_persistence()

            # 计算贝蒂数
            betti_0[i] = len(cubical_complex.persistence_intervals_in_dimension(0))
            betti_1[i] = len(cubical_complex.persistence_intervals_in_dimension(1))

        # 将numpy数组转换为PyTorch张量
        betti_curves = torch.tensor(np.stack([betti_0, betti_1]), dtype=torch.float32)

        return betti_curves

    def forward(self, x):
        """提取拓扑特征"""
        batch_size = x.shape[0]
        device = x.device

        # 初始化特征存储
        betti_features = []

        # 批处理
        for i in range(batch_size):
            # 计算贝蒂曲线
            betti_curve = self.compute_betti_curves(x[i])
            betti_features.append(betti_curve)

        # 堆叠batch维度
        betti_features = torch.stack(betti_features).to(device)

        return betti_features



class TopologicalLoss(nn.Module):
    """拓扑损失函数"""

    def __init__(self, topo_weight=0.1):
        super().__init__()
        self.topo_weight = topo_weight
        self.topo_feature_extractor = TopologicalFeatureExtractor()

    def wasserstein_distance(self, dgm1, dgm2):
        """计算两个持久性图之间的Wasserstein距离"""
        # 如果持久性图为空，返回0
        if len(dgm1) == 0 and len(dgm2) == 0:
            return 0.0

        return wasserstein(dgm1, dgm2)

    def betti_curve_loss(self, betti_curve1, betti_curve2):
        """计算贝蒂曲线之间的损失"""
        return F.mse_loss(betti_curve1, betti_curve2)

    def forward(self, generated_images, target_images=None, condition_images=None):
        """计算拓扑损失

        生成的图像应该与目标图像或条件图像在拓扑上相似
        """
        # 抽取生成图像的拓扑特征
        gen_topo_features = self.topo_feature_extractor(generated_images)

        # 如果提供了目标图像，使用目标图像计算损失
        if target_images is not None:
            target_topo_features = self.topo_feature_extractor(target_images)
            loss = self.betti_curve_loss(gen_topo_features, target_topo_features)
        # 否则使用条件图像计算损失
        elif condition_images is not None:
            cond_topo_features = self.topo_feature_extractor(condition_images)
            loss = self.betti_curve_loss(gen_topo_features, cond_topo_features)
        else:
            raise ValueError("必须提供target_images或condition_images之一")

        return self.topo_weight * loss


class TopoDiffusionUNet(nn.Module):
    """带拓扑特征的扩散U-Net模型"""


    def __init__(self, img_size=64, base_channels=64):
        super().__init__()
        self.img_size = img_size

        # 拓扑特征提取器
        self.topo_extractor = TopologicalFeatureExtractor()

        # 输入卷积
        self.conv_in = nn.Conv2d(1, base_channels, 3, padding=1)

        # Oracle图像编码器
        # 检查 oracle_encoder 的定义
        self.oracle_encoder = nn.ModuleList([
            nn.Conv2d(1, 64, 3, padding=1),  # 输出: [B, 64, 64, 64]
            nn.Conv2d(64, 128, 4, stride=2, padding=1),  # 输出: [B, 128, 32, 32]
            nn.Conv2d(128, 256, 4, stride=2, padding=1),  # 输出: [B, 256, 16, 16]
            nn.Conv2d(256, 256, 4, stride=2, padding=1)  # 输出: [B, 256, 8, 8]
        ])

        # 时间步编码
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(base_channels),
            nn.Linear(base_channels, base_channels * 4),
            nn.GELU(),
            nn.Linear(base_channels * 4, base_channels * 4)
        )
        # 在TopoDiffusionUNet.__init__()中：
        self.down1 = DownBlock(base_channels, base_channels * 2, time_channels=base_channels * 4)
        self.down2 = DownBlock(base_channels * 2, base_channels * 4, time_channels=base_channels * 4)
        self.down3 = DownBlock(base_channels * 4, base_channels * 4,time_channels=base_channels * 4)

        # 中间块
        self.mid = nn.ModuleList([
            TopoResBlock(base_channels * 4, topo_dim=2),
            AttentionBlock(base_channels * 4),
            TopoResBlock(base_channels * 4, topo_dim=2)
        ])
        # 上采样路径
        self.up1 = UpBlock(base_channels * 4, base_channels * 4)
        self.up1_block1 = TopoResBlock(base_channels * 8)  # 特征连接后通道数翻倍
        self.up1_block2 = TopoResBlock(base_channels * 8)
        self.up1_attn = AttentionBlock(base_channels * 8)

        self.up2 = UpBlock(base_channels * 8, base_channels * 2)
        self.up2_block1 = TopoResBlock(base_channels * 4)
        self.up2_block2 = TopoResBlock(base_channels * 4)

        self.up3 = UpBlock(base_channels * 4, base_channels)
        self.up3_block1 = TopoResBlock(base_channels * 2)
        self.up3_block2 = TopoResBlock(base_channels * 2)

        # 输出卷积
        self.conv_out = nn.Conv2d(base_channels * 2, 1, 3, padding=1)

    def forward(self, x, oracle_img, t, use_topo_guidance=True):
        """
        x: 噪声图像 [B, 1, H, W]
        oracle_img: 甲骨文字图像 [B, 1, H, W]
        t: 时间步 [B]
        use_topo_guidance: 是否使用拓扑引导
        """
        # 时间嵌入
        t_emb = self.time_mlp(t)

        # 拓扑特征提取(如果启用)
        # 在TopoDiffusionUNet的forward方法中修改
        # 拓扑特征提取(如果启用)
        """
           x: 噪声图像 [B, 1, H, W]
           oracle_img: 甲骨文字图像 [B, 1, H, W]
           t: 时间步 [B]
           use_topo_guidance: 是否使用拓扑引导
           """
        # 时间嵌入
        t_emb = self.time_mlp(t)

        # 拓扑特征提取(如果启用)
        topo_features = None
        if use_topo_guidance:
            try:
                print(f"Oracle image shape: {oracle_img.shape}")
                topo_features = self.topo_extractor(oracle_img)
                print(f"Extracted topo features shape: {topo_features.shape}")
                topo_features = torch.mean(topo_features, dim=2)
                print(f"Simplified topo features shape: {topo_features.shape}")
            except Exception as e:
                print(f"拓扑特征提取错误: {e}")
                use_topo_guidance = False

        # 甲骨文字编码 - 使用更健壮的方法
        try:
            # 存储特征列表
            oracle_features = []
            h_oracle = oracle_img

            # 添加原始输入
            oracle_features.append(h_oracle)  # 索引0是原始输入

            # 顺序应用编码器层
            print(f"开始提取oracle特征...")
            for i, layer in enumerate(self.oracle_encoder):
                print(f"  处理oracle_encoder层 {i}, 输入形状: {h_oracle.shape}")
                h_oracle = F.gelu(layer(h_oracle))
                print(f"  层 {i} 输出形状: {h_oracle.shape}")
                oracle_features.append(h_oracle)

            print(f"Oracle特征提取完成，共 {len(oracle_features)} 个特征")
        except Exception as e:
            print(f"Oracle特征提取错误: {e}")
            # 在提取失败时，创建一个空的特征列表
            oracle_features = [None] * (len(self.oracle_encoder) + 1)
            print(f"创建了空的oracle_features列表，长度: {len(oracle_features)}")

        # U-Net前向传播
        x1 = self.conv_in(x)
        print(f"x1 shape: {x1.shape}")

        # 下采样路径 (保存skip连接)
        print(f"执行down1...")
        x2 = self.down1(x1, t_emb)
        print(f"x2 shape: {x2.shape}")

        # 安全地添加甲骨文特征 - 使用索引1
        try:
            if oracle_features[1] is not None:
                if x2.shape == oracle_features[1].shape:
                    x2 = x2 + oracle_features[1]
                else:
                    print(f"形状不匹配，跳过特征融合")
            else:
                print(f"oracle_features[1]是None，跳过特征融合")
        except Exception as e:
            print(f"添加oracle特征到x2时出错: {e}")

        x3 = self.down2(x2, t_emb)

        # 安全地添加甲骨文特征 - 使用索引2
        try:
            if oracle_features[2] is not None:
                if x3.shape == oracle_features[2].shape:
                    x3 = x3 + oracle_features[2]
                else:
                    print(f"形状不匹配，跳过特征融合")
            else:
                print(f"oracle_features[2]是None，跳过特征融合")
        except Exception as e:
            print(f"添加oracle特征到x3时出错: {e}")

        print(f"执行down3...")
        x4 = self.down3(x3, t_emb)
        print(f"x4 shape: {x4.shape}")

        # 安全地添加甲骨文特征 - 使用索引3
        try:
            if oracle_features[3] is not None:
                if x4.shape == oracle_features[3].shape:
                    x4 = x4 + oracle_features[3]
                else:
                    print(f"形状不匹配，跳过特征融合")
            else:
                print(f"oracle_features[3]是None，跳过特征融合")
        except Exception as e:
            print(f"添加oracle特征到x4时出错: {e}")

        # 拓扑引导中间表示 - 使用更健壮的方法
        print(f"处理中间层...")
        for i, layer in enumerate(self.mid):
            try:
                if isinstance(layer, TopoResBlock) and use_topo_guidance:
                    x4 = layer(x4, topo_features)
                else:
                    x4 = layer(x4)
            except Exception as e:
                print(f"应用中间层 {i} 时出错: {e}")

        # 上采样路径 (使用skip连接)
        h = self.up1(x4)
        h = torch.cat([h, x3], dim=1)
        h = self.up1_block1(h, topo_features if use_topo_guidance else None)
        h = self.up1_block2(h, topo_features if use_topo_guidance else None)
        h = self.up1_attn(h)

        h = self.up2(h)
        h = torch.cat([h, x2], dim=1)
        h = self.up2_block1(h, topo_features if use_topo_guidance else None)
        h = self.up2_block2(h, topo_features if use_topo_guidance else None)

        h = self.up3(h)
        h = torch.cat([h, x1], dim=1)
        h = self.up3_block1(h, topo_features if use_topo_guidance else None)
        h = self.up3_block2(h, topo_features if use_topo_guidance else None)

        # 输出
        return self.conv_out(h)

    def get_losses(self, clean_images, oracle_images, noise_scheduler, topo_weight=0.1):
        """计算扩散模型的损失(噪声预测损失 + 拓扑损失)"""
        batch_size = clean_images.shape[0]
        device = next(self.parameters()).device

        # 随机时间步
        t = torch.randint(0, noise_scheduler.num_timesteps, (batch_size,), device=device).long()

        # 添加噪声
        noise = torch.randn_like(clean_images)
        x_t = noise_scheduler.add_noise(clean_images, noise, t)

        # 预测噪声
        noise_pred = self(x_t, oracle_images, t)

        # 噪声预测损失
        noise_loss = F.mse_loss(noise_pred, noise)

        # 拓扑损失 (仅在时间步较小时计算，以减少计算开销)
        topo_loss = 0.0
        if topo_weight > 0:
            # 随机选择一些样本计算拓扑损失
            topo_indices = torch.randperm(batch_size)[:min(4, batch_size)]

            # 重建当前噪声图像
            with torch.no_grad():
                # 使用预测的噪声移除一步噪声
                x_t_minus_1 = noise_scheduler.step_pred(noise_pred[topo_indices],
                                                        t[topo_indices],
                                                        x_t[topo_indices])

            # 计算拓扑损失
            topo_loss_fn = TopologicalLoss(topo_weight=1.0)
            topo_loss = topo_loss_fn(x_t_minus_1,
                                     target_images=clean_images[topo_indices],
                                     condition_images=oracle_images[topo_indices])

        # 总损失
        total_loss = noise_loss + topo_weight * topo_loss

        return total_loss

    def sample(self, oracle_img, noise_scheduler, num_inference_steps=100, use_topo_guidance=True):
        """从噪声生成图像，使用拓扑引导"""
        batch_size = oracle_img.shape[0]
        device = next(self.parameters()).device

        # 从纯高斯噪声开始
        img = torch.randn(batch_size, 1, self.img_size, self.img_size).to(device)

        # 时间步计划
        timesteps = torch.linspace(0, noise_scheduler.num_timesteps - 1, num_inference_steps)
        timesteps = timesteps.flip(0).long().to(device)  # 从T到0的降序

        # 逐步去噪
        for t in tqdm(timesteps, desc="生成采样"):
            # 防止梯度计算
            with torch.no_grad():
                # 预测噪声残差
                noise_pred = self(img, oracle_img, torch.full((batch_size,), t, device=device),
                                  use_topo_guidance=use_topo_guidance)

                # 计算前一时间步的图像
                img = noise_scheduler.step_pred(noise_pred, t, img)

        # 去噪后的图像
        return img

