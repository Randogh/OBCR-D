import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SinusoidalPositionEmbeddings(nn.Module):
    """正弦位置编码，用于时间步嵌入"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class AttentionBlock(nn.Module):
    """自注意力块"""

    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        b, c, h, w = x.shape

        # 标准化
        x_norm = self.norm(x)

        # 计算QKV
        qkv = self.qkv(x_norm)
        q, k, v = torch.chunk(qkv, 3, dim=1)

        # 重塑以进行注意力计算
        q = q.reshape(b, c, h * w).permute(0, 2, 1)  # [B, HW, C]
        k = k.reshape(b, c, h * w).permute(0, 2, 1)  # [B, HW, C]
        v = v.reshape(b, c, h * w).permute(0, 2, 1)  # [B, HW, C]

        # 计算注意力 - 使用矩阵乘法
        scale = 1 / math.sqrt(c)
        attn = torch.bmm(q, k.transpose(1, 2)) * scale  # [B, HW, HW]
        attn = F.softmax(attn, dim=-1)

        # 应用注意力
        out = torch.bmm(attn, v)  # [B, HW, C]
        out = out.permute(0, 2, 1).reshape(b, c, h, w)

        # 投影
        out = self.proj(out)

        return x + out


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.norm1 = nn.GroupNorm(8, in_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)

        # 时间嵌入投影
        print(f"DownBlock: 初始化time_mlp, 输入维度: {time_channels}, 输出维度: {out_channels}")
        self.time_mlp = nn.Sequential(
            nn.Linear(time_channels, out_channels),
            nn.GELU(),
            nn.Linear(out_channels, out_channels)
        )

    def forward(self, x, t_emb):
        print(f"DownBlock: x形状: {x.shape}, t_emb形状: {t_emb.shape}")
        print(f"time_mlp第一层权重形状: {self.time_mlp[0].weight.shape}")

        h = self.conv1(F.gelu(self.norm1(x)))

        # 添加时间嵌入
        try:
            time_emb = self.time_mlp(t_emb)
            time_emb = time_emb.view(time_emb.shape[0], time_emb.shape[1], 1, 1)
            h = h + time_emb
        except Exception as e:
            print(f"时间嵌入处理错误: {e}")
            # 如果出错，继续而不添加时间嵌入

        h = self.conv2(F.gelu(self.norm2(h)))
        return self.pool(h)

class UpBlock(nn.Module):
    """上采样块"""

    def __init__(self, in_channels, out_channels):
        super().__init__()

        # 上采样
        self.upsample = nn.ConvTranspose2d(in_channels, out_channels, 4, stride=2, padding=1)

    def forward(self, x):
        return self.upsample(x)


class AttentionBlock(nn.Module):
    """自注意力块"""

    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        b, c, h, w = x.shape

        # 标准化
        x_norm = self.norm(x)

        # 计算QKV
        qkv = self.qkv(x_norm)
        q, k, v = torch.chunk(qkv, 3, dim=1)

        # 重塑以进行注意力计算
        q = q.reshape(b, c, h * w).permute(0, 2, 1)  # [B, HW, C]
        k = k.reshape(b, c, h * w).permute(0, 2, 1)  # [B, HW, C]
        v = v.reshape(b, c, h * w).permute(0, 2, 1)  # [B, HW, C]

        # 计算注意力 - 使用矩阵乘法
        scale = 1 / math.sqrt(c)
        attn = torch.bmm(q, k.transpose(1, 2)) * scale  # [B, HW, HW]
        attn = F.softmax(attn, dim=-1)

        # 应用注意力
        out = torch.bmm(attn, v)  # [B, HW, C]
        out = out.permute(0, 2, 1).reshape(b, c, h, w)

        # 投影
        out = self.proj(out)

        return x + out


class TopoResBlock(nn.Module):
    """结合拓扑特征的残差块"""

    def __init__(self, channels, topo_dim=2):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, channels)
        self.norm2 = nn.GroupNorm(8, channels)

        # 拓扑特征投影
        self.topo_proj = nn.Sequential(
            nn.Linear(topo_dim, 64),
            nn.GELU(),
            nn.Linear(64, channels)
        )

    def forward(self, x, topo_features=None):
        """
        x: 输入特征 [B, C, H, W]
        topo_features: 拓扑特征 [B, topo_dim]
        """
        h = self.conv1(F.gelu(self.norm1(x)))

        # 如果提供了拓扑特征，将其融入残差块
        if topo_features is not None:
            try:
                print(f"在TopoResBlock中融合拓扑特征，特征形状: {topo_features.shape}")

                # 投影拓扑特征到通道维度
                b, c, height, width = h.shape
                topo_emb = self.topo_proj(topo_features)  # [B, C]

                # 调整拓扑特征的维度以进行广播加法
                topo_emb = topo_emb.view(b, c, 1, 1)

                # 添加到特征图
                h = h + topo_emb
                print(f"拓扑特征融合完成")
            except Exception as e:
                print(f"拓扑特征融合错误: {e}")
                # 在出错时继续而不使用拓扑特征
        else:
            print(f"TopoResBlock: 没有提供拓扑特征")

        h = self.conv2(F.gelu(self.norm2(h)))
        return x + h