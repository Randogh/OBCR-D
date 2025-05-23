import torch
import numpy as np


class NoiseScheduler:
    """噪声调度器，管理扩散过程的添加和移除噪声"""

    def __init__(self, num_timesteps=1000, beta_start=1e-4, beta_end=0.02):
        self.num_timesteps = num_timesteps

        # 线性噪声调度
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([torch.ones(1), self.alphas_cumprod[:-1]])

        # 扩散过程中使用的各种常数
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)

        # 后验方差
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)

    def add_noise(self, x_0, noise, t):
        """在给定时间步t给清晰图像x_0添加噪声"""
        batch_size = x_0.shape[0]
        device = x_0.device

        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].to(device)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].to(device)

        # 重塑以便广播
        sqrt_alphas_cumprod_t = sqrt_alphas_cumprod_t.view(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod_t.view(-1, 1, 1, 1)

        # 添加噪声
        x_t = sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise

        return x_t

    def to(self, device):
        # 将调度器的所有张量迁移到指定设备
        self.device = device
        for key, value in self.__dict__.items():
            if isinstance(value, torch.Tensor):
                setattr(self, key, value.to(device))
        return self

    def step_pred(self, noise_pred, t, x_t):
        # 预测性去噪：使用预测的噪声从x_t计算x_{t-1}
        batch_size = x_t.shape[0]
        device = x_t.device

        # 获取当前时间步的参数
        alpha_bar_t = self.alphas_cumprod[t].view(-1, 1, 1, 1)  # ᾱ_t
        alpha_bar_t_prev = self.alphas_cumprod_prev[t].view(-1, 1, 1, 1)  # ᾱ_{t-1}
        beta_t = self.betas[t].view(-1, 1, 1, 1)  # β_t

        # 计算预测的干净图像 x_0
        sqrt_recip_alpha_bar_t = torch.sqrt(1.0 / alpha_bar_t)  # 1/√ᾱ_t
        sqrt_one_minus_alpha_bar_t = torch.sqrt(1.0 - alpha_bar_t)  # √(1-ᾱ_t)
        pred_x0 = sqrt_recip_alpha_bar_t * (x_t - noise_pred * sqrt_one_minus_alpha_bar_t)

        # 计算后验均值 (DDPM 论文公式(7))
        posterior_mean_coeff1 = (alpha_bar_t_prev.sqrt() * beta_t) / (1.0 - alpha_bar_t)
        posterior_mean_coeff2 = ((1.0 - alpha_bar_t_prev) * torch.sqrt(alpha_bar_t)) / (1.0 - alpha_bar_t)
        posterior_mean = posterior_mean_coeff1 * pred_x0 + posterior_mean_coeff2 * x_t

        # 计算后验方差
        posterior_variance = self.posterior_variance[t].view(-1, 1, 1, 1)

        # 添加噪声（除非是最后一步）
        if t.min() > 0:
            noise = torch.randn_like(x_t)
            x_t_minus_1 = posterior_mean + torch.sqrt(posterior_variance) * noise
        else:
            x_t_minus_1 = posterior_mean

        return x_t_minus_1
    """
    def step_pred(self, noise_pred, t, x_t):
        #预测性去噪：使用预测的噪声从x_t计算x_{t-1}
        batch_size = x_t.shape[0]
        device = x_t.device

        # 获取当前时间步的参数
        betas_t = self.betas[t].to(device).view(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].to(device).view(-1, 1, 1, 1)
        sqrt_recip_alphas_t = self.sqrt_recip_alphas[t].to(device).view(-1, 1, 1, 1)

        # 预测x_0
        pred_x0 = sqrt_recip_alphas_t * (x_t - betas_t * noise_pred / sqrt_one_minus_alphas_cumprod_t)

        # 计算后验均值
        posterior_mean = pred_x0  # 简化版，实际应该考虑后验方差

        # 如果是最后一步，不添加噪声
        if t.min() > 0:
            noise = torch.randn_like(x_t)
            posterior_var = self.posterior_variance[t].to(device).view(-1, 1, 1, 1)
            x_t_minus_1 = posterior_mean + torch.sqrt(posterior_var) * noise
        else:
            x_t_minus_1 = posterior_mean

        return x_t_minus_1

    


    

    """