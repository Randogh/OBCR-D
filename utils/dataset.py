import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import pandas as pd
import torchvision.transforms as transforms


class OracleToModernDataset(Dataset):
    def __init__(self, data_dir, songti_dir, labels_path, split="train", img_size=64):
        self.data_dir = data_dir
        self.songti_dir = songti_dir
        self.img_size = img_size

        # 加载标签
        self.labels_df = pd.read_csv(labels_path)
        print(f"加载标签文件: {labels_path}")
        print(f"标签数量: {len(self.labels_df)}")

        # 分割训练/测试集
        if split == "train":
            self.labels_df = self.labels_df.sample(frac=0.8, random_state=42)
        else:
            full_df = pd.read_csv(labels_path)
            train_df = full_df.sample(frac=0.8, random_state=42)
            self.labels_df = full_df[~full_df.index.isin(train_df.index)]

        # 图像转换
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        try:
            # 获取文件名和标签
            row = self.labels_df.iloc[idx]
            oracle_filename = row["image_path"]
            modern_char = row["glyph_char"]

            print(f"加载甲骨文图像: {oracle_filename}, 对应现代字: {modern_char}")

            # 加载甲骨文图像
            oracle_path = os.path.join(self.data_dir, oracle_filename)
            if not os.path.exists(oracle_path):
                print(f"错误: 甲骨文图像不存在: {oracle_path}")
                return None

            oracle_img = Image.open(oracle_path).convert("L")
            oracle_tensor = self.transform(oracle_img)

            # 加载现代汉字图像
            modern_path = os.path.join(self.songti_dir, f"{modern_char}.png")
            if not os.path.exists(modern_path):
                print(f"错误: 宋体字图像不存在: {modern_path}")
                return None

            modern_img = Image.open(modern_path).convert("L")
            modern_tensor = self.transform(modern_img)

            return {
                "oracle": oracle_tensor,
                "modern": modern_tensor,
                "char": modern_char
            }
        except Exception as e:
            print(f"加载数据时出错: {e}")
            # 在出错时返回None会导致DataLoader出现问题
            # 应该返回默认值或跳过这个样本
            return {
                "oracle": torch.zeros(1, self.img_size, self.img_size),
                "modern": torch.zeros(1, self.img_size, self.img_size),
                "char": "错误"
            }