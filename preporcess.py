import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm


def preprocess_image(img_path, target_size=128):
    """读取并预处理单张图像"""
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"无法读取图像：{img_path}")
    img = cv2.resize(img, (target_size, target_size))
    return img.astype(np.float32) / 255.0  # 归一化到[0,1]


def batch_preprocess(csv_path, output_dir):
    """批量处理所有数据"""
    df = pd.read_csv(csv_path)
    os.makedirs(output_dir, exist_ok=True)

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        try:
            # 处理甲骨文图像
            oracle_img = preprocess_image(row['oracle_path'])
            np.save(os.path.join(output_dir, f"{row['label']}_{idx}_oracle.npy"), oracle_img)

            # 处理现代汉字图像
            hanzi_img = preprocess_image(row['hanzi_path'])
            np.save(os.path.join(output_dir, f"{row['label']}_{idx}_hanzi.npy"), hanzi_img)
        except Exception as e:
            print(f"处理 {row['oracle_path']} 失败: {str(e)}")
            continue


if __name__ == "__main__":
    batch_preprocess("data\data_index.csv", "processed_data")