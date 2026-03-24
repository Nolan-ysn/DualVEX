import os
import glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from collections import defaultdict
import random

# ==============================================================================
# 步骤 1: 导入下游任务模型结构
# ==============================================================================
try:
    from vote import EncoderClassifier, ClassifyConfig
except ImportError:
    print("错误：无法从 'vote.py' 导入所需类。请确保此脚本与 vote.py 在同一目录下。")
    exit()


# ==============================================================================
# 步骤 2: 创建一个包装类，用于提取特征
# ==============================================================================
class FeatureExtractor(EncoderClassifier):
    def __init__(self, config):
        super().__init__(config)

    def forward_features(self, x):
        x = x.unsqueeze(1)
        x = self.time_embedder(x)
        x = self.encoder.patch_embed(x)
        N, L, D_embed = x.shape
        d, h, w = self.encoder.num_patches_d, self.encoder.num_patches_h, self.encoder.num_patches_w
        x = x.reshape(N, d, h, w, D_embed) + self.pos_embed_space
        x = x.reshape(N, L, D_embed)
        x = self.pre_dropout(x)
        for blk in self.encoder.blocks: x = blk(x)
        return self.encoder.norm(x)


# ==============================================================================
# 步骤 3: 数据集类，现在需要同时返回 扫描ID 和 被试ID
# ==============================================================================
class HCP_Subject_Dataset(Dataset):
    def __init__(self, file_paths, window_size, spatial_dim):
        self.file_paths = file_paths
        self.window_size = window_size
        self.spatial_dim = spatial_dim
        self.samples = self._create_samples()

    def _create_samples(self):
        samples = []
        num_windows_per_scan = 4
        for fpath in tqdm(self.file_paths, desc="创建数据集样本"):
            try:
                data_shape = np.load(fpath, mmap_mode='r').shape
                if len(data_shape) != 4 or data_shape[1:] != self.spatial_dim or data_shape[0] < self.window_size:
                    continue
                base_name = os.path.basename(fpath)
                subject_id = base_name.split('_')[0]

                max_start = data_shape[0] - self.window_size
                possible_starts = list(range(max_start + 1))
                for _ in range(num_windows_per_scan):
                    start_t = random.choice(possible_starts)
                    samples.append((fpath, start_t, base_name, subject_id))
            except Exception as e:
                print(f"跳过文件 {fpath}，原因: {e}")
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fpath, st, scan_id, subject_id = self.samples[idx]
        data = np.load(fpath, mmap_mode='r')
        clip = data[st: st + self.window_size].copy()
        window = torch.from_numpy(clip).float()
        window = (window - window.mean()) / (window.std() + 1e-6)
        return window, scan_id, subject_id


# ==============================================================================
# 步骤 4: 主执行逻辑
# ==============================================================================
if __name__ == "__main__":
    # ================== 路径 ==================
    PRETRAINED_MAE_MODEL_PATH = "/data3/yyy/MAE_st_1/model/mae_fmri_epoch_26.pth"

    HCP_DATA_ROOT = "/data3/ysn/HCP_Seven_Tasks/"

    OUTPUT_DIR = "/home/ysn/PycharmProjects/MAE/HCP_7Classify/result/"
    # ==========================================================

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    TSNE_SAVE_PATH = os.path.join(OUTPUT_DIR, "tsne_hcp_colored_by_subject.png")

    # --- 配置 ---
    cfg = ClassifyConfig()
    cfg.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(cfg.device)
    print(f"使用设备: {device}")

    # --- 实例化并加载模型 ---
    print("创建下游任务模型结构 (EncoderClassifier)...")
    model = FeatureExtractor(cfg).to(device)
    print(f"从 '{PRETRAINED_MAE_MODEL_PATH}' 加载预训练权重...")
    checkpoint = torch.load(PRETRAINED_MAE_MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint['encoder_state_dict'], strict=False)
    model.eval()
    print("权重加载成功。")

    # --- 准备 HCP 数据子集 ---
    print(f"从 '{HCP_DATA_ROOT}' 加载HCP数据...")
    all_hcp_files = glob.glob(os.path.join(HCP_DATA_ROOT, '*.npy'))
    if not all_hcp_files: raise FileNotFoundError(f"在 '{HCP_DATA_ROOT}' 中未找到.npy文件。")

    subject_files = defaultdict(list)
    for f in all_hcp_files: subject_files[os.path.basename(f).split('_')[0]].append(f)

    all_subjects = list(subject_files.keys())
    random.shuffle(all_subjects)

    selected_subjects = all_subjects[:50]
    final_file_list = [f for subj_id in selected_subjects for f in subject_files[subj_id]]
    print(f"已随机选择 {len(selected_subjects)} 位被试，共计 {len(final_file_list)} 个扫描文件。")

    dataset = HCP_Subject_Dataset(final_file_list, cfg.window_size, cfg.spatial_dim)
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=8)
    print(f"数据集已创建，共 {len(dataset)} 个窗口样本。")

    # --- 提取并按扫描聚合特征 ---
    scan_features = defaultdict(list)
    scan_to_subject = {}
    print("开始提取并聚合特征...")
    with torch.no_grad():
        for batch_x, batch_scan_ids, batch_subject_ids in tqdm(loader, desc="提取特征"):
            batch_x = batch_x.to(device)
            features = model.forward_features(batch_x)
            pooled_features = torch.mean(features, dim=1)

            for i, scan_id in enumerate(batch_scan_ids):
                scan_features[scan_id].append(pooled_features[i].cpu().numpy())
                if scan_id not in scan_to_subject:
                    scan_to_subject[scan_id] = batch_subject_ids[i]

    # --- 对每个扫描的特征进行最终聚合 ---
    final_features, final_subjects = [], []
    for scan_id, features_list in tqdm(scan_features.items(), desc="聚合扫描特征"):
        aggregated_feature = np.mean(np.array(features_list), axis=0)
        final_features.append(aggregated_feature)
        final_subjects.append(scan_to_subject[scan_id])

    final_features = np.array(final_features)
    print(f"特征聚合完毕。最终用于t-SNE的样本数: {final_features.shape[0]}")

    # --- 运行 t-SNE 并按被试ID可视化 ---
    print("运行 t-SNE 降维...")
    tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42, n_jobs=-1)
    tsne_results = tsne.fit_transform(final_features)
    print("t-SNE 运行完毕。")

    df = pd.DataFrame({
        'tsne-1': tsne_results[:, 0],
        'tsne-2': tsne_results[:, 1],
        'subject': final_subjects
    })

    plt.figure(figsize=(14, 12))
    sns.scatterplot(
        x="tsne-1", y="tsne-2", hue="subject",
        palette=sns.color_palette("hsv", len(selected_subjects)),  # 使用hsv色板以获得更多不同颜色
        data=df,
        legend=False,  # **禁用图例，因为它会非常大**
        alpha=0.6,
        s=30
    )
    plt.title('t-SNE of Scan-Level Features, Colored by Subject ID', fontsize=18)
    plt.xlabel('t-SNE Dimension 1', fontsize=14)
    plt.ylabel('t-SNE Dimension 2', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()

    plt.savefig(TSNE_SAVE_PATH, dpi=300)
    print(f"t-SNE 可视化图像已保存至: {TSNE_SAVE_PATH}")