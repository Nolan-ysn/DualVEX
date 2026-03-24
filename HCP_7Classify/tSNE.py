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
# 步骤 1: 直接从您的 vote.py 文件中导入下游任务模型
# ==============================================================================
# 前提：此脚本需要和 vote.py 在同一个文件夹下
try:
    from vote import EncoderClassifier, ClassifyConfig
except ImportError:
    print("错误：无法从 'vote.py' 导入 EncoderClassifier 或 ClassifyConfig。")
    print("请确保 visualize_final.py 和 vote.py 在同一个目录下。")
    exit()


# ==============================================================================
# 步骤 2: 修改 EncoderClassifier 以方便提取特征
# 我们通过继承的方式，只添加一个用于提取特征的方法，而不改变原有结构
# ==============================================================================

class FeatureExtractor(EncoderClassifier):
    def __init__(self, config):
        super().__init__(config)

    def forward_features(self, x):
        """
        这个新方法执行了从输入到最终norm层的所有步骤，
        但在分类头（head）之前停止，返回 [B, L, E] 形状的特征。
        """
        x = x.unsqueeze(1)
        x = self.time_embedder(x)
        x = self.encoder.patch_embed(x)

        N, L, D_embed = x.shape
        d, h, w = self.encoder.num_patches_d, self.encoder.num_patches_h, self.encoder.num_patches_w

        x = x.reshape(N, d, h, w, D_embed)
        x = x + self.pos_embed_space  # 添加位置编码
        x = x.reshape(N, L, D_embed)

        x = self.pre_dropout(x)

        # 通过 Transformer blocks
        for blk in self.encoder.blocks:
            x = blk(x)

        x = self.encoder.norm(x)
        return x


# ==============================================================================
# 步骤 3: 数据集类，用于加载HCP数据子集
# ==============================================================================
class HCP_Scan_Dataset(Dataset):
    def __init__(self, file_paths, label_map, window_size, spatial_dim):
        self.file_paths = file_paths
        self.label_map = label_map
        self.window_size = window_size
        self.spatial_dim = spatial_dim
        self.samples = self._create_samples()

    def _create_samples(self):
        samples = []
        num_windows_per_scan = 2  # 从每个扫描采样8个窗口用于聚合
        for fpath in tqdm(self.file_paths, desc="Creating dataset samples"):
            try:
                data_shape = np.load(fpath, mmap_mode='r').shape
                if len(data_shape) != 4 or data_shape[1:] != self.spatial_dim or data_shape[0] < self.window_size:
                    continue
                base_name = os.path.basename(fpath)
                task_name = base_name.split('_')[1]
                label = self.label_map.get(task_name)
                if label is None: continue

                max_start = data_shape[0] - self.window_size
                possible_starts = list(range(max_start + 1))
                for _ in range(num_windows_per_scan):
                    start_t = random.choice(possible_starts)
                    samples.append((fpath, start_t, base_name, label))
            except Exception as e:
                print(f"Skipping file {fpath} due to error: {e}")
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fpath, st, scan_id, label = self.samples[idx]
        data = np.load(fpath, mmap_mode='r')
        clip = data[st: st + self.window_size].copy()
        window = torch.from_numpy(clip).float()
        window = (window - window.mean()) / (window.std() + 1e-6)
        return window, torch.tensor(label, dtype=torch.long), scan_id


# ==============================================================================
# 步骤 4: 主执行逻辑
# ==============================================================================
if __name__ == "__main__":
    # ================== 请在这里修改你的路径 ==================
    # 1. 指向你训练好的 MAE 预训练模型
    PRETRAINED_MAE_MODEL_PATH = "/data3/yyy/MAE_base/model/mae_fmri_epoch_47.pth"

    # 2. 指向 HCP 7 分类任务的数据集根目录
    HCP_DATA_ROOT = "/data3/ysn/HCP_Seven_Tasks/"

    # 3. (可选) 可视化结果保存目录
    OUTPUT_DIR = "/home/ysn/PycharmProjects/MAE/HCP_7Classify/result/"
    # ==========================================================

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    TSNE_SAVE_PATH = os.path.join(OUTPUT_DIR, "tsne_hcp_from_mae.png")

    # --- 配置 ---
    cfg = ClassifyConfig()  # 直接使用 vote.py 中的配置类
    cfg.device = "cuda:0" if torch.cuda.is_available() else "cpu"  # 可以覆盖配置
    device = torch.device(cfg.device)

    tasks = ['EMOTION', 'GAMBLING', 'LANGUAGE', 'MOTOR', 'RELATIONAL', 'SOCIAL', 'WM']
    label_map = {task: i for i, task in enumerate(tasks)}
    print(f"使用设备: {device}")

    # --- 实例化下游模型结构 ---
    print("正在创建下游任务模型结构 (EncoderClassifier)...")
    model = FeatureExtractor(cfg).to(device)

    # --- 加载预训练权重 ---
    print(f"正在从 '{PRETRAINED_MAE_MODEL_PATH}' 加载 MAE 预训练权重...")
    checkpoint = torch.load(PRETRAINED_MAE_MODEL_PATH, map_location=device)

    # strict=False 意味着我们只加载那些键名匹配的权重。
    # 分类头(head)的权重在预训练模型中不存在，所以会被自动忽略，这正是我们想要的！
    missing_keys, unexpected_keys = model.load_state_dict(checkpoint['encoder_state_dict'], strict=False)
    print(f"权重加载完毕。")
    print(f"缺失的键 (应为分类头): {missing_keys}")
    print(f"意外的键 (应为空): {unexpected_keys}")

    model.eval()

    # --- 准备 HCP 数据子集 ---
    print(f"正在从 '{HCP_DATA_ROOT}' 加载 HCP 数据...")
    all_hcp_files = glob.glob(os.path.join(HCP_DATA_ROOT, '*.npy'))
    if not all_hcp_files: raise FileNotFoundError(f"在 '{HCP_DATA_ROOT}' 中未找到.npy文件。")

    subject_files = defaultdict(list)
    for f in all_hcp_files:
        subject_files[os.path.basename(f).split('_')[0]].append(f)

    all_subjects = list(subject_files.keys())
    random.shuffle(all_subjects)

    selected_subjects = all_subjects  # 选用所有被试
    final_file_list = [f for subj_id in selected_subjects for f in subject_files[subj_id]]
    print(f"已随机选择 {len(selected_subjects)} 位被试，共计 {len(final_file_list)} 个扫描文件。")

    dataset = HCP_Scan_Dataset(final_file_list, label_map, cfg.window_size, cfg.spatial_dim)
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=8)
    print(f"数据集准备完毕，共生成 {len(dataset)} 个窗口样本。")

    # --- 提取窗口级特征 ---
    scan_features = defaultdict(list)
    scan_labels = {}
    print("Extracting and aggregating features at scan level...")
    with torch.no_grad():
        for batch_x, batch_y, batch_scan_ids in tqdm(loader, desc="Extracting features"):
            batch_x = batch_x.to(device)
            features = model.forward_features(batch_x)  # (B, L, E)
            pooled_features = torch.mean(features, dim=1)  # (B, E)

            for i, scan_id in enumerate(batch_scan_ids):
                scan_features[scan_id].append(pooled_features[i].cpu().numpy())
                if scan_id not in scan_labels:
                    scan_labels[scan_id] = batch_y[i].item()

    # --- 对每个扫描的特征进行最终聚合 ---
    final_features, final_labels = [], []
    for scan_id, features_list in tqdm(scan_features.items(), desc="Aggregating scan features"):
        aggregated_feature = np.mean(np.array(features_list), axis=0)
        final_features.append(aggregated_feature)
        final_labels.append(scan_labels[scan_id])

    final_features = np.array(final_features)
    final_labels = np.array(final_labels)
    print(f"Feature aggregation complete. Final number of samples for t-SNE: {final_features.shape[0]}")

    # --- 运行 t-SNE 并可视化 ---
    print("Running t-SNE dimensionality reduction...")
    tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42, n_jobs=-1)
    tsne_results = tsne.fit_transform(final_features)
    print("t-SNE complete.")

    df = pd.DataFrame(
        {'tsne-1': tsne_results[:, 0], 'tsne-2': tsne_results[:, 1], 'label': [tasks[i] for i in final_labels]})

    plt.figure(figsize=(14, 12))

    sns.scatterplot(
        x="tsne-1", y="tsne-2", hue="label",
        palette=sns.color_palette("hsv", len(tasks)),
        data=df, legend="full", alpha=0.5, s=15
    )
    plt.title('t-SNE of Aggregated Scan-Level Features from Pre-trained MAE', fontsize=18)
    plt.xlabel('t-SNE Dimension 1', fontsize=14)
    plt.ylabel('t-SNE Dimension 2', fontsize=14)
    plt.legend(title='Task Category', fontsize=12, title_fontsize=13)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()

    plt.savefig(TSNE_SAVE_PATH, dpi=300)
    print(f"t-SNE 可视化图像已保存至: {TSNE_SAVE_PATH}")