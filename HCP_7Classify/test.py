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
from timm.models.layers import DropPath
from collections import Counter, defaultdict
import random

# ==============================================================================
#  步骤 1: 从你的 `vote.py` 脚本中复制必要的类和配置
# (为了让这个脚本可以独立运行，我们需要这些定义)
# ==============================================================================

# 尝试导入 flash_attn
try:
    from flash_attn.modules.mha import MHA
except ImportError:
    MHA = None
    print("Warning: flash_attn.modules.mha not found. FlashAttention will not be used.")


# ------------------- 1.1 配置 (与 vote.py 保持一致) -------------------
class ClassifyConfig:
    data_root = "/data3/ysn/HCP_Seven_Tasks/"
    spatial_dim = (80, 96, 80)
    window_size = 16
    embed_dim_intermediate = 16
    patch_size = (8, 8, 8)
    embed_dim = 1024
    depth = 6
    num_heads = 8
    mlp_ratio = 4.0
    num_classes = 7
    dropout = 0.1
    batch_size = 16  # 可根据显存大小调整
    num_workers = 8
    device = 'cuda:0'  # 可根据你的可用GPU进行修改


# ------------------- 1.2 模型组件 (直接从 vote.py 复制) -------------------
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.1):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))


class FlashAttention(nn.Module):
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super().__init__()
        if MHA is None:
            raise ImportError("flash_attn.modules.mha is required for FlashAttention.")
        self.attn = MHA(embed_dim=dim, num_heads=num_heads, dropout=dropout)

    def forward(self, x):
        return self.attn(x)


class Block3D(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, drop=0.1, attn_drop=0.1, drop_path=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = FlashAttention(dim, num_heads, dropout=attn_drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, int(dim * mlp_ratio), drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        return x + self.drop_path(self.mlp(self.norm2(x)))


class PatchEmbed3D(nn.Module):
    def __init__(self, patch_size, in_chans, embed_dim, spatial_dim):
        super().__init__()
        self.patch_size = patch_size
        pd, ph, pw = patch_size
        D, H, W = spatial_dim
        self.num_patches_d = D // pd;
        self.num_patches_h = H // ph;
        self.num_patches_w = W // pw
        self.num_patches = self.num_patches_d * self.num_patches_h * self.num_patches_w
        self.patch_volume = pd * ph * pw * in_chans
        self.proj = nn.Linear(self.patch_volume, embed_dim)

    def forward(self, x):
        N, C, D, H, W = x.shape
        pd, ph, pw = self.patch_size
        x = x.reshape(N, C, D // pd, pd, H // ph, ph, W // pw, pw).permute(0, 2, 4, 6, 1, 3, 5, 7)
        return self.proj(x.reshape(N, -1, self.patch_volume))


class TimeEmbedModule(nn.Module):
    def __init__(self, time_steps, embed_dim_intermediate=16):
        super().__init__()
        self.embed_dim = embed_dim_intermediate
        self.time_embedder = nn.Sequential(
            nn.Conv3d(time_steps, embed_dim_intermediate, kernel_size=1),
            nn.BatchNorm3d(embed_dim_intermediate), nn.ReLU())

    def forward(self, x):
        N, C, T, D, H, W = x.shape
        x = x.reshape(N * C, T, D, H, W)
        x = self.time_embedder(x)
        return x.reshape(N, self.embed_dim, D, H, W)


class VisionTransformer3D(nn.Module):
    def __init__(self, patch_size, in_chans, embed_dim, depth, num_heads, mlp_ratio, spatial_dim, dropout):
        super().__init__()
        self.patch_embed = PatchEmbed3D(patch_size, in_chans, embed_dim, spatial_dim)
        self.num_patches_d = self.patch_embed.num_patches_d;
        self.num_patches_h = self.patch_embed.num_patches_h
        self.num_patches_w = self.patch_embed.num_patches_w;
        self.num_patches = self.patch_embed.num_patches
        self.blocks = nn.ModuleList([Block3D(embed_dim, num_heads, mlp_ratio, drop=dropout) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.patch_embed(x)
        x = self.dropout(x)
        for blk in self.blocks: x = blk(x)
        return self.norm(x)


class EncoderClassifier(nn.Module):
    def __init__(self, config: ClassifyConfig):
        super().__init__()
        self.config = config
        self.time_embedder = TimeEmbedModule(config.window_size, config.embed_dim_intermediate)
        self.encoder = VisionTransformer3D(
            config.patch_size, config.embed_dim_intermediate, config.embed_dim, config.depth,
            config.num_heads, config.mlp_ratio, config.spatial_dim, config.dropout)
        d, h, w = self.encoder.num_patches_d, self.encoder.num_patches_h, self.encoder.num_patches_w
        self.pos_embed_space = nn.Parameter(torch.zeros(d, h, w, config.embed_dim))
        nn.init.trunc_normal_(self.pos_embed_space, std=0.05)
        self.pre_dropout = nn.Dropout(config.dropout)
        self.head = nn.Linear(config.embed_dim * self.encoder.num_patches, config.num_classes)

    def forward_features(self, x):
        """ 提取 backbone 特征的辅助函数 """
        x = x.unsqueeze(1)
        x = self.time_embedder(x)
        x = self.encoder.patch_embed(x)
        N, L, D_embed = x.shape
        d, h, w = self.encoder.num_patches_d, self.encoder.num_patches_h, self.encoder.num_patches_w
        x = x.reshape(N, d, h, w, D_embed)
        x = x + self.pos_embed_space
        x = x.reshape(N, L, D_embed)
        x = self.pre_dropout(x)
        for blk in self.encoder.blocks: x = blk(x)
        return self.encoder.norm(x)

    def forward(self, x):
        x = self.forward_features(x)
        x = x.flatten(start_dim=1)
        return self.head(x)


# ------------------- 1.3 数据集 (直接从 vote.py 复制) -------------------
class BlockClassificationDataset(Dataset):
    def __init__(self, file_paths, label_map, window_size=16, spatial_dim=(80, 96, 80), mode='val'):
        self.file_paths = file_paths
        self.label_map = label_map
        self.window_size = window_size
        self.spatial_dim = spatial_dim
        self.samples = self._create_samples()

    def _create_samples(self):
        samples = []
        num_windows_per_scan = 8
        for fpath in tqdm(self.file_paths, desc="Creating validation samples", leave=False):
            try:
                data = np.load(fpath, mmap_mode='r')
                if data.shape[1:] != self.spatial_dim or data.shape[0] < self.window_size: continue
                base_name = os.path.basename(fpath)
                task_name = base_name.split('_')[1]
                label = self.label_map.get(task_name)
                if label is None: continue
                max_start = data.shape[0] - self.window_size
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
        clip_torch = torch.from_numpy(clip).float()
        mean, std = clip_torch.mean(), clip_torch.std()
        clip_norm = (clip_torch - mean) / (std + 1e-6)
        return clip_norm, torch.tensor(label, dtype=torch.long), scan_id


# ==============================================================================
#  步骤 2: 主执行逻辑
# ==============================================================================
if __name__ == "__main__":
    if MHA is None:
        print("无法执行，因为 `flash_attn.modules.mha` 未安装。")
        exit()

    LOG_DIR_PATH = "/home/ysn/PycharmProjects/MAE/HCP_7Classify/result"

    BEST_MODEL_PATH = "/home/ysn/PycharmProjects/MAE/HCP_7Classify/result/HCP_7Class_MAE_200_fine-tuning_20251005_173401/models/best_model_epoch_98_acc_0.9110.pth"
    VAL_FILES_PATH = os.path.join(LOG_DIR_PATH, "validation_files.npy")
    TSNE_SAVE_PATH = os.path.join(LOG_DIR_PATH, "tsne_visualization_10_fine_tuning.png")

    # --- 设置 ---
    cfg = ClassifyConfig()
    cfg.device = "cuda:4" if torch.cuda.is_available() else "cpu"
    device = torch.device(cfg.device)
    tasks = ['EMOTION', 'GAMBLING', 'LANGUAGE', 'MOTOR', 'RELATIONAL', 'SOCIAL', 'WM']
    label_map = {task: i for i, task in enumerate(tasks)}
    print(f"使用设备: {device}")

    # --- 加载模型 ---
    print(f"正在从 '{BEST_MODEL_PATH}' 加载模型...")
    model = EncoderClassifier(cfg)
    model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
    print("模型加载成功。")

    # --- 准备数据 ---
    print(f"--- 准备 HCP 数据子集 (从验证集) ---")
    try:
        print(f"正在从 '{VAL_FILES_PATH}' 加载验证集文件列表...")
        val_files = np.load(VAL_FILES_PATH, allow_pickle=True)
    except FileNotFoundError:
        print(f"错误: 验证集文件 '{VAL_FILES_PATH}' 未找到！请确保路径正确。")
        exit()

    # 1. 从验证集文件列表中提取所有唯一的被试ID
    val_subject_ids = sorted(list(set(os.path.basename(f).split('_')[0] for f in val_files)))
    print(f"在验证集中共找到 {len(val_subject_ids)} 位唯一的被试。")

    # 2. 随机打乱验证集被试ID的顺序
    random.shuffle(val_subject_ids)

    # 3. 从打乱后的列表中选择被试
    num_subjects_to_use = 10
    if len(val_subject_ids) < num_subjects_to_use:
        print(f"警告: 验证集被试数量 ({len(val_subject_ids)}) 少于期望的 {num_subjects_to_use}。将使用所有找到的被试。")
        num_subjects_to_use = len(val_subject_ids)

    selected_subjects = val_subject_ids[:num_subjects_to_use]
    selected_subjects_set = set(selected_subjects)  # 转换为集合以便快速查找
    print(f"已从验证集中随机选择 {len(selected_subjects)} 位被试进行分析。")

    # 4. 过滤原始的验证集文件列表，只保留属于被选中被试的文件
    final_file_list = [f for f in val_files if os.path.basename(f).split('_')[0] in selected_subjects_set]
    print(f"已筛选出属于这 {len(selected_subjects)} 位被试的 {len(final_file_list)} 个扫描文件。")
    val_dataset = BlockClassificationDataset(final_file_list, label_map, cfg.window_size, cfg.spatial_dim)
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)
    print(f"验证集准备完毕，共 {len(val_dataset)} 个窗口样本。")

    # --- 提取并聚合特征 ---
    all_features = []
    all_labels = []
    print("开始提取特征...")
    with torch.no_grad():
        for batch_x, batch_y, batch_scan_ids in tqdm(val_loader, desc="提取特征"):
            batch_x = batch_x.to(device)
            # 1. 提取 backbone 特征 (B, L, E)
            features = model.forward_features(batch_x)
            # 2. 对每个窗口的 patch 特征进行全局平均池化 (B, L, E) -> (B, E)
            pooled_features = torch.mean(features, dim=1)

            all_features.append(pooled_features.cpu().numpy())
            all_labels.append(batch_y.cpu().numpy())

    all_features = np.concatenate(all_features, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    print(f"特征提取完毕。最终用于t-SNE的样本数: {all_features.shape[0]}")

    # --- 运行 t-SNE 并可视化 ---
    print("正在运行 t-SNE 降维...(这可能需要几分钟)")
    tsne = TSNE(n_components=2, perplexity=50, n_iter=5000, learning_rate=300, random_state=42, n_jobs=-1)
    tsne_results = tsne.fit_transform(all_features)
    print("t-SNE 运行完毕。")

    df = pd.DataFrame({
        'tsne-1': tsne_results[:, 0],
        'tsne-2': tsne_results[:, 1],
        'label': [tasks[i] for i in all_labels]
    })

    # 1. 定义《原神》七元素的十六进制颜色码列表
    genshin_element_colors = [
        '#FF4D4D',  # 火元素 (Pyro) - 鲜红
        '#45AAF2',  # 水元素 (Hydro) - 天蓝
        '#46DDC3',  # 风元素 (Anemo) - 青绿
        '#A955E3',  # 雷元素 (Electro) - 电紫
        '#A6C938',  # 草元素 (Dendro) - 草绿
        '#A0E3E8',  # 冰元素 (Cryo) - 冰蓝
        '#FAB632'  # 岩元素 (Geo) - 岩金
    ]

    # 2. 创建从任务名称到元素颜色的映射字典
    #    您可以根据自己的喜好和研究内容，调整任务和元素颜色的对应关系。
    #    当前默认对应关系:
    #    - EMOTION    -> 火 (红)
    #    - GAMBLING   -> 水 (蓝)
    #    - LANGUAGE   -> 风 (青)
    #    - MOTOR      -> 雷 (紫)
    #    - RELATIONAL -> 草 (绿)
    #    - SOCIAL     -> 冰 (浅蓝)
    #    - WM         -> 岩 (金)
    color_dict = {task: color for task, color in zip(tasks, genshin_element_colors)}

    plt.figure(figsize=(14, 12))
    sns.scatterplot(
        x="tsne-1", y="tsne-2",
        hue="label",
        palette=color_dict,
        data=df,
        legend="full",
        alpha=0.9,
        s=70
    )
    plt.title('t-SNE of Aggregated Scan-Level Features', fontsize=20)
    plt.xlabel('t-SNE Dimension 1', fontsize=16)
    plt.ylabel('t-SNE Dimension 2', fontsize=16)
    plt.legend(title='Task Category', fontsize=12, title_fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()

    # 保存图像
    plt.savefig(TSNE_SAVE_PATH, dpi=300)
    print(f"t-SNE 可视化图像已保存至: {TSNE_SAVE_PATH}")