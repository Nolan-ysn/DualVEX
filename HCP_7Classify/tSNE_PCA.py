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
from sklearn.decomposition import PCA
from timm.models.layers import DropPath
from einops import rearrange
from collections import Counter, defaultdict
import random
import string

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
        num_windows_per_scan = 1
        for fpath in tqdm(self.file_paths, desc="Creating validation samples", leave=False):
            try:
                data_shape = np.load(fpath, mmap_mode='r').shape
                if len(data_shape) != 4 or data_shape[1:] != self.spatial_dim or data_shape[0] < self.window_size:
                    continue
                subject_id = os.path.basename(fpath).split('_')[0]
                base_name = os.path.basename(fpath)
                task_name = base_name.split('_')[1]
                label = self.label_map.get(task_name)
                if label is None: continue
                max_start = data_shape[0] - self.window_size
                possible_starts = list(range(max_start + 1))
                for _ in range(num_windows_per_scan):
                    start_t = random.choice(possible_starts)
                    samples.append((fpath, start_t, subject_id, label))
            except Exception as e:
                print(f"Skipping file {fpath} due to error: {e}")
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fpath, st, subject_id, label = self.samples[idx]
        data = np.load(fpath, mmap_mode='r')
        clip = data[st: st + self.window_size].copy()
        clip_torch = torch.from_numpy(clip).float()
        mean, std = clip_torch.mean(), clip_torch.std()
        clip_norm = (clip_torch - mean) / (std + 1e-6)
        return clip_norm, torch.tensor(label, dtype=torch.long), subject_id


# ==============================================================================
#  步骤 2: 主执行逻辑
# ==============================================================================
if __name__ == "__main__":
    if MHA is None:
        print("无法执行，因为 `flash_attn.modules.mha` 未安装。")
        exit()

    LOG_DIR_PATH = "/home/ysn/PycharmProjects/MAE/HCP_7Classify/result"

    BEST_MODEL_PATH = "/home/ysn/PycharmProjects/MAE/HCP_7Classify/result/HCP_7Class_MAE_ST_200_fine-tuning_20251118_124834/models/best_model_epoch_32_acc_0.9301.pth"
    VAL_FILES_PATH = os.path.join(LOG_DIR_PATH, "validation_files.npy")
    TSNE_SAVE_PATH = os.path.join(LOG_DIR_PATH, "tsne_pca_fine_tuning_final.png")

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
    val_files = [f for f in val_files if "LR" in os.path.basename(f)]
    # 1. 从验证集文件列表中提取所有唯一的被试ID
    val_subject_ids = sorted(list(set(os.path.basename(f).split('_')[0] for f in val_files)))
    print(f"在验证集中共找到 {len(val_subject_ids)} 位唯一的被试。")

    # 2. 随机打乱验证集被试ID的顺序
    # random.shuffle(val_subject_ids)

    # 3. 从打乱后的列表中选择被试
    num_subjects_to_use = 50
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

    # --- 提取特征 (核心修改部分) ---
    all_features_list,all_labels_list, all_subjects_list = [], [], []
    print("开始提取窗口级特征...")
    with torch.no_grad():
        for batch_x, batch_y, batch_subject_ids in tqdm(val_loader, desc="提取特征、标签和被试ID"):
            features = model.forward_features(batch_x.to(device))
            all_features_list.append(features.cpu().numpy())
            all_labels_list.append(batch_y.cpu().numpy())
            all_subjects_list.extend(batch_subject_ids)
    all_features = np.concatenate(all_features_list, axis=0)
    all_labels = np.concatenate(all_labels_list, axis=0)
    final_subjects = all_subjects_list
    print(f"特征提取完毕，特征维度: {all_features.shape}")

    # --- 使用PCA进行预降维 ---
    N, L, E = all_features.shape  # N=总样本数, L=patch数, E=嵌入维度

    # 步骤 1: 重塑 (B, L, E) -> (B*E, L)
    # 首先需要交换L和E轴，然后才能正确地重塑
    features_reshaped = all_features.transpose(0, 2, 1).reshape(N * E, L)

    # 步骤 2: 对空间维度 (L) 进行PCA降维
    num_spatial_components = 64  # 主空间模式数量，可以调整
    print(f"正在对 {L} 维的空间信息进行PCA，降至 {num_spatial_components} 维...")
    pca_spatial = PCA(n_components=num_spatial_components, random_state=42)
    features_pca = pca_spatial.fit_transform(features_reshaped)

    # 步骤 3: 再次重塑 (B*E, k) -> (B, E*k)
    final_features_for_tsne = features_pca.reshape(N, E * num_spatial_components)
    print(f"自定义PCA流程完成。最终用于t-SNE的特征维度: {final_features_for_tsne.shape}")

    # --- 运行 t-SNE 并按任务类别可视化 ---
    print("运行 t-SNE 降维...")
    tsne = TSNE(n_components=2, perplexity=7, n_iter=5000, random_state=42, n_jobs=-1)
    tsne_results = tsne.fit_transform(final_features_for_tsne)
    print("t-SNE 运行完毕。")

    df = pd.DataFrame(
        {'tsne-1': tsne_results[:, 0], 'tsne-2': tsne_results[:, 1],
         'label': all_labels, 'subject': final_subjects})

    # 任务 -> 颜色 映射
    genshin_element_colors = ['#FF4D4D', '#45AAF2', '#46DDC3', '#A955E3', '#A6C938', '#A0E3E8', '#FAB632']
    color_dict = {i: color for i, color in enumerate(genshin_element_colors)}
    df['color'] = df['label'].map(color_dict)

    # 被试 -> 字母 映射
    letter_markers = list(string.ascii_lowercase + string.ascii_uppercase)
    subject_to_letter = {subject_id: letter_markers[i] for i, subject_id in enumerate(selected_subjects)}
    df['letter'] = df['subject'].map(subject_to_letter)

    print("\n--- 诊断信息：请根据以下信息设置手动范围 ---")
    x_coords = df['tsne-1']
    y_coords = df['tsne-2']
    print(f"X 坐标范围: Min={x_coords.min():.2f}, Max={x_coords.max():.2f}")
    print(f"Y 坐标范围: Min={y_coords.min():.2f}, Max={y_coords.max():.2f}")
    print("\n建议的稳健范围 (基于 1% 和 99% 分位数):")
    print(f"X 轴核心数据大约在: ({np.quantile(x_coords, 0.01):.2f}, {np.quantile(x_coords, 0.99):.2f})")
    print(f"Y 轴核心数据大约在: ({np.quantile(y_coords, 0.01):.2f}, {np.quantile(y_coords, 0.99):.2f})")
    print("--------------------------------------------------\n")

    # --- 步骤 2: 手动设置 - 根据上面的诊断信息，在这里填入您想要的范围 ---
    # 第一次运行后，根据上面的输出修改这里的数值。例如，如果X的核心范围是(-40, 50)，您可以设置 (-45, 55)。
    MANUAL_X_LIMITS = (
    np.quantile(x_coords, 0.005) - 10, np.quantile(x_coords, 0.995) + 10)  # <-- 在这里手动设置X轴范围 (最小值, 最大值)
    MANUAL_Y_LIMITS = (
    np.quantile(y_coords, 0.005) - 10, np.quantile(y_coords, 0.995) + 10)  # <-- 在这里手动设置Y轴范围 (最小值, 最大值)

    #  创建图表和坐标轴
    fig, ax = plt.subplots(figsize=(7, 6))

    #  遍历DataFrame中的每一行，并用 ax.text() 绘制字母
    print("正在绘制字母标记点...")
    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="绘制点"):
        x, y = row['tsne-1'], row['tsne-2']
        if MANUAL_X_LIMITS[0] <= x <= MANUAL_X_LIMITS[1] and MANUAL_Y_LIMITS[0] <= y <= MANUAL_Y_LIMITS[1]:
            ax.text(x, y, row['letter'], color=row['color'],
                    fontdict={'weight': 'bold', 'size': 10}, ha='center', va='center', alpha=0.5)

    # **应用您手动设置的坐标轴范围**
    ax.set_xlim(MANUAL_X_LIMITS)
    ax.set_ylim(MANUAL_Y_LIMITS)
    # 3. 设置标题和标签
    ax.set_title('', fontsize=15)
    ax.set_xlabel('X', fontsize=16)
    ax.set_ylabel('Y', fontsize=16)

    # 4. **将图例移动到图表外部**
    #    bbox_to_anchor=(1.05, 1) 的含义:
    #      - x=1.05: 将锚点放在坐标轴宽度102%的位置 (即，在右侧边缘外一点点)
    #      - y=1:    将锚点放在坐标轴高度100%的位置 (即，与顶部对齐)
    #    loc='upper left' 的含义:
    #      - 将图例自身的 "左上角" 对准我们上面设置的锚点
    # ax.legend(title='Task Category',bbox_to_anchor=(1.05, 1),loc='upper left',borderaxespad=0.)

    # 5. 添加网格线
    # ax.grid(True, linestyle='--', alpha=0.5)
    ax.set_xticks([])
    ax.set_yticks([])

    plt.savefig(TSNE_SAVE_PATH, dpi=300, bbox_inches='tight')

    print(f"t-SNE 可视化图像已保存至: {TSNE_SAVE_PATH}")