import os
import glob
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, f1_score, confusion_matrix,
                             precision_recall_fscore_support)
from scipy.special import softmax
from timm.models.layers import DropPath
from einops import rearrange
import itertools
from torch.cuda.amp import autocast

# 保持绘图风格一致
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Helvetica', 'Bitstream Vera Sans']
plt.rcParams['font.size'] = 14

try:
    from flash_attn.modules.mha import MHA
except ImportError:
    MHA = None
    print("Warning: flash_attn.modules.mha not found.")


# ==================================================
# 1. 配置
# ==================================================
class TestConfig:
    # --- 路径设置 ---
    data_root = "/data3/ysn/IBC_for_test/"

    # 请修改为您要测试的具体模型路径
    model_path ="/data3/ysn/LoRA_HCP/HCP_7Class_MAE_ST_20_LoRA_20260106_194808/models/best_model_epoch_55_acc_0.7564.pth"
    # MAE预训练的模型
    '''
    "/home/ysn/PycharmProjects/MAE/HCP_7Classify/result/HCP_7Class_MAE_ST_20_fine-tuning_20251125_103931/models/best_model_epoch_60_acc_0.7770.pth"
    "/home/ysn/PycharmProjects/MAE/HCP_7Classify/result/HCP_7Class_MAE_ST_50_fine-tuning_20251125_210837/models/best_model_epoch_88_acc_0.8602.pth"
    "/home/ysn/PycharmProjects/MAE/HCP_7Classify/result/HCP_7Class_MAE_ST_100_fine-tuning_20251129_190801/models/best_model_epoch_81_acc_0.9051.pth"
    "/home/ysn/PycharmProjects/MAE/HCP_7Classify/result/HCP_7Class_MAE_ST_200_fine-tuning_20251118_124834/models/best_model_epoch_32_acc_0.9301.pth"
    "/home/ysn/PycharmProjects/MAE/HCP_7Classify/result/HCP_7Class_MAE_ST_900_fine-tuning_20251119_151830/models/best_model_epoch_16_acc_0.9617.pth"
    '''
    # 从头训练的模型
    '''
    "/home/ysn/PycharmProjects/MAE/HCP_7Classify/result/HCP_7Class_20_from_scratch_20250928_134810/models/best_model_epoch_12_acc_0.3429.pth"
    "/home/ysn/PycharmProjects/MAE/HCP_7Classify/result/HCP_7Class_50_from_scratch_20250924_210355/models/best_model_epoch_60_acc_0.7498.pth"
    "/home/ysn/PycharmProjects/MAE/HCP_7Classify/result/HCP_7Class_100_from_scratch_20250925_102324/models/best_model_epoch_32_acc_0.8683.pth"
    "/home/ysn/PycharmProjects/MAE/HCP_7Classify/result/HCP_7Class_200_from_scratch_20250925_103521/models/best_model_epoch_47_acc_0.9191.pth"
    "/home/ysn/PycharmProjects/MAE/HCP_7Classify/result/HCP_7Class_900_from_scratch_20251003_151347/models/best_model_epoch_52_acc_0.9558.pth"
    '''
    # 冻结预训练参数的模型
    '''
    "/home/ysn/PycharmProjects/MAE/HCP_7Classify/result/HCP_7Class_MAE_ST_20_frozen_20251229_182212/models/best_model_epoch_62_acc_0.5114.pth"
    "/home/ysn/PycharmProjects/MAE/HCP_7Classify/result/HCP_7Class_MAE_ST_50_frozen_20251230_143318/models/best_model_epoch_64_acc_0.6365.pth"
    "/home/ysn/PycharmProjects/MAE/HCP_7Classify/result/HCP_7Class_MAE_ST_100_frozen_20251229_204238/models/best_model_epoch_41_acc_0.7167.pth"
    "/home/ysn/PycharmProjects/MAE/HCP_7Classify/result/HCP_7Class_MAE_ST_900_frozen_20251231_154700/models/best_model_epoch_29_acc_0.8241.pth"
    '''
    log_dir = "/home/ysn/PycharmProjects/MAE/IBC/result/"

    num_subjects = 20 # 微调时使用的被试数量
    # --- 模型参数 (保持一致) ---
    spatial_dim = (80, 96, 80)
    window_size = 16
    embed_dim_intermediate = 16
    patch_size = (8, 8, 8)
    embed_dim = 1024
    depth = 6
    num_heads = 8
    mlp_ratio = 4.0
    num_classes = 7
    dropout = 0.0

    # --- 推理参数 ---
    batch_size = 16
    num_workers = 8
    device = 'cuda:2'


os.makedirs(TestConfig.log_dir, exist_ok=True)


# ==================================================
# 2. 辅助函数 & 模型定义 (保持不变)
# ==================================================
def get_logger(filename):
    logger = logging.getLogger(filename)
    logger.setLevel(logging.INFO)
    if logger.hasHandlers(): logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s %(message)s")
    fh = logging.FileHandler(filename, mode='a');
    fh.setFormatter(formatter);
    logger.addHandler(fh)
    sh = logging.StreamHandler();
    sh.setFormatter(formatter);
    logger.addHandler(sh)
    return logger


def save_confusion_matrix(cm, classes, save_path, normalize=True, title='Confusion Matrix'):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(10, 8))
    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        cm_norm = cm.astype('float') / row_sums * 100
    else:
        cm_norm = cm
    ax = plt.subplot(111)
    cax = ax.imshow(cm_norm, cmap='GnBu', vmin=0, vmax=100 if normalize else None)
    cbar = plt.colorbar(cax)
    plt.title(title, fontsize=16);
    plt.xlabel('Predicted', fontsize=14);
    plt.ylabel('True', fontsize=14)
    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks);
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(classes, rotation=45, ha="right");
    ax.set_yticklabels(classes)
    fmt = '.1f' if normalize else 'd'
    thresh = cm_norm.max() / 2.
    for i, j in itertools.product(range(cm_norm.shape[0]), range(cm_norm.shape[1])):
        val_text = f"{cm_norm[i, j]:{fmt}}" + ('%' if normalize else '')
        plt.text(j, i, val_text, ha="center", va="center", color="white" if cm_norm[i, j] > thresh else "black")
    plt.tight_layout();
    plt.savefig(save_path, dpi=300);
    plt.close()


# --- 模型组件 (必须与训练代码一致) ---
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.1):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU();
        self.fc2 = nn.Linear(hidden_features, out_features);
        self.drop = nn.Dropout(drop)

    def forward(self, x): return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))


class FlashAttention(nn.Module):
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super().__init__()
        if MHA is None: raise ImportError("flash_attn required")
        self.attn = MHA(embed_dim=dim, num_heads=num_heads, dropout=dropout)

    def forward(self, x): return self.attn(x)


class Block3D(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, drop=0.1, attn_drop=0.1, drop_path=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim);
        self.attn = FlashAttention(dim, num_heads, dropout=attn_drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim);
        self.mlp = Mlp(dim, int(dim * mlp_ratio), drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        return x + self.drop_path(self.mlp(self.norm2(x)))


class PatchEmbed3D(nn.Module):
    def __init__(self, patch_size, in_chans, embed_dim, spatial_dim):
        super().__init__()
        self.patch_size = patch_size;
        pd, ph, pw = patch_size;
        D, H, W = spatial_dim
        self.num_patches = (D // pd) * (H // ph) * (W // pw)
        self.proj = nn.Linear(pd * ph * pw * in_chans, embed_dim)

    def forward(self, x):
        N, C, D, H, W = x.shape;
        pd, ph, pw = self.patch_size
        x = x.reshape(N, C, D // pd, pd, H // ph, ph, W // pw, pw).permute(0, 2, 4, 6, 1, 3, 5, 7)
        return self.proj(x.reshape(N, -1, pd * ph * pw * C))


class TimeEmbedModule(nn.Module):
    def __init__(self, time_steps, embed_dim_intermediate=16):
        super().__init__()
        self.embed_dim = embed_dim_intermediate
        self.time_embedder = nn.Sequential(nn.Conv3d(time_steps, embed_dim_intermediate, 1),
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
        self.num_patches = self.patch_embed.num_patches
        self.blocks = nn.ModuleList([Block3D(embed_dim, num_heads, mlp_ratio, drop=dropout) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim);
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.dropout(self.patch_embed(x))
        for blk in self.blocks: x = blk(x)
        return self.norm(x)


class EncoderClassifier(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.time_embedder = TimeEmbedModule(config.window_size, config.embed_dim_intermediate)
        self.encoder = VisionTransformer3D(config.patch_size, config.embed_dim_intermediate, config.embed_dim,
                                           config.depth, config.num_heads, config.mlp_ratio, config.spatial_dim,
                                           config.dropout)
        d, h, w = config.spatial_dim[0] // config.patch_size[0], config.spatial_dim[1] // config.patch_size[1], \
                  config.spatial_dim[2] // config.patch_size[2]
        self.pos_embed_space = nn.Parameter(torch.zeros(d, h, w, config.embed_dim))
        self.pre_dropout = nn.Dropout(config.dropout)
        self.head = nn.Linear(config.embed_dim * self.encoder.num_patches, config.num_classes)

    def forward(self, x):
        # x: [B, 1, T, D, H, W]
        x = x.unsqueeze(1)
        x = self.time_embedder(x)
        x = self.encoder.patch_embed(x)
        N, L, D_embed = x.shape
        d, h, w = self.config.spatial_dim[0] // self.config.patch_size[0], self.config.spatial_dim[1] // \
                  self.config.patch_size[1], self.config.spatial_dim[2] // self.config.patch_size[2]
        x = x.reshape(N, d, h, w, D_embed) + self.pos_embed_space
        x = x.reshape(N, L, D_embed)
        x = self.pre_dropout(x)
        for blk in self.encoder.blocks: x = blk(x)
        x = self.encoder.norm(x)
        logits = self.head(x.flatten(start_dim=1))
        return logits


# ==================================================
# 3. 数据集 (插值逻辑)
# ==================================================
class IBCClassificationDataset(Dataset):
    def __init__(self, file_paths, label_map, target_window_size=16):
        self.file_paths = file_paths
        self.label_map = label_map
        self.target_window_size = target_window_size
        self.samples = self._create_samples()

    def _create_samples(self):
        samples = []
        for fpath in self.file_paths:
            try:
                basename = os.path.basename(fpath)
                label = None
                for task_name, lbl_idx in self.label_map.items():
                    if task_name in basename:
                        label = lbl_idx
                        break
                if label is not None:
                    # 不再需要 scan_id 进行聚合，但为了方便debug还是可以保留
                    samples.append((fpath, label))
            except Exception as e:
                print(f"Skipping {fpath}: {e}")
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fpath, label = self.samples[idx]
        data = np.load(fpath)  # [T=8, D, H, W]
        # 物理时间对齐
        # 既然模型是在 11.5s 的窗口上训练的 (16 * 0.72s)
        # 而 IBC 的 TR=2.0s，取前 6 帧 (12s) 最接近这个时间长度

        # 1. 先只取前 6 帧 (对应约 12秒)
        # 注意：如果有的数据本身不足6帧，需要做padding，但之前代码保证了至少8帧
        data_crop = data[:6, ...]  # [T=6, D, H, W]

        data_torch = torch.from_numpy(data_crop).unsqueeze(0)  # [1, 6, D, H, W]

        # 2. 插值
        D, H, W = data_torch.shape[2:]
        T_in = data_torch.shape[1]  # 这里是 6

        # [1, 6, D, H, W] -> [1, D*H*W, 6]
        temp = data_torch.reshape(1, T_in, -1).permute(0, 2, 1)

        # 将 6 帧插值到 16 帧
        # 这样时间上的“拉伸感”会被消除，保留了波形的原始速度特征
        temp_resized = F.interpolate(temp, size=self.target_window_size, mode='linear', align_corners=False)

        # [1, D*H*W, 16] -> [1, 16, D, H, W]
        data_resized = temp_resized.permute(0, 2, 1).reshape(1, self.target_window_size, D, H, W)

        # 归一化
        mean = data_resized.mean()
        std = data_resized.std()
        data_final = (data_resized - mean) / (std + 1e-6)

        return data_final.squeeze(0), torch.tensor(label, dtype=torch.long)


# ==================================================
# 4. 评估函数 (修改版：无投票，窗口级)
# ==================================================
@torch.no_grad()
def evaluate(model, loader, device, class_names):
    model.eval()

    all_preds = []
    all_labels = []
    all_probs = []

    print("Begin testing...")
    # 注意：这里不需要 scan_id 了
    for batch_x, batch_y in tqdm(loader, desc="Inference"):
        batch_x = batch_x.to(device)  # [B, 1, 16, 80, 96, 80]
        batch_y = batch_y.to(device)

        with autocast():
            logits = model(batch_x)  # [B, 7]

        # 直接获取每个窗口的预测结果
        probs = softmax(logits.cpu().numpy(), axis=1)
        preds = np.argmax(probs, axis=1)
        labels = batch_y.cpu().numpy()

        all_preds.extend(preds)
        all_labels.extend(labels)
        all_probs.extend(probs)

    # 转换为numpy数组
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # 计算指标 (基于每个窗口独立计算)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    cm = confusion_matrix(all_labels, all_preds, labels=range(len(class_names)))

    precision, recall, f1_cls, _ = precision_recall_fscore_support(
        all_labels, all_preds, labels=range(len(class_names)), zero_division=0
    )

    results = {
        'accuracy': acc,
        'f1': f1,
        'cm': cm,
        'precision_cls': precision,
        'recall_cls': recall,
        'f1_cls': f1_cls,
        'labels': all_labels,
        'probs': all_probs
    }
    return results


# ==================================================
# 5. 主程序
# ==================================================
def main():
    cfg = TestConfig()
    device = torch.device(cfg.device)
    print(f"[INFO] Using device: {device}")

    class_names = ['EMOTION', 'GAMBLING', 'LANGUAGE', 'MOTOR', 'RELATIONAL', 'SOCIAL', 'WM']
    label_map = {name: i for i, name in enumerate(class_names)}

    all_files = glob.glob(os.path.join(cfg.data_root, '*.npy'))
    if len(all_files) == 0:
        print(f"[ERROR] No .npy files found in {cfg.data_root}")
        return

    print(f"[INFO] Found {len(all_files)} IBC window samples.")

    test_dataset = IBCClassificationDataset(all_files, label_map, target_window_size=cfg.window_size)
    test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)

    print(f"[INFO] Loading model from: {cfg.model_path}")
    model = EncoderClassifier(cfg).to(device)

    if os.path.exists(cfg.model_path):
        checkpoint = torch.load(cfg.model_path, map_location=device)
        state_dict = checkpoint if 'model_state_dict' not in checkpoint else checkpoint['model_state_dict']
        if 'encoder.patch_embed.proj.weight' not in state_dict and 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']

        try:
            model.load_state_dict(state_dict, strict=True)
        except RuntimeError as e:
            print(f"[WARN] Strict loading failed, trying non-strict: {e}")
            model.load_state_dict(state_dict, strict=False)

        print("[INFO] Model weights loaded successfully.")
    else:
        print(f"[ERROR] Model file not found!")
        return

    results = evaluate(model, test_loader, device, class_names)

    print("\n" + "=" * 40)
    print(f"IBC Dataset Window-Level Test Results")
    print("=" * 40)
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"Weighted F1: {results['f1']:.4f}")
    print("-" * 20)
    print("Per-Class Performance:")
    for i, name in enumerate(class_names):
        print(
            f"{name:<12} | Prec: {results['precision_cls'][i]:.4f} | Rec: {results['recall_cls'][i]:.4f} | F1: {results['f1_cls'][i]:.4f}")
    print("=" * 40)

    logger = get_logger(os.path.join(cfg.log_dir, f"test_frozen_{cfg.num_subjects}.log"))
    logger.info(f"Model: {cfg.model_path}")
    logger.info(f"Window-Level Acc: {results['accuracy']:.4f}")

    save_confusion_matrix(results['cm'], class_names,
                          os.path.join(cfg.log_dir, f"ibc_lora_confusion_matrix_{cfg.num_subjects}.png"),
                          title="IBC Window-Level Confusion Matrix")


if __name__ == "__main__":
    main()