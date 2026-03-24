import os
import time
import glob
import logging
import itertools
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import nibabel as nib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import argparse
import matplotlib as mpl
from sklearn.metrics import (accuracy_score, f1_score, confusion_matrix,
                             recall_score, precision_score, precision_recall_fscore_support,
                             roc_curve, auc)
from sklearn.preprocessing import label_binarize
from scipy.special import softmax
from timm.models.layers import DropPath
from einops import rearrange
from collections import Counter, defaultdict
from torch.cuda.amp import autocast, GradScaler

# 绘图默认设置
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Helvetica', 'Bitstream Vera Sans']
plt.rcParams['font.size'] = 14
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8
plt.rcParams['legend.fontsize'] = 12

try:
    from flash_attn.modules.mha import MHA
except ImportError:
    MHA = None
    print("Warning: flash_attn.modules.mha not found. Will not use MHA module.")


# ==================================================
# 1. 配置
# ==================================================
class ClassifyConfig:
    data_root = "/data3/ysn/HCP_Seven_Tasks/"
    spatial_dim = (80, 96, 80)
    window_size = 16

    # --- 被试划分设置 ---
    num_train_subjects = 50
    num_val_subjects = 100

    # --- 模型超参数 ---
    pretrained_encoder_path = "/data3/yyy/MAE_base/model/mae_fmri_epoch_47.pth"
    embed_dim_intermediate = 16  # TimeEmbedModule的输出通道数
    patch_size = (8, 8, 8)  # 空间patch大小
    embed_dim = 1024  # Encoder的token维度
    depth = 4  # Encoder的Transformer层数
    num_heads = 8  # Encoder的多头注意力头数
    mlp_ratio = 4.0
    num_classes = 7
    dropout = 0.1

    # --- 训练超参数 ---
    batch_size = 16
    learning_rate = 1e-4
    weight_decay = 1e-5
    epochs = 100
    num_workers = 8
    device = 'cuda:5'
    log_dir = "/home/ysn/PycharmProjects/MAE/HCP_7Classify/result/"


os.makedirs(ClassifyConfig.log_dir, exist_ok=True)


# ==================================================
# 2. 辅助函数
# ==================================================
def get_logger(filename):
    logger = logging.getLogger(filename)
    logger.setLevel(logging.INFO)
    if logger.hasHandlers():
        logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s %(message)s")
    fh = logging.FileHandler(filename, mode='a')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
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
    if normalize:
        cbar.set_ticks(np.arange(0, 101, 20))
        cbar.set_ticklabels([f'{x}%' for x in np.arange(0, 101, 20)])
    plt.title(title, fontsize=16)
    plt.xlabel('Predicted Label', fontsize=14)
    plt.ylabel('True Label', fontsize=14)
    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_yticklabels(classes)
    fmt = '.1f' if normalize else 'd'
    thresh = cm_norm.max() / 2.
    for i, j in itertools.product(range(cm_norm.shape[0]), range(cm_norm.shape[1])):
        val_text = f"{cm_norm[i, j]:{fmt}}" + ('%' if normalize else '')
        plt.text(j, i, val_text, ha="center", va="center", color="white" if cm_norm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', format='png')
    plt.close()


def plot_roc_curves(y_true, y_score, class_names, save_path=None):
    mpl.rcParams['font.size'] = 14
    y_score_proba = softmax(y_score, axis=1)
    n_classes = len(class_names)
    y_true_bin = label_binarize(y_true, classes=range(n_classes))
    plt.figure(figsize=(12, 10))
    fpr, tpr, roc_auc = dict(), dict(), dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_score_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    colors = plt.cm.get_cmap('tab20').colors
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2, label=f'{class_names[i]} (AUC = {roc_auc[i]:.3f})')
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true_bin.ravel(), y_score_proba.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    plt.plot(fpr["micro"], tpr["micro"],
             label=f'Micro-average AUC = {roc_auc["micro"]:.3f}',
             color='deeppink', linestyle=':', linewidth=4)
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0]);
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR)');
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def calculate_class_metrics(y_true, y_pred, num_classes):
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=None, labels=range(num_classes),
                                                               zero_division=0)
    return precision, recall, f1


# ==================================================
# 3. 模型
# ==================================================

class Mlp(nn.Module):
    """ Transformer Block 内部的 MLP """

    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.1):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class FlashAttention(nn.Module):
    """ 利用 flash-attn 库提供的高级模块 MHA 实现高效自注意力计算。 """

    def __init__(self, dim, num_heads=8, dropout=0.1):
        super().__init__()
        if MHA is None:
            raise ImportError("flash_attn.modules.mha is required for FlashAttention. Please install flash-attn.")
        self.attn = MHA(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            cross_attn=False,
            causal=False,
        )

    def forward(self, x):
        out = self.attn(x)
        return out


class Block3D(nn.Module):
    """ 基于 3D patch token 的 Transformer Block，使用 flash attention """

    def __init__(self, dim, num_heads, mlp_ratio=4.0, drop=0.1, attn_drop=0.1, drop_path=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = FlashAttention(dim, num_heads, dropout=attn_drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        hidden_features = int(dim * mlp_ratio)
        self.mlp = Mlp(dim, hidden_features, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchEmbed3D(nn.Module):
    """ 3D Patch Embedding (来自代码1) """

    def __init__(self, patch_size, in_chans, embed_dim, spatial_dim):
        super().__init__()
        self.patch_size = patch_size
        pd, ph, pw = patch_size
        D, H, W = spatial_dim
        self.num_patches_d = D // pd
        self.num_patches_h = H // ph
        self.num_patches_w = W // pw
        self.num_patches = self.num_patches_d * self.num_patches_h * self.num_patches_w
        self.embed_dim = embed_dim
        self.in_chans = in_chans
        self.patch_volume = pd * ph * pw * in_chans
        self.proj = nn.Linear(self.patch_volume, embed_dim)

    def forward(self, x):
        N, C, D, H, W = x.shape
        pd, ph, pw = self.patch_size
        assert D % pd == 0 and H % ph == 0 and W % pw == 0, \
            f"输入数据 {x.shape} 无法整除 patch_size {self.patch_size}"
        x = x.reshape(N, C, D // pd, pd, H // ph, ph, W // pw, pw)
        x = x.permute(0, 2, 4, 6, 1, 3, 5, 7)
        x = x.reshape(N, -1, self.patch_volume)
        x = self.proj(x)
        return x


class TimeEmbedModule(nn.Module):
    """ 时间嵌入模块 (来自代码1) """

    def __init__(self, time_steps, embed_dim_intermediate=16):
        super().__init__()
        self.embed_dim = embed_dim_intermediate
        self.time_embedder = nn.Sequential(
            nn.Conv3d(time_steps, embed_dim_intermediate, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm3d(embed_dim_intermediate),
            nn.ReLU()
        )

    def forward(self, x):
        N, C, T, D, H, W = x.shape
        x = x.reshape(N * C, T, D, H, W)
        x = self.time_embedder(x)
        x = x.reshape(N, self.embed_dim, D, H, W)
        return x


class VisionTransformer3D(nn.Module):
    """ "Transformer" 主干 """

    def __init__(self, patch_size, in_chans, embed_dim, depth, num_heads, mlp_ratio, spatial_dim, dropout):
        super().__init__()
        self.patch_embed = PatchEmbed3D(patch_size, in_chans, embed_dim, spatial_dim)
        self.num_patches_d = self.patch_embed.num_patches_d
        self.num_patches_h = self.patch_embed.num_patches_h
        self.num_patches_w = self.patch_embed.num_patches_w
        self.num_patches = self.patch_embed.num_patches
        self.blocks = nn.ModuleList([
            Block3D(embed_dim, num_heads, mlp_ratio, drop=dropout, attn_drop=dropout, drop_path=0.1)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.patch_embed(x)
        x = self.dropout(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x


# --- 新的分类器模型 ---
class EncoderClassifier(nn.Module):
    def __init__(self, config: ClassifyConfig):
        super().__init__()
        self.config = config

        # 1. 时间嵌入模块
        self.time_embedder = TimeEmbedModule(
            time_steps=config.window_size,
            embed_dim_intermediate=config.embed_dim_intermediate
        )

        # 2. Transformer 主干
        self.encoder = VisionTransformer3D(
            patch_size=config.patch_size,
            in_chans=config.embed_dim_intermediate,
            embed_dim=config.embed_dim,
            depth=config.depth,
            num_heads=config.num_heads,
            mlp_ratio=config.mlp_ratio,
            spatial_dim=config.spatial_dim,
            dropout=config.dropout
        )

        # 3. 空间位置编码 (可学习)
        d = self.encoder.num_patches_d
        h = self.encoder.num_patches_h
        w = self.encoder.num_patches_w
        self.pos_embed_space = nn.Parameter(torch.zeros(d, h, w, config.embed_dim))
        nn.init.trunc_normal_(self.pos_embed_space, std=0.05)

        self.pre_dropout = nn.Dropout(config.dropout)

        # 4. 分类头
        num_patches = self.encoder.num_patches
        self.head = nn.Linear(config.embed_dim * num_patches, config.num_classes)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        # 输入 x 形状: [B, T, D, H, W]
        # 添加一个通道维度以匹配 TimeEmbedModule 的输入
        x = x.unsqueeze(1)  # [B, 1, T, D, H, W]

        # 1. 时间嵌入
        x = self.time_embedder(x)  # [B, embed_dim_intermediate, D, H, W]

        # 2. Patch 嵌入
        x = self.encoder.patch_embed(x)  # [B, L, embed_dim]

        # 3. 添加空间位置编码
        N, L, D_embed = x.shape
        d, h, w = self.encoder.num_patches_d, self.encoder.num_patches_h, self.encoder.num_patches_w
        x = x.reshape(N, d, h, w, D_embed)
        x = x + self.pos_embed_space
        x = x.reshape(N, L, D_embed)

        # 4. 通过 Transformer Blocks
        x = self.pre_dropout(x)
        for blk in self.encoder.blocks:
            x = blk(x)
        x = self.encoder.norm(x)

        # 5. 分类
        x = x.flatten(start_dim=1)  # 展平所有 tokens
        logits = self.head(x)

        return logits


# ==================================================
# 4. 数据集
# ==================================================
class BlockClassificationDataset(Dataset):
    def __init__(self, file_paths, label_map, window_size=16, spatial_dim=(80, 96, 80), mode='train'):
        super().__init__()
        self.file_paths = file_paths
        self.label_map = label_map
        self.window_size = window_size
        self.spatial_dim = spatial_dim
        self.mode = mode
        self.samples = self._create_samples()

    def _create_samples(self):
        samples = []
        for fpath in tqdm(self.file_paths, desc=f"Creating {self.mode} samples", leave=False):
            try:
                data = np.load(fpath, mmap_mode='r')
                if data.shape[1:] != self.spatial_dim: continue
                total_time = data.shape[0]
                if total_time < self.window_size: continue
                base_name = os.path.basename(fpath)
                parts = base_name.split('_')
                subj_id, task_name = parts[0], parts[1]
                label = self.label_map.get(task_name, -1)
                if label == -1: continue
                max_start = total_time - self.window_size
                if self.mode == 'train':
                    num_windows = 6
                    for _ in range(num_windows):
                        start_t = random.randint(0, max_start)
                        samples.append((fpath, start_t, subj_id, label))
                else:
                    start_t = 0
                    samples.append((fpath, start_t, subj_id, label))
            except Exception as e:
                print(f"Error loading file {fpath}: {e}")

        print(f"Finished creating samples for {self.mode} mode. Total samples: {len(samples)}")
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fpath, st, subj_id, label = self.samples[idx]
        data = np.load(fpath, mmap_mode='r')

        clip_orig = data[st: st + self.window_size]
        clip_orig_torch = torch.from_numpy(clip_orig).float()

        mean, std = clip_orig_torch.mean(), clip_orig_torch.std()
        clip_norm = (clip_orig_torch - mean) / (std + 1e-6)

        final_clip = clip_norm

        return final_clip, torch.tensor(label, dtype=torch.long), subj_id


# ==================================================
# 5. 训练与评估
# ==================================================
@torch.no_grad()
def evaluate(model, loader, criterion, device, class_names):
    model.eval()
    all_preds, all_labels, all_logits = [], [], []
    total_loss = 0.0
    for batch_x, batch_y, _ in tqdm(loader, desc="Evaluating", leave=False):
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        with autocast():
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
        total_loss += loss.item()
        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(batch_y.cpu().numpy())
        all_logits.extend(logits.cpu().numpy())
    avg_loss = total_loss / len(loader) if len(loader) > 0 else 0
    total_acc = accuracy_score(all_labels, all_preds)
    weighted_f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)

    return {'loss': avg_loss, 'accuracy': total_acc, 'f1': weighted_f1,
            'labels': all_labels, 'predictions': all_preds,
            'probabilities': np.array(all_logits)}


def train_one_epoch(model, train_loader, criterion, optimizer, scaler, device, epoch, logger):
    model.train()
    total_loss = 0.0
    all_preds, all_labels = [], []
    for batch_x, batch_y, _ in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}", leave=False):
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        optimizer.zero_grad(set_to_none=True)
        with autocast():
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(batch_y.cpu().numpy())
    avg_loss = total_loss / len(train_loader) if len(train_loader) > 0 else 0
    avg_acc = accuracy_score(all_labels, all_preds)
    logger.info(f"Epoch {epoch + 1} Training: Loss={avg_loss:.4f}, Accuracy={avg_acc:.4f}")

    return {'loss': avg_loss, 'accuracy': avg_acc, 'predictions': all_preds, 'labels': all_labels}


# ==================================================
# 6. 主函数
# ==================================================
def main():
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    device = torch.device(ClassifyConfig.device if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    all_files = glob.glob(os.path.join(ClassifyConfig.data_root, '*.npy'))
    tasks = sorted(list(set(os.path.basename(f).split('_')[1] for f in all_files)))
    label_map = {task: i for i, task in enumerate(tasks)}
    class_names = tasks
    if len(class_names) != ClassifyConfig.num_classes:
        ClassifyConfig.num_classes = len(class_names)
    print(f"Found {len(class_names)} tasks: {class_names}")
    print(f"Label map: {label_map}")

    subject_files = defaultdict(list)
    for f in all_files:
        try:
            subject_id = os.path.basename(f).split('_')[0]
            subject_files[subject_id].append(f)
        except IndexError:
            print(f"Warning: Could not parse subject ID from {f}")

    all_subjects = sorted(list(subject_files.keys()))
    random.shuffle(all_subjects)

    num_train_subjects = ClassifyConfig.num_train_subjects
    num_val_subjects = ClassifyConfig.num_val_subjects
    if len(all_subjects) < num_train_subjects + num_val_subjects:
        raise ValueError(
            f"Not enough subjects for splitting. Found {len(all_subjects)}, but need {num_train_subjects + num_val_subjects}.")

    train_subjects = all_subjects[:num_train_subjects]
    val_subjects = all_subjects[num_train_subjects: num_train_subjects + num_val_subjects]
    train_files = [f for subj in train_subjects for f in subject_files[subj]]
    val_files = [f for subj in val_subjects for f in subject_files[subj]]

    print(f"Total subjects: {len(all_subjects)}")
    print(f"Training subjects: {len(train_subjects)}, Validation subjects: {len(val_subjects)}")
    print(f"Training files: {len(train_files)}, Validation files: {len(val_files)}")

    date_str = time.strftime("%Y%m%d_%H%M%S")

    log_name = f"HCP_EncoderAsClassifier_{date_str}"
    main_log_dir = os.path.join(ClassifyConfig.log_dir, log_name)
    figures_dir = os.path.join(main_log_dir, "figures")
    models_dir = os.path.join(main_log_dir, "models")
    os.makedirs(figures_dir, exist_ok=True);
    os.makedirs(models_dir, exist_ok=True)

    # 保存验证集文件路径列表为 npy 文件
    val_files_save_path = os.path.join(main_log_dir, "validation_files.npy")
    np.save(val_files_save_path, np.array(val_files))
    print(f"[INFO] Validation file paths saved to {val_files_save_path}")

    logger = get_logger(os.path.join(main_log_dir, "training_log.log"))

    logger.info(f"==== Starting HCP 7-Task Classification: {log_name} ====")
    for key, value in ClassifyConfig.__dict__.items():
        if not key.startswith('__'): logger.info(f"{key}: {value}")
    logger.info(f"Class names: {class_names}")

    train_dataset = BlockClassificationDataset(train_files, label_map, ClassifyConfig.window_size,
                                               ClassifyConfig.spatial_dim, 'train')
    val_dataset = BlockClassificationDataset(val_files, label_map, ClassifyConfig.window_size,
                                             ClassifyConfig.spatial_dim, 'val')
    train_loader = DataLoader(train_dataset, batch_size=ClassifyConfig.batch_size, shuffle=True,
                              num_workers=ClassifyConfig.num_workers, drop_last=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=ClassifyConfig.batch_size, shuffle=False,
                            num_workers=ClassifyConfig.num_workers, pin_memory=True)

    # 实例化新的模型
    model = EncoderClassifier(ClassifyConfig).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=ClassifyConfig.learning_rate,
                                  weight_decay=ClassifyConfig.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=ClassifyConfig.epochs, eta_min=1e-7)
    scaler = GradScaler()

    best_val_acc = 0.0
    best_model_path = ""

    logger.info("\n" + "=" * 50 + "\nStarting Training\n" + "=" * 50)

    for epoch in range(ClassifyConfig.epochs):
        train_results = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device, epoch, logger)
        lr_scheduler.step()

        val_results = evaluate(model, val_loader, criterion, device, class_names)
        logger.info(
            f"Epoch {epoch + 1} Validation: Loss={val_results['loss']:.4f}, Accuracy={val_results['accuracy']:.4f}, F1-score={val_results['f1']:.4f}")

        train_cm = confusion_matrix(train_results['labels'], train_results['predictions'],
                                    labels=range(len(class_names)))
        save_confusion_matrix(train_cm, class_names,
                              os.path.join(figures_dir, f"train_cm_epoch_{epoch + 1}.png"),
                              title=f"Training CM (Epoch {epoch + 1})")

        val_cm = confusion_matrix(val_results['labels'], val_results['predictions'], labels=range(len(class_names)))
        save_confusion_matrix(val_cm, class_names,
                              os.path.join(figures_dir, f"val_cm_epoch_{epoch + 1}.png"),
                              title=f"Validation CM (Epoch {epoch + 1})")

        if val_results['accuracy'] > best_val_acc:
            best_val_acc = val_results['accuracy']
            if best_model_path and os.path.exists(best_model_path):
                try:
                    os.remove(best_model_path)
                except OSError as e:
                    print(f"Error removing old model: {e}")
            best_model_path = os.path.join(models_dir, f"best_model_epoch_{epoch + 1}_acc_{best_val_acc:.4f}.pth")
            torch.save(model.state_dict(), best_model_path)
            logger.info(f"Saved new best model at epoch {epoch + 1} with validation accuracy: {best_val_acc:.4f}")

    logger.info("\n" + "=" * 50 + "\nTraining finished. Performing final evaluation with the best model.\n" + "=" * 50)

    if not best_model_path:
        logger.error("No best model was saved. Cannot perform final evaluation.")
        return

    logger.info(f"Loading best model from {best_model_path} for final evaluation.")
    model.load_state_dict(torch.load(best_model_path))

    final_val_results = evaluate(model, val_loader, criterion, device, class_names)
    final_cm = confusion_matrix(final_val_results['labels'], final_val_results['predictions'],
                                labels=range(len(class_names)))
    save_confusion_matrix(final_cm, class_names,
                          os.path.join(figures_dir, f"final_val_confusion_matrix.png"),
                          title=f"Final Validation Confusion Matrix (Accuracy: {final_val_results['accuracy']:.3f})")
    plot_roc_curves(final_val_results['labels'], final_val_results['probabilities'], class_names,
                    os.path.join(figures_dir, f"final_val_roc_curve.png"))

    logger.info(f"--- Final Validation Performance ---")
    logger.info(f"Accuracy: {final_val_results['accuracy']:.4f}")
    logger.info(f"Weighted F1-score: {final_val_results['f1']:.4f}")
    final_val_prec, final_val_recall, final_val_f1 = calculate_class_metrics(final_val_results['labels'],
                                                                             final_val_results['predictions'],
                                                                             len(class_names))
    for i, class_name in enumerate(class_names):
        logger.info(
            f"Class '{class_name}' | Precision: {final_val_prec[i]:.4f}, Recall: {final_val_recall[i]:.4f}, F1-score: {final_val_f1[i]:.4f}")

    logger.info(f"\nTraining and evaluation complete! Results saved to {main_log_dir}")


if __name__ == "__main__":
    if MHA is None:
        print("`flash_attn.modules.mha` is not installed.")
    else:
        main()