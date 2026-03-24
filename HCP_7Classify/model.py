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
mpl.use('Agg')
from sklearn.metrics import (accuracy_score, f1_score, confusion_matrix,
                             recall_score, precision_score, precision_recall_fscore_support,
                             roc_curve, auc)
from sklearn.preprocessing import label_binarize
from scipy.special import softmax
from timm.models.layers import DropPath
from einops import rearrange
from collections import Counter

# 导入混合精度训练所需的模块
from torch.cuda.amp import autocast, GradScaler

# ===================== 绘图默认设置 =====================
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Helvetica', 'Bitstream Vera Sans']
plt.rcParams['font.size'] = 14
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8
plt.rcParams['legend.fontsize'] = 12


try:
    from flash_attn import flash_attn_func
except ImportError:
    flash_attn_func = None
    print("Warning: flash_attn not found. Using standard attention mechanism.")


##################################################
# 1. 配置
##################################################

class ClassifyConfig:

    # ------------------- 数据相关 -------------------
    data_root = "/data3/ysn/HCP_Seven_Tasks"  # fMRI数据根目录

    spatial_dim = (80, 96, 80)  # D, H, W
    in_chans = 1

    # ------------------- 滑动窗口 -------------------
    window_size = 16
    step_size = 2
    # ------------------- 训练划分 -------------------
    num_train_subj = 50
    num_val_subj = 100

    # ------------------- 模型相关 -------------------

    pretrained_encoder_path = "/data3/yyy/MAE_base/model/mae_fmri_epoch_47.pth"
    # Encoder 的超参
    patch_size = (8, 8, 8)
    embed_dim = 1024
    depth = 4
    num_heads = 8
    mlp_ratio = 4.0

    # ------------------- 训练超参 -------------------
    batch_size = 16
    learning_rate = 1e-4
    weight_decay = 1e-5
    epochs = 50
    warmup_epochs = 0
    num_workers = 6 

    # ------------------- 单GPU配置 -------------------
    device = 'cuda:4'

    # ------------------- 其他 -------------------
    log_dir = "/home/ysn/PycharmProjects/MAE/HCP_7Classify/result/"


# 创建必要的目录
os.makedirs(ClassifyConfig.log_dir, exist_ok=True)


##################################################
# 2. 帮助函数 (日志、绘图等)
##################################################

def get_logger(filename):
    """创建日志记录器"""
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(filename)s %(funcName)s %(message)s")
    fh = logging.FileHandler(filename)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger


def save_confusion_matrix(cm, classes, save_path, normalize=True, title='Confusion Matrix'):
    """
    绘制混淆矩阵
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.figure(figsize=(8, 6))

    if normalize:
        row_sums = cm.sum(axis=1)
        row_sums[row_sums == 0] = 1 #避免除零错误
        cm = cm.astype('float') / row_sums[:, np.newaxis]
        cm = cm * 100  # 转换为百分比

    # 使用imshow代替matshow，这样可以将x轴标签放在底部
    ax = plt.subplot(111)
    cax = ax.imshow(cm, cmap='GnBu', vmin=0, vmax=100 if normalize else None)
    cbar = plt.colorbar(cax)
    if normalize:
        cbar.set_ticks([0, 20, 40, 60, 80, 100])
        cbar.set_ticklabels(['0%', '20%', '40%', '60%', '80%', '100%'])

    # 坐标轴标签
    plt.title(title, fontsize=16)
    plt.xlabel('Predicted Label', fontsize=14)
    plt.ylabel('True Label', fontsize=14)

    # 刻度标签 - 现在x轴标签会显示在底部
    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(classes, rotation=0)
    ax.set_yticklabels(classes)

    # 数值标签
    fmt = '.1f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        text_label = f"{cm[i, j]:.1f}%" if normalize else f"{cm[i, j]}"
        color = "white" if cm[i, j] > thresh else "black"
        if i == j:
            plt.text(j, i, text_label, ha="center", va="center", color="white")
        else:
            plt.text(j, i, text_label, ha="center", va="center", color="black")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', format='png')
    print(f"Saved confusion matrix to {save_path}")
    plt.close()


def plot_violin(data, save_path, class_names=None, ylabel=None, figsize=(10, 6), dpi=300):
    """
    绘制小提琴图
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # 将数据转换为DataFrame格式
    if class_names is None:
        class_names = [f'Class {i}' for i in range(data.shape[1])]

    df = pd.DataFrame(data, columns=class_names)

    # 创建小提琴图
    plt.figure(figsize=figsize, dpi=dpi)

    # 定义颜色方案
    colors = ['#dfdfdf', '#4a7298', '#f3c846', '#d93c3e', '#75bf71', '#9966CC', '#FF6347', '#00CED1']

    # 绘制小提琴图
    with plt.style.context({'font.family': 'sans-serif', 'font.sans-serif': ['Arial']}):
        sns.violinplot(data=df, palette=colors[:len(class_names)], inner='box', cut=0)

        # 设置标签
        if ylabel:
            plt.ylabel(ylabel, fontsize=16)

        # 去除网格和标题
        plt.grid(False)
        plt.title('', fontsize=16)

        plt.tight_layout()
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight', format='png')
        print(f"Saved violin plot to {save_path}")

    plt.close()


def plot_roc_curves(y_true, y_score, class_names, save_path=None):
    """
    绘制ROC曲线
    """
    mpl.rcParams['font.size'] = 14

    # 二值化标签
    y_true_bin = label_binarize(y_true, classes=range(len(class_names)))

    # 如果y_score是logits，转换为概率
    if len(y_score.shape) == 2 and y_score.shape[1] == len(class_names):
        y_score_proba = softmax(y_score, axis=1)
    else:
        y_score_proba = y_score

    # 计算每个类别的ROC曲线和AUC
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(len(class_names)):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_score_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # # 计算微平均ROC曲线和AUC
    # fpr["micro"], tpr["micro"], _ = roc_curve(y_true_bin.ravel(), y_score_proba.ravel())
    # roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # 绘制所有ROC曲线
    colors = ['#dfdfdf', '#4a7298', '#f3c846', '#d93c3e', '#75bf71', '#9966CC', '#FF6347']
    plt.figure(figsize=(10, 8))

    for i, color in zip(range(len(class_names)), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label=f'{class_names[i]} (AUC = {roc_auc[i]:.3f})')

    # # 绘制平均ROC曲线
    # plt.plot(fpr["micro"], tpr["micro"],
    #          label=f'Average (AUC = {roc_auc["micro"]:.3f})',
    #          color='deeppink', linestyle=':', linewidth=3)

    # 绘制对角线
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    # plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def calculate_class_metrics(y_true, y_pred, num_classes=7):
    """
    计算每个类别的准确率和F1分数
    """
    accuracy_per_class = np.zeros(num_classes)
    f1_per_class = np.zeros(num_classes)

    for i in range(num_classes):
        y_true_binary = (np.array(y_true) == i).astype(int)
        y_pred_binary = (np.array(y_pred) == i).astype(int)
        accuracy_per_class[i] = accuracy_score(y_true_binary, y_pred_binary)
        f1_per_class[i] = f1_score(y_true_binary, y_pred_binary)

    return accuracy_per_class, f1_per_class


def calculate_segment_metrics(segments_preds, segments_labels, num_classes=7):
    """
    计算每个片段的每个类别的准确率和F1分数
    """
    segment_ids = list(segments_preds.keys())
    n_segments = len(segment_ids)

    # 准备结果矩阵，形状为 [n_segments, n_classes+1]
    # 最后一列用于存储平均值
    segment_acc_matrix = np.zeros((n_segments, num_classes + 1))
    segment_f1_matrix = np.zeros((n_segments, num_classes + 1))

    for i, seg_id in enumerate(segment_ids):
        preds = np.array(segments_preds[seg_id])
        labels = np.array(segments_labels[seg_id])

        # 跳过没有预测或标签的片段
        if len(preds) == 0 or len(labels) == 0:
            continue

        # 计算整体准确率和F1分数
        segment_acc_matrix[i, -1] = accuracy_score(labels, preds)
        segment_f1_matrix[i, -1] = f1_score(labels, preds, average='weighted')

        # 计算每个类别的准确率和F1分数
        for c in range(num_classes):
            # 对于准确率，计算该类别被正确预测的比例
            class_indices = (labels == c)
            if np.sum(class_indices) > 0:
                segment_acc_matrix[i, c] = np.mean(preds[class_indices] == c)

            # 对于F1分数，使用二分类方式计算
            y_true_binary = (labels == c).astype(int)
            y_pred_binary = (preds == c).astype(int)
            if np.sum(y_true_binary) > 0 and np.sum(y_pred_binary) > 0:
                segment_f1_matrix[i, c] = f1_score(y_true_binary, y_pred_binary)

    return segment_acc_matrix, segment_f1_matrix


def calculate_subject_metrics(subjects_preds, subjects_labels, num_classes=7):
    """
    计算每个被试的每个类别的准确率和F1分数
    """
    subject_ids = list(subjects_preds.keys())
    n_subjects = len(subject_ids)

    # 准备结果矩阵，形状为 [n_subjects, n_classes+1]
    # 最后一列用于存储平均值
    subject_acc_matrix = np.zeros((n_subjects, num_classes + 1))
    subject_f1_matrix = np.zeros((n_subjects, num_classes + 1))

    for i, subj_id in enumerate(subject_ids):
        preds = np.array(subjects_preds[subj_id])
        labels = np.array(subjects_labels[subj_id])

        # 跳过没有预测或标签的被试
        if len(preds) == 0 or len(labels) == 0:
            continue

        # 计算整体准确率和F1分数
        subject_acc_matrix[i, -1] = accuracy_score(labels, preds)
        subject_f1_matrix[i, -1] = f1_score(labels, preds, average='weighted')

        # 计算每个类别的准确率和F1分数
        for c in range(num_classes):
            # 对于准确率，计算该类别被正确预测的比例
            class_indices = (labels == c)
            if np.sum(class_indices) > 0:
                subject_acc_matrix[i, c] = np.mean(preds[class_indices] == c)

            # 对于F1分数，使用二分类方式计算
            y_true_binary = (labels == c).astype(int)
            y_pred_binary = (preds == c).astype(int)
            if np.sum(y_true_binary) > 0 and np.sum(y_pred_binary) > 0:
                subject_f1_matrix[i, c] = f1_score(y_true_binary, y_pred_binary)

    return subject_acc_matrix, subject_f1_matrix


##################################################
# 3. 投票机制相关函数（block级分类不需要）
##################################################

def expand_predictions(window_predictions, start_positions, sequence_length=284):
    """
    将窗口预测扩展到完整序列中的正确位置
    """
    expanded_predictions = {t: [] for t in range(sequence_length)}

    for i, preds in enumerate(window_predictions):
        start_pos = start_positions[i]

        # 对于每个时间步的预测，放入相应的位置
        for t, pred in enumerate(preds):
            # 计算在原始序列中的位置
            orig_pos = start_pos + t

            # 确保位置在有效范围内
            if 0 <= orig_pos < sequence_length:
                expanded_predictions[orig_pos].append(pred.item())

    return expanded_predictions


def vote_predictions(expanded_predictions):
    """
    对每个时间步的多个预测进行投票
    """
    sequence_length = len(expanded_predictions)
    final_predictions = np.full(sequence_length, -1)  # 初始化为-1表示无效预测

    for t in range(sequence_length):
        preds = expanded_predictions[t]
        if preds:  # 如果有预测
            # 使用Counter进行投票
            counter = Counter(preds)
            # 获取出现次数最多的类别
            final_predictions[t] = counter.most_common(1)[0][0]

    return final_predictions


##################################################
# 4. 模型组件: 3D时间通道编码器 + 分类器
##################################################

class TimeChannelConverter(nn.Module):
    """
    转换 [N, C, T, D, H, W] <--> [N, C*T, D, H, W]
    """

    def __init__(self):
        super().__init__()

    def time_to_channel(self, x):
        # x: [B, C, T, D, H, W] -> [B, C*T, D, H, W]
        return rearrange(x, 'b c t d h w -> b (c t) d h w')

    def channel_to_time(self, x, T):
        # x: [B, C*T, D, H, W] -> [B, C, T, D, H, W]
        C = x.shape[1] // T
        return rearrange(x, 'b (c t) d h w -> b c t d h w', t=T)


class Attention3D(nn.Module):

    def __init__(self, dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [B, N, C]
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B, num_heads, N, head_dim]
        q = q * self.scale

        if flash_attn_func is not None:
            out = flash_attn_func(q, k, v, dropout_p=self.dropout.p, causal=False)
        else:
            # 标准attention
            attn = torch.matmul(q, k.transpose(-2, -1))
            attn = F.softmax(attn, dim=-1)
            attn = self.dropout(attn)
            out = torch.matmul(attn, v)

        out = out.transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        out = self.dropout(out)
        return out


class Mlp3D(nn.Module):
    """基本MLP模块"""

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


class Block3D(nn.Module):
    """3D patch tokens的Transformer块"""

    def __init__(self, dim, num_heads=8, mlp_ratio=4.0, drop=0.1, attn_drop=0.1, drop_path=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention3D(dim, num_heads, dropout=attn_drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        hidden_features = int(dim * mlp_ratio)
        self.mlp = Mlp3D(dim, hidden_features, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchEmbed3D(nn.Module):
    """
    3D Patch Embedding:
      input shape [B, in_chans, D, H, W]
      output shape [B, num_patches, embed_dim]
    """

    def __init__(self,
                 patch_size=(5, 6, 5),
                 in_chans=16,  # e.g. 1 * T
                 embed_dim=384,
                 spatial_dim=(80, 96, 80)):
        super().__init__()
        self.patch_size = patch_size
        pd, ph, pw = patch_size
        D, H, W = spatial_dim
        self.num_patches_d = D // pd
        self.num_patches_h = H // ph
        self.num_patches_w = W // pw
        self.num_patches = self.num_patches_d * self.num_patches_h * self.num_patches_w
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.patch_volume = pd * ph * pw * in_chans

        self.proj = nn.Linear(self.patch_volume, embed_dim)

    def forward(self, x):
        """
        x: [B, (C*T), D, H, W]
        """
        B, CT, D, H, W = x.shape
        pd, ph, pw = self.patch_size
        # 1) reshape -> patch
        x = x.reshape(
            B, CT,
            D // pd, pd,
            H // ph, ph,
            W // pw, pw
        )
        # 2) permute
        x = x.permute(0, 2, 4, 6, 1, 3, 5, 7)  # [B, d_, h_, w_, CT, pd, ph, pw]
        # 3) flatten
        x = x.reshape(B, -1, self.patch_volume)
        # 4) linear
        x = self.proj(x)  # [B, L, embed_dim]
        return x


class VisionTransformer3D(nn.Module):
    """单GPU 3D ViT"""

    def __init__(self,
                 patch_size=(5, 6, 5),
                 in_chans=16,
                 embed_dim=384,
                 depth=8,
                 num_heads=12,
                 mlp_ratio=4.0,
                 spatial_dim=(80, 96, 80),
                 dropout=0.1):
        super().__init__()

        self.patch_embed = PatchEmbed3D(
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            spatial_dim=spatial_dim
        )
        self.num_patches = self.patch_embed.num_patches

        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.blocks = nn.ModuleList([
            Block3D(dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    drop=dropout,
                    attn_drop=dropout,
                    drop_path=0.1)
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        # x -> [B, L, embed_dim]
        x = self.patch_embed(x)
        # 添加位置编码
        x = x + self.pos_embed
        x = self.drop(x)

        # transformer blocks
        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return x


class MAE_FMRI_Encoder_3DTimeChan(nn.Module):
    """
    3D时间通道编码器 - 直接返回[B, L, embed_dim]格式的特征
    """

    def __init__(self, time_steps=16, spatial_dim=(80, 96, 80), patch_size=(5, 6, 5),
                 in_chans=1, embed_dim=384, depth=8, num_heads=12, mlp_ratio=4.0,
                 dropout=0.1, pretrained_path=None):
        super().__init__()
        self.time_steps = time_steps
        self.spatial_dim = spatial_dim
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.time_channel_converter = TimeChannelConverter()

        self.vit3d = VisionTransformer3D(
            patch_size=patch_size,
            in_chans=in_chans * time_steps,  # 16
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            spatial_dim=spatial_dim,
            dropout=dropout
        )

        # 加载预训练模型
        if pretrained_path and os.path.exists(pretrained_path):
            ckpt = torch.load(pretrained_path, map_location="cpu")
            self.vit3d.load_state_dict(ckpt, strict=False)
            print(f"[INFO] Loaded 3D-time-channel MAE encoder from {pretrained_path}")

    def forward(self, x):
        # x shape: [B, 1, T=16, D=80, H=96, W=80]
        # => time_to_channel => [B, (1*T=16), D, H, W]
        x = self.time_channel_converter.time_to_channel(x)
        # => pass through 3D transformer => [B, L, embed_dim]
        x = self.vit3d(x)
        return x


class BlockLevelClassifier(nn.Module):
    def __init__(self, embed_dim=384, num_classes=7,num_patches = 4096):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_classes = num_classes

        # 添加全局平均池化层
        # self.pooling = nn.AdaptiveAvgPool1d(1)

        self.classifier = nn.Sequential(
            nn.BatchNorm1d(embed_dim*num_patches),
            nn.Dropout(0.5),
            nn.Linear(embed_dim*num_patches, num_classes)
        )

    def forward(self, x):
        # x: [B, L, embed_dim]

        # 转置为 [B, embed_dim, L] 进行1D池化
        # x = rearrange(x, 'b l e -> b e l')

        # 全局平均池化: [B, embed_dim, L] -> [B, embed_dim, 1]
        # x = self.pooling(x)

        x = x.flatten(1) #[B, embed_dim*num_patches]

        logits = self.classifier(x)

        return logits


class MAE_FMRI_Classifier(nn.Module):
    """完整的模型: 3D时间通道编码器 + 帧级分类器"""

    def __init__(self, time_steps=16, spatial_dim=(80, 96, 80), patch_size=(8, 8, 8),
                 in_chans=1, embed_dim=1024, depth=6, num_heads=8, mlp_ratio=4.0,
                 num_classes=7, pretrained_path=None):
        super().__init__()
        self.encoder = MAE_FMRI_Encoder_3DTimeChan(
            time_steps=time_steps,
            spatial_dim=spatial_dim,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=0.1,
            pretrained_path=pretrained_path
        )

        # 计算patch数量
        D, H, W = spatial_dim
        pd, ph, pw = patch_size
        num_patches = (D // pd) * (H // ph) * (W // pw)

        self.classifier = BlockLevelClassifier(
            embed_dim=embed_dim,
            num_classes=num_classes,
            num_patches = num_patches
        )

    def forward(self, x):
        # x: [B, 1, T=16, D, H, W]
        feat = self.encoder(x)  # => [B, L, embed_dim]
        logits = self.classifier(feat)  # => [B, num_classes]
        return logits


##################################################
# 5. 数据集 (训练时随机取窗，验证时用最前面8帧)
##################################################

class BlockClassificationDataset(Dataset):
    def __init__(self,
                 file_paths,
                 window_size=16,
                 spatial_dim=(80, 96, 80),
                 class_names = None,
                 mode = 'train'
                 ):
        super().__init__()
        self.file_paths = file_paths

        self.window_size = window_size
        self.spatial_dim = spatial_dim

        self.file_lengths = {}  # 存储数据时间长度
        valid_files = []

        for fpath in self.file_paths:
            try:
                # 仅打开文件检查数据
                with open(fpath, 'r') as f:
                    data = np.load(fpath, mmap_mode='r')
                    if data.shape[1:] != spatial_dim:
                        print(f"警告: 文件 {fpath} 空间形状 {data.shape[1:]} 与预期 {spatial_dim} 不符，跳过")
                        continue
                    valid_files.append(fpath)
                    self.file_lengths[fpath] = data.shape[0]  # 存储数据时间长度
            except Exception as e:
                print(f"加载文件 {fpath} 时出错: {e}，跳过")

        # 更新file_paths为有效文件列表
        self.file_paths = valid_files
        print(f"文件筛查完毕，有效文件数: {len(valid_files)}")

        # 构建所有样本列表 (file_idx, start_t, end_t, label)
        self.samples = []

        for fpath in self.file_paths:

            # 提取被试ID
            subj_id = os.path.basename(fpath).split('_')[0]
            label = None

            # 获取标签
            for j in range(len(class_names)):
                if os.path.basename(fpath).split('_')[1] == class_names[j]:
                    label = j
                    break

            total_time = self.file_lengths[fpath]
            max_start = total_time - self.window_size
            if max_start < 0:
                continue

            if mode == 'train':
                for k in range(6):
                    start_t = random.randint(0, max_start)
                    end_t = start_t + self.window_size
                    # 记录file_idx, start_t, end_t, subj_id和frame_labels
                    self.samples.append((fpath, start_t, end_t, subj_id, label))
            else:
                start_t = 0
                end_t = start_t + self.window_size
                # 记录file_idx, start_t, end_t, subj_id和label
                self.samples.append((fpath, start_t, end_t, subj_id, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fpath, st, ed, subj_id, label = self.samples[idx]

        with open(fpath, 'rb') as f:
            data = np.load(fpath, mmap_mode='r')
            clip = data[st:ed]  # (16, 80, 96, 80)

        # 转换为torch张量
        clip_torch = torch.from_numpy(clip).float()  # [16, 80, 96, 80]

        # 全局时空归一化 - 对整个窗口计算单一的均值和标准差
        clip_torch = (clip_torch - clip_torch.mean()) / (clip_torch.std() + 1e-6)

        # 转换标签为torch张量
        label = torch.tensor(label, dtype=torch.long)

        # 增加channel维度 => [1, 16, 80, 96, 80]
        clip_torch = clip_torch.unsqueeze(0)

        # 片段ID - 使用文件名+起始帧作为标识
        segment_id = f"{os.path.basename(fpath)}_{st}"

        return clip_torch, label, subj_id, segment_id

##################################################
# 6. 评估函数 - 使用混合精度
##################################################

@torch.no_grad()
def evaluate(model, loader, device, class_names=None):
    model.eval()
    criterion = nn.CrossEntropyLoss()


    # 存储被试级和片段级的预测和标签
    subject_preds = {}
    subject_labels = {}

    all_preds = []
    all_labels = []
    all_logits = []
    total_loss = 0.0

    for batch_idx, (batch_x, batch_y, subject_ids, segment_ids) in enumerate(loader):
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device) # 单标签[B]

        with autocast():
            logits = model(batch_x)  # [B, num_classes]
            B, C = logits.shape
            loss = criterion(logits, batch_y)

        total_loss += loss.item()

        # 获取预测类别
        preds = torch.argmax(logits, dim=1)  # [B]
        batch_logits = logits.detach().cpu().numpy()
        batch_preds = preds.cpu().numpy()
        batch_labels = batch_y.cpu().numpy()

        # 收集数据
        all_preds.extend(batch_preds)
        all_labels.extend(batch_labels)
        all_logits.extend(batch_logits)

        for b in range(B):
            subj_id = subject_ids[b]

            if subj_id not in subject_preds:
                subject_preds[subj_id] = []
                subject_labels[subj_id] = []

            subject_preds[subj_id].append(batch_preds[b])
            subject_labels[subj_id].append(batch_labels[b])

    avg_loss = total_loss / len(loader)
    cm = confusion_matrix(all_labels, all_preds, labels=range(len(class_names)))
    total_acc = accuracy_score(all_labels, all_preds)
    weighted_f1 = f1_score(all_labels, all_preds, average='weighted')

    # 计算每个类别的准确率和F1
    num_classes = C
    class_acc, class_f1 = calculate_class_metrics(all_labels, all_preds, num_classes)

    # 计算被试级别的准确率和F1矩阵
    subject_acc_matrix, subject_f1_matrix = calculate_subject_metrics(
        subject_preds, subject_labels, num_classes)

    # ROC计算
    y_score_proba = softmax(np.array(all_logits), axis=1)

    results = {
        'loss': avg_loss,
        'accuracy': total_acc,
        'f1': weighted_f1,
        'confusion_matrix': cm,
        'class_acc': class_acc,
        'class_f1': class_f1,
        'subject_acc_matrix': subject_acc_matrix,
        'subject_f1_matrix': subject_f1_matrix,
        'labels': all_labels,
        'predictions': all_preds,
        'probabilities': y_score_proba,
    }
    return results


##################################################
# 7. 训练函数 - 使用混合精度
##################################################

def train_one_epoch(model, train_loader, optimizer, scaler, device, epoch, logger):
    """
    训练一个epoch - 使用混合精度训练
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss()

    # 存储被试级和片段级的预测和标签
    subject_preds = {}
    subject_labels = {}

    all_preds = []
    all_labels = []

    for batch_x, batch_y, subject_ids, segment_ids in tqdm(train_loader, desc=f"Train Epoch {epoch + 1}"):
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device) # 单标签[B]

        optimizer.zero_grad(set_to_none=True)  # 更高效的梯度清零

        # 使用混合精度训练
        with autocast():
            # 前向传播
            logits = model(batch_x)  # [B, num_classes]

            # 计算损失
            B, C = logits.shape
            loss = criterion(logits, batch_y)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

        # 获取预测类别
        preds = torch.argmax(logits, dim=1)  # [B]

        # 统计准确率
        correct += (preds == batch_y).sum().item()
        total += batch_y.numel()

        # 收集所有预测和标签
        preds_np = preds.cpu().numpy()
        labels_np = batch_y.cpu().numpy()

        for b in range(B):
            subj_id = subject_ids[b]

            if subj_id not in subject_preds:
                subject_preds[subj_id] = []
                subject_labels[subj_id] = []

            subject_preds[subj_id].append(preds_np[b])
            subject_labels[subj_id].append(labels_np[b])

        all_preds.extend(preds_np)
        all_labels.extend(labels_np)


    # 计算平均损失和准确率
    avg_loss = total_loss / len(train_loader)
    avg_acc = correct / total

    # 计算每个类别的准确率和F1
    num_classes = C
    class_acc, class_f1 = calculate_class_metrics(all_labels, all_preds, num_classes)

    # 计算被试级别的准确率和F1矩阵
    subject_acc_matrix, subject_f1_matrix = calculate_subject_metrics(
        subject_preds, subject_labels, num_classes)

    # 记录训练信息
    logger.info(f"Epoch {epoch + 1} Train: loss={avg_loss:.4f}")
    # logger.info("Per-class train acc: [%s]", ", ".join([f"{acc:.4f}" for acc in class_acc]))

    results = {
        'loss': avg_loss,
        'accuracy': avg_acc,
        'class_acc': class_acc,
        'class_f1': class_f1,
        'subject_acc_matrix': subject_acc_matrix,
        'subject_f1_matrix': subject_f1_matrix,
        'all_preds': all_preds,
        'all_labels': all_labels
    }

    return results


##################################################
# 8. 主函数
##################################################

def main():
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    torch.cuda.manual_seed_all(42)

    # 检查GPU
    if not torch.cuda.is_available():
        raise RuntimeError("No GPU available.")
    device = torch.device(ClassifyConfig.device)
    print(f"[INFO] Using device: {device}")

    # 类别名称
    class_names = ['EMOTION', 'GAMBLING', 'LANGUAGE', 'MOTOR', 'RELATIONAL', 'SOCIAL', 'WM']
    num_classes = len(class_names)

    # 获取所有文件
    all_files = [os.path.join(ClassifyConfig.data_root, f)
                 for f in os.listdir(ClassifyConfig.data_root)
                 if f.endswith('.npy')]

    # 按被试ID进行分组
    subject_files = {}
    for file_path in all_files:
        # 从文件名中提取被试ID
        subj_id = os.path.basename(file_path).split('_')[0]
        if subj_id not in subject_files:
            subject_files[subj_id] = []
        subject_files[subj_id].append(file_path)

    # 随机打乱被试ID
    subject_ids = list(subject_files.keys())
    np.random.shuffle(subject_ids)

    # 按被试ID划分训练/验证集
    train_subject_ids = subject_ids[:ClassifyConfig.num_train_subj]
    val_subject_ids = subject_ids[ClassifyConfig.num_train_subj:
                                  ClassifyConfig.num_train_subj + ClassifyConfig.num_val_subj]

    # 获取训练和验证文件
    train_files = []
    for subj_id in train_subject_ids:
        train_files.extend(subject_files[subj_id])

    val_files = []
    for subj_id in val_subject_ids:
        val_files.extend(subject_files[subj_id])

    print(f"[INFO] 训练被试数: {len(train_subject_ids)}, 训练文件数: {len(train_files)}")
    print(f"[INFO] 验证被试数: {len(val_subject_ids)}, 验证文件数: {len(val_files)}")

    # 创建数据集
    train_dataset = BlockClassificationDataset(
        file_paths=train_files,
        window_size=ClassifyConfig.window_size,
        spatial_dim=ClassifyConfig.spatial_dim,
        class_names=class_names,
        mode = 'train'
    )

    val_dataset = BlockClassificationDataset(
        file_paths=val_files,
        window_size=ClassifyConfig.window_size,
        spatial_dim=ClassifyConfig.spatial_dim,
        class_names=class_names,
        mode = 'val'
    )

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=ClassifyConfig.batch_size,
        shuffle=True,
        num_workers=ClassifyConfig.num_workers,
        drop_last=False,
        pin_memory=True  # 加快数据传输到GPU
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=ClassifyConfig.batch_size,
        shuffle=False,
        num_workers=ClassifyConfig.num_workers,
        drop_last=False,
        pin_memory=True  # 加快数据传输到GPU
    )

    # 创建模型
    model = MAE_FMRI_Classifier(
        time_steps=ClassifyConfig.window_size,
        spatial_dim=ClassifyConfig.spatial_dim,
        patch_size=ClassifyConfig.patch_size,
        in_chans=ClassifyConfig.in_chans,
        embed_dim=ClassifyConfig.embed_dim,
        depth=ClassifyConfig.depth,
        num_heads=ClassifyConfig.num_heads,
        mlp_ratio=ClassifyConfig.mlp_ratio,
        num_classes=num_classes,
        pretrained_path=ClassifyConfig.pretrained_encoder_path
    )

    best_val_acc = 0
    model = model.to(device)

    # 优化器和学习率调度器

    encoder_lr = ClassifyConfig.learning_rate # 编码器使用更小学习率
    classifier_lr = ClassifyConfig.learning_rate   # 分类头使用更大学习率
    param_groups = [
        {'params': model.encoder.parameters(), 'lr': encoder_lr},
        {'params': model.classifier.parameters(), 'lr': classifier_lr},
    ]

    optimizer = torch.optim.AdamW(
        param_groups,
        weight_decay=ClassifyConfig.weight_decay
    )

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=ClassifyConfig.epochs,
        eta_min=1e-6
    )

    # 创建混合精度训练所需的梯度缩放器
    scaler = GradScaler()

    # 创建日志目录
    date_str = time.strftime("%m%d_%H%M", time.localtime())
    log_name = f"HCP_7Class_with_MAE_{ClassifyConfig.num_train_subj}_fine-tuning_{date_str}"
    log_dir = os.path.join(ClassifyConfig.log_dir, log_name)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(os.path.join(log_dir, "figures"), exist_ok=True)
    os.makedirs(os.path.join(log_dir, "models"), exist_ok=True)
    os.makedirs(os.path.join(log_dir, "plots"), exist_ok=True)

    # 保存验证集文件路径列表为 npy 文件
    val_files_save_path = os.path.join(log_dir, "validation_files.npy")
    np.save(val_files_save_path, np.array(val_files))
    print(f"[INFO] Validation file paths saved to {val_files_save_path}")

    # 创建日志记录器
    logger = get_logger(os.path.join(log_dir, "training.log"))
    logger.info(f"==== Block-design Classification: {log_name} ====")

    # 记录训练配置
    for key, value in ClassifyConfig.__dict__.items():
        if not key.startswith('__'): logger.info(f"{key}: {value}")

    # 训练循环
    for epoch in range(ClassifyConfig.epochs):

        # 学习率预热
        if epoch < ClassifyConfig.warmup_epochs:
            warmup_factor = (epoch + 1) / ClassifyConfig.warmup_epochs
            warmup_lr = ClassifyConfig.learning_rate * warmup_factor
            for param_group in optimizer.param_groups:
                param_group['lr'] = warmup_lr
        else:
            lr_scheduler.step()

        # 训练一个epoch，传入scaler用于混合精度训练
        train_results = train_one_epoch(model, train_loader, optimizer, scaler, device, epoch, logger)

        # 绘制训练结果的可视化图表
        # 1. 混淆矩阵
        cm = confusion_matrix(train_results['all_labels'], train_results['all_preds'], labels=range(num_classes))

        save_confusion_matrix(
            cm, class_names,
            os.path.join(log_dir, "figures", f"train_epoch_{epoch + 1}.png"),
            normalize=True, title=f"Train Confusion (Epoch {epoch + 1})"
        )
        
        # 2. 被试级别的小提琴图
        '''
        plot_violin(
            train_results['subject_acc_matrix'],
            os.path.join(log_dir, "plots", f"train_subject_acc_epoch_{epoch + 1}.png"),
            class_names=class_names + ['Mean'],
            ylabel='Accuracy per Subject'
        )

        plot_violin(
            train_results['subject_f1_matrix'],
            os.path.join(log_dir, "plots", f"train_subject_f1_epoch_{epoch + 1}.png"),
            class_names=class_names + ['Mean'],
            ylabel='F1-Score per Subject'
        )
        '''


        # 验证
        logger.info(f"--- Evaluating epoch {epoch + 1} ---")
        val_results = evaluate(model, val_loader, device, class_names)

        # 记录验证结果
        logger.info(f"Epoch {epoch + 1} Val: loss={val_results['loss']:.4f}, acc={val_results['accuracy']:.4f}, f1={val_results['f1']:.4f}")
        # logger.info("Per-class val acc: [%s]",
        #         ", ".join([f"{acc:.4f}" for acc in val_results['class_acc']]))

        # 绘制验证结果的可视化图表

        # 1. 混淆矩阵

        save_confusion_matrix(
            val_results['confusion_matrix'],
            class_names,
            os.path.join(log_dir, "figures", f"val_epoch_{epoch + 1}.png"),
            normalize=True,
            title=f"Val Confusion (Epoch {epoch + 1})"
        )
        '''
        # 2. ROC曲线
        plot_roc_curves(
            val_results['labels'],
            val_results['probabilities'],
            class_names,
            os.path.join(log_dir, "plots", f"val_roc_epoch_{epoch + 1}.png")
        )
        
        # 3. 被试级别的小提琴图
        plot_violin(
            val_results['subject_acc_matrix'],
            os.path.join(log_dir, "plots", f"val_subject_acc_epoch_{epoch + 1}.png"),
            class_names=class_names + ['Mean'],
            ylabel='Accuracy per Subject'
        )

        plot_violin(
            val_results['subject_f1_matrix'],
            os.path.join(log_dir, "plots", f"val_subject_f1_epoch_{epoch + 1}.png"),
            class_names=class_names + ['Mean'],
            ylabel='F1-Score per Subject'
        )
        '''

        # 保存最佳模型
        if val_results['accuracy'] > best_val_acc:
            best_val_acc = val_results['accuracy']
            torch.save(
                model.state_dict(),
                os.path.join(log_dir, "models", f"best_model_epoch_{epoch + 1}.pth")
            )
            logger.info(f"[INFO] Saved best model at epoch {epoch + 1} with val_acc={val_results['accuracy']:.4f}")

        # 每5个epoch保存一次检查点
        if (epoch + 1) % 5 == 0:
            torch.save(
                {
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': val_results['loss'],
                },
                os.path.join(log_dir, "models", f"checkpoint_epoch_{epoch + 1}.pth")
            )

    logger.info(f"Training completed! Best val acc={best_val_acc:.4f}")


if __name__ == "__main__":
    main()