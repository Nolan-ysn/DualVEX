import os
import time
import glob
import logging
import itertools
import numpy as np
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
import math
# 导入混合精度训练所需的模块
from torch.cuda.amp import autocast, GradScaler

# ===================== 绘图默认设置 =====================
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Helvetica', 'Bitstream Vera Sans']
plt.rcParams['font.size'] = 14
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12


try:
    from flash_attn import flash_attn_func
except ImportError:
    flash_attn_func = None
    print("Warning: flash_attn not found. Using standard attention mechanism.")


##################################################
# 1. 配置
##################################################

class Config:
    """
    用于帧分类任务的配置
    """
    # ------------------- 数据相关 -------------------
    data_root = "/data2/ysn/ds005237_stroop/"
    label_root = "/home/ysn/PycharmProjects/ResTran/Downstream/label/"

    spatial_dim = (80, 96, 80)  # D, H, W
    total_time = 284  # 每条数据的时间序列长度
    time_step = 0.8  # 时间分辨率
    in_chans = 1


    # ------------------- 滑动窗口 -------------------
    window_size = 16
    embed_dim_intermediate = 16
    window_step = 8  # 窗口滑动步长
    latency = 4  # 每帧对应 time_lag 帧后的标签


    # ------------------- 模型相关 -------------------
    pretrained_encoder_path = "/data3/yyy/MAE_st_1/model/mae_fmri_epoch_26.pth"
    # pretrained_encoder_path = None
    # Encoder 的超参
    patch_size = (8, 8, 8)
    embed_dim = 1024
    depth = 6
    num_heads = 8
    mlp_ratio = 4.0

    # ------------------- 优化器训练 -------------------
    batch_size = 16
    learning_rate = 1e-4
    weight_decay = 1e-5
    epochs = 50
    num_workers = 8
    train_val_ratio = 0.75
    loss_mode = 'focal'

    # ------------------- 单GPU配置 -------------------
    device = 'cuda:1'  # 使用单个GPU

    # ------------------- 其他 -------------------
    log_dir = "/home/ysn/PycharmProjects/MAE/Stroop/result/"



# 创建必要的目录
os.makedirs(Config.log_dir, exist_ok=True)


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
        row_sums[row_sums == 0] = 1  # 避免除零错误
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
# 3. focal loss备用
##################################################
class MultiClassFocalLossWithAlpha(nn.Module):
    def __init__(self, alpha=None, gamma=5, reduction='mean'):
        """
        :param alpha: 权重系数列表
        :param gamma: 困难样本挖掘的gamma
        :param reduction:
        """
        super(MultiClassFocalLossWithAlpha, self).__init__()
        if alpha is None:
            alpha = [0.33, 0.33, 0.33]
        self.alpha = torch.tensor(alpha)
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, pred, target):
        alpha = self.alpha[target]  # 为当前batch内的样本，逐个分配类别权重，shape=(bs), 一维向量
        log_softmax = torch.log_softmax(pred, dim=1) # 对模型裸输出做softmax再取log, shape=(bs, 3)
        log_pt = torch.gather(log_softmax, dim=1, index=target.view(-1, 1))  # 取出每个样本在类别标签位置的log_softmax值, shape=(bs, 1)
        log_pt = log_pt.view(-1)  # 降维，shape=(bs)
        ce_loss = -log_pt  # 对log_softmax再取负，就是交叉熵了
        pt = torch.exp(log_pt)  #对log_softmax取exp，把log消了，就是每个样本在类别标签位置的softmax值了，shape=(bs)
        focal_loss = alpha * (1 - pt) ** self.gamma * ce_loss  # 根据公式计算focal loss，得到每个样本的loss值，shape=(bs)
        if self.reduction == "mean":
            return torch.mean(focal_loss)
        if self.reduction == "sum":
            return torch.sum(focal_loss)
        return focal_loss

##################################################
# 4. 模型组件: 3D时间通道编码器 + 分类器
##################################################

class TimeEmbedModule(nn.Module):
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
        x = x.reshape(N*C, T, D, H, W)
        x = self.time_embedder(x)
        x = x.reshape(N, self.embed_dim, D, H, W)
        return x


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
        self.time_embedder = TimeEmbedModule(
            time_steps = Config.window_size,
            embed_dim_intermediate=Config.embed_dim_intermediate)

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
        x = self.time_embedder(x)
        # => pass through 3D transformer => [B, L, embed_dim]
        x = self.vit3d(x)
        return x


class FrameLevelClassifier(nn.Module):
    """
    帧级分类器 - 使用rearrange将特征重组为(b t) (l v)格式
    """

    def __init__(self, embed_dim=384, time_steps=16, num_patches=256, num_classes=7):
        super().__init__()
        self.embed_dim = embed_dim
        self.time_steps = time_steps
        self.num_classes = num_classes
        self.num_patches = num_patches

        # 检查embed_dim是否可以被time_steps整除
        if embed_dim % time_steps != 0:
            raise ValueError("embed_dim必须能被time_steps整除以进行特征分解")

        # 每个时间步的特征维度
        self.v_dim = embed_dim // time_steps

        # 线性层接收num_patches * v_dim作为输入
        self.fc = nn.Linear(num_patches * self.v_dim, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # x: [B, L, embed_dim]
        B, L, E = x.shape

        # 使用rearrange重组特征
        # 'b l (t v) -> (b t) (l v)', t=时间帧数, v=embed_dim/时间帧数
        x = rearrange(x, 'b l (t v) -> (b t) (l v)', t=self.time_steps)
        # x形状: [B*16, L*(embed_dim/16)]

        x = self.dropout(x)
        # 线性分类
        logits = self.fc(x)  # [B*16, num_classes]

        # 重塑为[B, 16, num_classes]
        logits = logits.reshape(B, self.time_steps, self.num_classes)

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

        self.classifier = FrameLevelClassifier(
            embed_dim=embed_dim,
            time_steps=time_steps,
            num_patches=num_patches,
            num_classes=num_classes
        )

    def forward(self, x):
        # x: [B, 1, T=16, D, H, W]
        feat = self.encoder(x)  # => [B, L, embed_dim]
        logits = self.classifier(feat)  # => [B, 16, num_classes]
        return logits


##################################################
# 5. 数据集
##################################################

class StroopDataset(Dataset):
    def __init__(self,
                 data_path,
                 label_path,
                 latency,
                 window_size,
                 time_step,
                 window_step,
                 spatial_dim,
                 ):
        self.data_path = data_path
        self.label_path = label_path
        self.latency = latency
        self.window_size = window_size
        self.time_step = time_step #时间分辨率
        self.window_step = window_step
        self.spatial_dim = spatial_dim

        self.samples = []
        self.data_arrays = []

        for i, fpath in enumerate(self.data_path):
            data = np.load(fpath)
            if data.shape[1] != spatial_dim[0] or data.shape[2] != spatial_dim[1] or data.shape[3] != spatial_dim[2]:
                print(f"警告: 文件 {fpath} 空间形状 {data.shape} 与预期 {spatial_dim} 不符，跳过")
                continue
            self.data_arrays.append(data)

            total_time = data.shape[0]
            # 获取被试ID（每个被试就一个序列因此不需要序列号）
            subj_id = os.path.basename(fpath).split('_')[0]

            #找到对应的label
            subj_label = {}
            for j, lpath in enumerate(self.label_path):
                lname = os.path.basename(lpath)
                if subj_id in lname:
                    subj_label = np.load(lpath, allow_pickle=True)
                    break
            #打标
            subj_label_pro = np.zeros(total_time, dtype=int)
            for k in range(len(subj_label)):
                subj_label_pro[int(subj_label[k]['onset']/time_step)] = 1 if "con" in subj_label[k]['trial_type'] else 2
                subj_label_pro[int(subj_label[k]['onset'] / time_step)+1] = 1 if "con" in subj_label[k]['trial_type'] else 2
                # label_cur = 1 if "con" in subj_label[k]['trial_type'] else 0
                # for t in range(int(subj_label[k]['onset']/time_step),int(subj_label[k]['onset']/time_step+subj_label[k]['duration']/time_step)):
                #    subj_label_pro[t] = label_cur


            max_start = total_time - self.window_size - self.latency
            for start_t in range(0, max_start, self.window_step):
                end_t = start_t + self.window_size
                # 获取标签
                frame_labels = subj_label_pro[start_t + self.latency : end_t + self.latency]
                # 记录file_idx, start_t, end_t, subj_id和frame_labels
                self.samples.append((i, subj_id, start_t, end_t, frame_labels))
        print(f"共创建{len(self.samples)}个数据样本")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if idx >= len(self.samples):
            raise IndexError(f"Index{idx} out of range")

        fidx, subj_id, st, ed, frame_labels = self.samples[idx]
        data_arr = self.data_arrays[fidx]
        clip = data_arr[st:ed]  # (16, 80, 96, 80)

        clip_torch = torch.from_numpy(clip).float()  # [16, 80, 96, 80]

        # 全局时空归一化 - 对整个窗口计算单一的均值和标准差
        clip_torch = (clip_torch - clip_torch.mean()) / (clip_torch.std() + 1e-6)

        frame_labels = torch.tensor(frame_labels, dtype=torch.long)

        # 增加channel维度 => [1, 16, 80, 96, 80]
        clip_torch = clip_torch.unsqueeze(0)

        # 片段ID - 使用文件名+起始帧作为标识
        segment_id = f"{os.path.basename(self.data_path[fidx])}_{st}"

        return clip_torch, frame_labels, subj_id, segment_id


def train_one_epoch(model, train_loader, optimizer, scaler, device, epoch, logger, class_weights=None):
    """
    训练一个epoch - 使用混合精度训练
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    # 使用带权重的损失函数
    if Config.loss_mode == "cross_weighted":
        class_weights = class_weights.to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    elif Config.loss_mode == "focal":
        criterion = MultiClassFocalLossWithAlpha(alpha=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()

    # 存储被试级预测和标签
    subject_preds = {}
    subject_labels = {}
    all_preds = []
    all_labels = []

    for batch_x, batch_y, subject_ids, segment_ids in tqdm(train_loader, desc=f"Train Epoch {epoch + 1}"):
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        optimizer.zero_grad(set_to_none=True)  # 更高效的梯度清零

        # 使用混合精度训练
        with autocast():
            # 前向传播
            logits = model(batch_x)  # [B, 16, num_classes]

            # 计算损失
            B, T, C = logits.shape

            logits_flat = logits.reshape(-1, C)
            labels_flat = batch_y.reshape(-1)

            if Config.loss_mode == "cross_weighted":
                # 动态调整时间样本权重
                event_mask = (labels_flat != 0)
                event_weight = 0.3 * (1 + 0.5 * torch.sigmoid(torch.tensor(epoch - 10)))
                base_loss = criterion(logits_flat, labels_flat)
                event_loss = criterion(logits_flat[event_mask], labels_flat[event_mask])
                loss = base_loss + event_weight * event_loss

            else:
                loss = criterion(logits_flat, labels_flat)

        # 使用梯度缩放器进行反向传播和优化器更新
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

        # 获取预测类别
        preds = torch.argmax(logits, dim=2)  # [B, 16]

        # 统计准确率
        correct += (preds == batch_y).sum().item()
        total += batch_y.numel()

        # 收集所有预测和标签
        for b in range(B):
            subj_id = subject_ids[b]

            if subj_id not in subject_preds:
                subject_preds[subj_id] = []
                subject_labels[subj_id] = []

            for t in range(T):
                pred = preds[b, t].item()
                label = batch_y[b, t].item()

                all_preds.append(pred)
                all_labels.append(label)

                subject_preds[subj_id].append(pred)
                subject_labels[subj_id].append(label)

    # 计算平均损失和准确率
    avg_loss = total_loss / len(train_loader)
    avg_acc = correct / total

    # 计算每个类别的准确率和F1
    num_classes = logits.size(-1)
    class_acc, class_f1 = calculate_class_metrics(all_labels, all_preds, num_classes)

    # 计算被试级别的准确率和F1矩阵
    subject_acc_matrix, subject_f1_matrix = calculate_subject_metrics(
        subject_preds, subject_labels, num_classes)

    # 记录训练信息
    logger.info(f"Epoch {epoch + 1} Train: loss={avg_loss:.4f}, acc={avg_acc:.4f}")
    logger.info(f"Per-class train acc: {class_acc}")

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


'''验证函数'''
@torch.no_grad()
def validate(model, val_loader, device, epoch, class_names):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    # 验证时不使用加权损失
    criterion = nn.CrossEntropyLoss()

    # 存储被试级预测和标签
    subject_preds = {}
    subject_labels = {}
    all_preds = []
    all_labels = []
    all_logits = []

    for batch_x, batch_y, subject_ids, segment_ids in tqdm(val_loader, desc=f"Validate Epoch {epoch + 1}"):
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        # 使用混合精度训练
        with autocast():
            # 前向传播
            logits = model(batch_x)  # [B, 16, num_classes]

            # 计算损失
            B, T, C = logits.shape
            logits_flat = logits.reshape(-1, C)
            labels_flat = batch_y.reshape(-1)

            loss = criterion(logits_flat, labels_flat)

        total_loss += loss.item()

        # 获取预测类别
        preds = torch.argmax(logits, dim=2)  # [B, 16]

        correct += (preds == batch_y).sum().item()
        total += batch_y.numel()

        # 收集所有预测和标签
        for b in range(B):
            subj_id = subject_ids[b]

            if subj_id not in subject_preds:
                subject_preds[subj_id] = []
                subject_labels[subj_id] = []

            for t in range(T):
                pred = preds[b, t].item()
                label = batch_y[b, t].item()
                logit = logits[b, t].detach().cpu().numpy()

                all_preds.append(pred)
                all_labels.append(label)
                all_logits.append(logit)

                subject_preds[subj_id].append(pred)
                subject_labels[subj_id].append(label)

    # 计算平均损失和准确率
    avg_loss = total_loss / len(val_loader)
    avg_acc = correct / total

    # 计算每个类别的准确率和F1
    num_classes = len(class_names)
    class_acc, class_f1 = calculate_class_metrics(all_labels, all_preds, num_classes)

    # 计算被试级别的准确率和F1矩阵
    subject_acc_matrix, subject_f1_matrix = calculate_subject_metrics(subject_preds, subject_labels, num_classes)

    # 混淆矩阵
    cm = confusion_matrix(all_labels, all_preds, labels=range(num_classes))

    # ROC曲线
    all_logits = np.array(all_logits)

    results = {
        'loss': avg_loss,
        'accuracy': avg_acc,
        'class_acc': class_acc,
        'class_f1': class_f1,
        'subject_acc_matrix': subject_acc_matrix,
        'subject_f1_matrix': subject_f1_matrix,
        'predictions': all_preds,
        'labels': all_labels,
        'probabilities': all_logits,
        'confusion_matrix': cm,
    }

    return results


'''主函数'''


def main():
    # 检查GPU
    if not torch.cuda.is_available():
        raise RuntimeError("No GPU available.")
    device = torch.device(Config.device)
    print(f"[INFO] Using device: {device}")

    class_names = ['Inc', 'Con', 'Rest']
    num_classes = len(class_names)

    # 获取所有文件
    all_files = [os.path.join(Config.data_root, f)
                 for f in os.listdir(Config.data_root)
                 if f.endswith('.npy')]

    # 随机分组
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
    train_subject_ids = subject_ids[:int(Config.train_val_ratio * len(subject_ids))]
    val_subject_ids = subject_ids[int(Config.train_val_ratio * len(subject_ids)):]

    # 获取训练和验证文件
    train_files = []
    for subj_id in train_subject_ids:
        train_files.extend(subject_files[subj_id])

    val_files = []
    for subj_id in val_subject_ids:
        val_files.extend(subject_files[subj_id])

    print(f"[INFO] 训练被试数: {len(train_subject_ids)}, 训练文件数: {len(train_files)}")
    print(f"[INFO] 验证被试数: {len(val_subject_ids)}, 验证文件数: {len(val_files)}")

    # 获取所有标签
    label_files = [os.path.join(Config.label_root, f)
                   for f in os.listdir(Config.label_root)
                   if f.endswith('.npy')]

    # 创建数据集和dataloader
    train_dataset = StroopDataset(data_path=train_files,
                                  label_path=label_files,
                                  latency=Config.latency,
                                  window_size=Config.window_size,
                                  time_step=Config.time_step,
                                  window_step=Config.window_step,
                                  spatial_dim=Config.spatial_dim,
                                  )
    print("训练集创建完毕")

    val_dataset = StroopDataset(data_path=val_files,
                                label_path=label_files,
                                latency=Config.latency,
                                window_size=Config.window_size,
                                time_step=Config.time_step,
                                window_step=Config.window_step,
                                spatial_dim=Config.spatial_dim,
                                )
    print("验证集创建完毕")

    # 标签较为不均匀，rest很多，计算各类别权重
    all_train_labels = []
    for i in range(len(train_dataset)):
        _, frame_labels, _, _ = train_dataset[i]
        all_train_labels.extend(frame_labels.numpy())

    class_counts = Counter(all_train_labels)
    class_weights = 1.0 / torch.tensor(list(class_counts.values()), dtype=torch.float32)
    class_weights = class_weights / class_weights.sum() * len(class_counts)
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

    print(f"[INFO]Class distribution: {class_counts}")
    print(f"[INFO]Class weights: {class_weights}")


    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.batch_size,
        shuffle=True,
        num_workers=Config.num_workers,
        drop_last=False,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
        drop_last=False,
        pin_memory=True
    )

    # 创建模型
    model = MAE_FMRI_Classifier(
        time_steps=Config.window_size,
        spatial_dim=Config.spatial_dim,
        patch_size=Config.patch_size,
        in_chans=Config.in_chans,
        embed_dim=Config.embed_dim,
        depth=Config.depth,
        num_heads=Config.num_heads,
        mlp_ratio=Config.mlp_ratio,
        num_classes=num_classes,
        pretrained_path=Config.pretrained_encoder_path
    )

    best_val_acc = 0
    best_model_path = ""
    model = model.to(device)


    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=Config.learning_rate,
        weight_decay=Config.weight_decay
    )

    start_epoch = 0  # 默认从头开始
    checkpoint_path = ""
    scaler = GradScaler()

    if os.path.exists(checkpoint_path):
        print(f"加载检查点: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)

        # 恢复模型和优化器状态
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # 设置起始epoch（从保存的epoch+1开始）
        start_epoch = checkpoint['epoch']
        print(f"从 epoch {start_epoch} 恢复训练")

    # 学习率调度器（注意调整last_epoch）
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=Config.epochs,
        eta_min=1e-6,
        last_epoch=start_epoch - 1
    )

    # 创建日志目录
    date_str = time.strftime("%m%d_%H%M", time.localtime())
    log_name = f"Stroop_MAE_Vit_{date_str}"
    log_dir = os.path.join(Config.log_dir, log_name)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(os.path.join(log_dir, "figures"), exist_ok=True)
    os.makedirs(os.path.join(log_dir, "models"), exist_ok=True)

    # 创建日志记录器
    logger = get_logger(os.path.join(log_dir, "training.log"))
    logger.info(f"==== Volume_Wise (Mixed Precision) Classification for Stroop Task: {log_name} ====")

    # 记录训练配置
    for key, value in Config.__dict__.items():
        if not key.startswith('__'): logger.info(f"{key}: {value}")

    # 训练循环
    for epoch in range(Config.epochs):

        # 训练一个epoch，传入scaler用于混合精度训练
        train_results = train_one_epoch(model, train_loader, optimizer, scaler, device, epoch, logger, class_weights)

        # 更新学习率
        lr_scheduler.step()

        # 绘制训练结果的可视化图表
        # 1. 混淆矩阵
        cm = confusion_matrix(train_results['all_labels'], train_results['all_preds'], labels=range(num_classes))
        save_confusion_matrix(
            cm, class_names,
            os.path.join(log_dir, "figures", f"train_epoch_{epoch + 1}.png"),
            normalize=True, title=f"Train Confusion (Epoch {epoch + 1})"
        )

        # 验证
        logger.info(f"--- Evaluating epoch {epoch + 1} ---")
        val_results = validate(model, val_loader, device, epoch, class_names)

        # 记录验证结果
        logger.info(
            f"Epoch {epoch + 1} Val: loss={val_results['loss']:.4f}, acc={val_results['accuracy']:.4f}")
        # logger.info(f"Per-class val acc: {val_results['class_acc']}")

        # 绘制验证结果的可视化图表
        # 1. 混淆矩阵
        save_confusion_matrix(
            val_results['confusion_matrix'],
            class_names,
            os.path.join(log_dir, "figures", f"val_epoch_{epoch + 1}.png"),
            normalize=True,
            title=f"Val Confusion (Epoch {epoch + 1})"
        )

        # 保存最佳模型
        if val_results['accuracy'] > best_val_acc:
            best_val_acc = val_results['accuracy']
            if best_model_path and os.path.exists(best_model_path):
                try:
                    os.remove(best_model_path)
                except OSError as e:
                    print(f"Error removing old model: {e}")
            best_model_path = os.path.join(log_dir, 'models',
                                           f"best_model_epoch_{epoch + 1}_acc_{best_val_acc:.4f}.pth")
            torch.save(model.state_dict(), best_model_path)
            logger.info(f"Saved new best model at epoch {epoch + 1} with validation accuracy: {best_val_acc:.4f}")

            # 每10个epoch保存一次检查点
        if (epoch + 1) % 10 == 0:
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