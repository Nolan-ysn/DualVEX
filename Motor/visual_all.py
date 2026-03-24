import os
import glob
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import defaultdict
from torch.cuda.amp import autocast
from matplotlib.patches import Rectangle

from matplotlib.colors import ListedColormap

# 导入您的模型定义

from model import (ClassifyConfig, FrameClassificationDataset,MAE_FMRI_Classifier, expand_predictions, vote_predictions)


# ================= 配置区域 =================
# 1. 最佳微调模型路径
MODEL_PATH = "/data3/ysn/Motor/Motor_MAE_ST_900_fine-tuning_1214_1642/models/best_model_epoch_31_acc_0.9524.pth"

# 2. Motor 任务数据目录
DATA_ROOT = "/data/wyy/about_cache/HCP_motor_labeled"

# 只加载 RL 的标签
LABEL_PATH_RL = "/data/wyy/HCP_motor_voxelnormed/RL_label.npy"
LABEL_PATH_LR = "/data/wyy/HCP_motor_voxelnormed/LR_label.npy"

VAL_FILES_PATH = "/data3/ysn/Motor/Motor_MAE_ST_900_fine-tuning_1214_1642/validation_files.npy"

# 3. 结果保存路径
OUTPUT_FILE = "motor_sequence_visualization_1.png"

RANK_START_PCT = 0.0
RANK_END_PCT = 1.0

# 4. 任务类别定义与颜色 (仿照 Fig.3)
CLASS_NAMES = ['Rest', 'LH', 'RH', 'LF', 'RF', 'Tongue', 'Cue']
COLOR_MAP = {
    0: '#E0E0E0',  # Rest:   浅灰
    1: '#f2b56f',  # LH:     珊瑚橙
    2: '#71b7ed',  # RH:     天蓝
    3: '#b8aeeb',  # LF:     中紫
    4: '#84c3b7',  # RF:     绿
    5: '#f57c6e',  # Tongue: 红
    6: '#fae69e'   # Cue:    淡黄
}

# 5. 设备
DEVICE = "cuda:5" if torch.cuda.is_available() else "cpu"


# ===========================================

def load_model():
    print(f"正在加载模型: {MODEL_PATH}")
    cfg = ClassifyConfig()
    cfg.num_classes = 7
    cfg.window_size = 16

    # 实例化模型

    model = MAE_FMRI_Classifier(
        time_steps=cfg.window_size,
        spatial_dim=cfg.spatial_dim,
        patch_size=cfg.patch_size,
        in_chans=cfg.in_chans,
        embed_dim=cfg.embed_dim,
        depth=cfg.depth,
        num_heads=cfg.num_heads,
        mlp_ratio=cfg.mlp_ratio,
        num_classes=cfg.num_classes
    ).to(DEVICE)


    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model


def get_test_data(phase):
    print(f"正在加载验证集文件列表: {VAL_FILES_PATH}")

    if not os.path.exists(VAL_FILES_PATH):
        print(f"错误: 找不到文件 {VAL_FILES_PATH}")
        exit()

    # 加载文件列表
    all_val_files = np.load(VAL_FILES_PATH)
    target_files = [str(f) for f in all_val_files if phase in str(f)]
    print(f"[{phase}] 文件数: {len(target_files)}")

    if len(target_files) == 0:
        print(f"警告: 没有找到 {phase} 文件。")
        exit()

    dataset = FrameClassificationDataset(
        file_paths=target_files,
        label_path_lr=LABEL_PATH_LR,
        label_path_rl=LABEL_PATH_RL,
        window_size=16,
        time_lag=4,
        step=4,
        spatial_dim=(80, 96, 80),
        total_time=284
    )

    loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=8)
    return loader

@torch.no_grad()
def run_inference(model, loader,gt_label_path):
    print("开始推理...")
    model.eval()

    # results[subject_id] = {'preds': [], 'starts': []}
    results = defaultdict(lambda: {'preds': [], 'starts': []})

    for batch_x, batch_y, batch_subjs, batch_seg_ids in tqdm(loader, desc="Inference"):
        batch_x = batch_x.to(DEVICE)

        with autocast():
            logits = model(batch_x)
            preds = torch.argmax(logits, dim=2).cpu().numpy()

        for i, subj_id in enumerate(batch_subjs):
            seg_id = batch_seg_ids[i]
            start_t = int(seg_id.split('.npy_')[1])

            results[subj_id]['preds'].append(preds[i])
            results[subj_id]['starts'].append(start_t)

    # 组装序列并排序
    final_results = []
    gt_label = np.load(gt_label_path)

    print("正在组装序列并计算准确率...")
    for subj_id, data in results.items():
        if len(data['preds']) == 0: continue

        expanded = expand_predictions(data['preds'], data['starts'], sequence_length=284)
        voted = vote_predictions(expanded)

        # 填充无效值
        voted[voted == -1] = 0

        # 计算 RL 准确率
        acc = np.mean(voted == gt_label)

        final_results.append({
            'subj': subj_id,
            'pred': voted,
            'acc': acc
        })

    # 按准确率排序 (高 -> 低)
    final_results.sort(key=lambda x: x['acc'], reverse=True)

    return final_results, gt_label


def draw_panel(fig, gs_main, row_idx, results, gt, phase_name):
    """
    在指定的 GridSpec 位置绘制一组 (GT + Results)
    gs_main: 主网格
    row_idx: 主网格的行索引 (0 或 1)
    """
    # 在主网格的指定行内，再创建一个子网格 (2行: 1行GT, 12行Results)
    gs_sub = gs_main[row_idx].subgridspec(2, 1, height_ratios=[1, 12], hspace=0.05)

    ax_gt = fig.add_subplot(gs_sub[0])
    ax_res = fig.add_subplot(gs_sub[1])

    # --- 数据准备 ---
    # 截取
    total = len(results)
    start = int(total * RANK_START_PCT)
    end = int(total * RANK_END_PCT)
    results_plot = results[start:end]
    num_subjs = len(results_plot)
    seq_len = 284

    gt_matrix = gt.reshape(1, -1)
    res_matrix = np.zeros((num_subjs, seq_len))
    for i, r in enumerate(results_plot):
        res_matrix[i, :] = r['pred']

    # Colormap
    c_list = [COLOR_MAP[i] for i in range(7)]
    cmap = ListedColormap(c_list)

    # --- 绘图 GT ---
    ax_gt.imshow(gt_matrix, cmap=cmap, aspect='auto', interpolation='nearest', vmin=0, vmax=6)
    rect_gt = Rectangle((-0.5, -0.5), seq_len, 1, fill=False, edgecolor='black', lw=1.5)
    ax_gt.add_patch(rect_gt)

    ax_gt.set_xticks([])
    ax_gt.set_yticks([])
    ax_gt.text(-0.02, 0.5, f'{phase_name}_GT', transform=ax_gt.transAxes,
               va='center', ha='right', fontsize=22, rotation=90)
    for spine in ax_gt.spines.values(): spine.set_visible(False)

    # --- 绘图 Results ---
    ax_res.imshow(res_matrix, cmap=cmap, aspect='auto', interpolation='nearest', vmin=0, vmax=6)
    rect_res = Rectangle((-0.5, -0.5), seq_len, num_subjs, fill=False, edgecolor='black', lw=1.5)
    ax_res.add_patch(rect_res)

    ax_res.set_yticks([])
    ax_res.text(-0.02, 0.5, f'{phase_name}_results', transform=ax_res.transAxes,
                va='center', ha='right', fontsize=22, rotation=90)

    # 强制去掉刻度
    ax_res.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False,
                       labelbottom=False, labelleft=False)
    ax_res.set_xticks([])


    for spine in ax_res.spines.values(): spine.set_visible(False)


# [修改] 组合绘图函数
def plot_combined_style(res_rl, gt_rl, res_lr, gt_lr, output_path):
    print("正在绘制组合图...")

    # 增加高度以容纳两组图
    fig = plt.figure(figsize=(12, 10))

    # 创建主网格：2行1列 (Row 0 for RL, Row 1 for LR)
    # hspace 控制两组大图之间的距离
    gs_main = fig.add_gridspec(2, 1, hspace=0.08)

    # 绘制 RL (上面)
    draw_panel(fig, gs_main, 0, res_rl, gt_rl, "RL")

    # 绘制 LR (下面)
    draw_panel(fig, gs_main, 1, res_lr, gt_lr, "LR")

    # 图例 (放在最顶端)
    patches = [mpatches.Patch(color=COLOR_MAP[i], label=CLASS_NAMES[i]) for i in range(7)]
    fig.legend(handles=patches, loc='upper center', bbox_to_anchor=(0.52, 1.0),
               ncol=7, frameon=False, fontsize=22,
               handlelength=1.7,  # 减小色块的宽度 (默认是2.0左右)
               handletextpad=0.4,  # 减小色块和文字之间的距离 (默认是0.8)
               columnspacing=1.1)  # 减小不同类别之间的距离 (默认是2.0)

    # 布局调整 (top 留给图例, bottom 留给 x轴标签)
    plt.subplots_adjust(left=0.08, right=0.98, top=0.93, bottom=0.05)

    plt.savefig(output_path, dpi=300)
    print(f"图表已保存至: {output_path}")


# 仅画RL时使用，RL LR都画时用draw_panel
def plot_color_strip_rl(results, gt_rl, output_path):
    total_subjs = len(results)
    start_idx = int(total_subjs * RANK_START_PCT)
    end_idx = int(total_subjs * RANK_END_PCT)

    results_to_plot = results[start_idx:end_idx]
    num_plot_subjs = len(results_to_plot)

    if num_plot_subjs == 0:
        print("错误：截取区间内没有被试，请检查数据量或百分比设置。")
        return

    print(
        f"总被试: {total_subjs}, 截取排名 {start_idx} 到 {end_idx} (Top {RANK_START_PCT * 100:.0f}% - {RANK_END_PCT * 100:.0f}%), 共 {num_plot_subjs} 人。")

    # 2. 准备绘图数据
    seq_len = 284

    # GT 矩阵 [1, T]
    gt_matrix = gt_rl.reshape(1, -1)

    # Results 矩阵 [N, T]
    res_matrix = np.zeros((num_plot_subjs, seq_len))
    for i, res in enumerate(results_to_plot):
        res_matrix[i, :] = res['pred']

    # 3. 设置画布布局
    # 使用 GridSpec 来分割上下两部分，中间留一点空隙
    fig = plt.figure(figsize=(12, 7))

    # height_ratios=[1, 12] 表示下面的图高度是上面的12倍 (结果区域大，GT区域小)
    # hspace=0.08 控制上下两图的间距
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 12], hspace=0.08)

    ax_gt = fig.add_subplot(gs[0])
    ax_res = fig.add_subplot(gs[1])

    # 4. 颜色映射
    c_list = [COLOR_MAP[i] for i in range(7)]
    cmap = ListedColormap(c_list)

    # 5. 绘制 GT
    ax_gt.imshow(gt_matrix, cmap=cmap, aspect='auto', interpolation='nearest', vmin=0, vmax=6)

    # [修改] 添加黑色边框 (使用 Rectangle)
    # xy=(-0.5, -0.5), width=seq_len, height=1
    rect_gt = Rectangle((-0.5, -0.5), seq_len, 1, fill=False, edgecolor='black', lw=1.5)
    ax_gt.add_patch(rect_gt)

    # [修改] 设置 GT 标签 (左侧纵向)
    ax_gt.set_yticks([])  # 去掉刻度
    ax_gt.set_xticks([])  # 去掉X轴
    # text 坐标 (-0.02, 0.5) 表示在轴左侧一点，垂直居中
    ax_gt.text(-0.02, 0.5, 'RL_GT', transform=ax_gt.transAxes,
               va='center', ha='right', fontsize=22, rotation=90)

    # 去掉自带边框线 (spines)，因为我们自己画了 Rectangle
    for spine in ax_gt.spines.values():
        spine.set_visible(False)

    # 6. 绘制 Results
    ax_res.imshow(res_matrix, cmap=cmap, aspect='auto', interpolation='nearest', vmin=0, vmax=6)

    # [修改] 添加黑色边框
    rect_res = Rectangle((-0.5, -0.5), seq_len, num_plot_subjs, fill=False, edgecolor='black', lw=1.5)
    ax_res.add_patch(rect_res)

    # [修改] 设置 Results 标签
    ax_res.set_yticks([])  # 去掉刻度
    ax_res.text(-0.02, 0.5, 'RL_results', transform=ax_res.transAxes,
                va='center', ha='right', fontsize=22, rotation=90)

    ax_gt.set_xticks([])

    ax_res.tick_params(axis='both',  # x和y轴
                       which='both',  # 主刻度和副刻度
                       bottom=False,  # 底部刻度线关
                       top=False,  # 顶部刻度线关
                       left=False,  # 左侧刻度线关
                       right=False,  # 右侧刻度线关
                       labelbottom=False,  # 底部数字关
                       labelleft=False)  # 左侧数字关
    # X轴标签
    # ax_res.set_xlabel('Time Frames (TRs)', fontsize=12)

    # 7. 图例
    patches = [mpatches.Patch(color=COLOR_MAP[i], label=CLASS_NAMES[i]) for i in range(7)]
    # 放在最上方
    fig.legend(handles=patches, loc='upper center', bbox_to_anchor=(0.52, 1.0),
               ncol=7, frameon=False, fontsize=22,
               handlelength=1.7,      # 减小色块的宽度 (默认是2.0左右)
               handletextpad=0.4,     # 减小色块和文字之间的距离 (默认是0.8)
               columnspacing=1.1  )   # 减小不同类别之间的距离 (默认是2.0)

    # 调整整体布局，给左边的文字和上面的图例留空间
    # left=0.08 (左边留白给label), top=0.9 (上面留白给legend)
    plt.subplots_adjust(left=0.08, right=0.98, top=0.9, bottom=0.1)

    plt.savefig(output_path, dpi=300)
    print(f"图表已保存至: {output_path}")

def main():
    model = load_model()

    print("\n=== Processing RL Data ===")
    loader_rl = get_test_data(phase="RL")
    results_rl, gt_rl = run_inference(model, loader_rl, LABEL_PATH_RL)

    print("\n=== Processing LR Data ===")
    loader_lr = get_test_data(phase="LR")
    results_lr, gt_lr = run_inference(model, loader_lr, LABEL_PATH_LR)

    plot_combined_style(results_rl, gt_rl, results_lr, gt_lr, OUTPUT_FILE)

if __name__ == "__main__":
    main()