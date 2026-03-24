import os
import glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from collections import defaultdict
import random
import string

# ==============================================================================
# 步骤 1 & 2: 导入模型并创建特征提取器 (与之前脚本相同)
# ==============================================================================
try:
    from vote import EncoderClassifier, ClassifyConfig
except ImportError:
    print("错误：无法从 'vote.py' 导入所需类。请确保此脚本与 vote.py 在同一目录下。")
    exit()


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
# 步骤 3: 数据集类，返回窗口和对应的被试ID
# ==============================================================================
class HCP_Window_Subject_Dataset(Dataset):
    def __init__(self, file_paths, label_map, window_size, spatial_dim):
        self.file_paths = file_paths
        self.label_map = label_map
        self.window_size = window_size
        self.spatial_dim = spatial_dim
        self.samples = self._create_samples()

    def _create_samples(self):
        samples = []
        num_windows_per_scan = 1  # 从每个扫描文件采样4个窗口
        for fpath in tqdm(self.file_paths, desc="创建数据集样本"):
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
                print(f"跳过文件 {fpath}，原因: {e}")
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fpath, st, subject_id, label = self.samples[idx]
        data = np.load(fpath, mmap_mode='r')
        clip = data[st: st + self.window_size].copy()
        window = torch.from_numpy(clip).float()
        window = (window - window.mean()) / (window.std() + 1e-6)
        return window, torch.tensor(label, dtype=torch.long), subject_id


# ==============================================================================
# 步骤 4: 主执行逻辑
# ==============================================================================
if __name__ == "__main__":
    # ================== 路径配置 ==================
    PRETRAINED_MAE_MODEL_PATH = "/data3/yyy/MAE_st_1/model/mae_fmri_epoch_26.pth"

    OUTPUT_DIR = "/home/ysn/PycharmProjects/MAE/HCP_7Classify/result/"

    VAL_FILES_PATH = os.path.join(OUTPUT_DIR, "validation_files.npy")
    NUM_SUBJECT = 50

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    TSNE_SAVE_PATH = os.path.join(OUTPUT_DIR, "tsne_pca_pretrain_final.png")

    # --- 配置 ---
    cfg = ClassifyConfig()
    cfg.device = "cuda:1" if torch.cuda.is_available() else "cpu"
    device = torch.device(cfg.device)
    print(f"使用设备: {device}")
    tasks = ['EMOTION', 'GAMBLING', 'LANGUAGE', 'MOTOR', 'RELATIONAL', 'SOCIAL', 'WM']
    label_map = {task: i for i, task in enumerate(tasks)}

    # --- 加载微调前的模型 ---
    print("创建下游任务模型结构 (EncoderClassifier)...")
    model = FeatureExtractor(cfg).to(device)
    print(f"从 '{PRETRAINED_MAE_MODEL_PATH}' 加载权重...")
    checkpoint = torch.load(PRETRAINED_MAE_MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint['encoder_state_dict'], strict=False)
    model.eval()
    print("权重加载成功。")

    # --- 准备 HCP 数据子集  ---
    print(f"--- 准备 HCP 数据子集 (从验证集) ---")
    try:
        print(f"正在从 '{VAL_FILES_PATH}' 加载验证集文件列表...")
        all_hcp_files = np.load(VAL_FILES_PATH, allow_pickle=True)
    except FileNotFoundError:
        print(f"错误: 验证集文件 '{VAL_FILES_PATH}' 未找到！请确保路径正确。")
        exit()
    #只用LR的数据
    all_hcp_files = [f for f in all_hcp_files if "LR" in os.path.basename(f)]
    print(f"总LR文件数{len(all_hcp_files)}")

    subject_files = defaultdict(list)
    for f in all_hcp_files: subject_files[os.path.basename(f).split('_')[0]].append(f)

    all_subjects = list(subject_files.keys())
    # random.shuffle(all_subjects)

    selected_subjects = all_subjects[:NUM_SUBJECT]
    final_file_list = [f for subj_id in selected_subjects for f in subject_files[subj_id]]
    print(f"已随机选择 {len(selected_subjects)} 位被试，共计 {len(final_file_list)} 个扫描文件。")

    dataset = HCP_Window_Subject_Dataset(final_file_list, label_map, cfg.window_size, cfg.spatial_dim)
    loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=8)
    print(f"数据集已创建，共 {len(dataset)} 个窗口样本。")

    # --- 提取所有 (N, L, E) 特征 ---
    all_features_list, all_labels_list, all_subjects_list = [], [], []
    with torch.no_grad():
        for batch_x, batch_y, batch_subject_ids in tqdm(loader, desc="提取特征、标签和被试ID"):
            features = model.forward_features(batch_x.to(device))
            all_features_list.append(features.cpu().numpy())
            all_labels_list.append(batch_y.cpu().numpy())
            all_subjects_list.extend(batch_subject_ids)

    all_features = np.concatenate(all_features_list, axis=0)
    all_labels = np.concatenate(all_labels_list, axis=0)
    final_subjects = all_subjects_list
    print(f"特征提取完毕。原始特征形状: {all_features.shape}")

    # --- 应用 "Spatial PCA" 降维方法 ---
    N, L, E = all_features.shape
    features_reshaped = all_features.transpose(0, 2, 1).reshape(N * E, L)

    num_spatial_components = 64
    print(f"正在对 {L} 维的空间信息进行PCA，降至 {num_spatial_components} 维...")
    pca_spatial = PCA(n_components=num_spatial_components, random_state=42)
    features_pca = pca_spatial.fit_transform(features_reshaped)

    final_features_for_tsne = features_pca.reshape(N, E * num_spatial_components)
    print(f"自定义PCA流程完成。最终用于t-SNE的特征维度: {final_features_for_tsne.shape}")

    # --- 运行 t-SNE 并按被试ID可视化 ---
    print("运行 t-SNE 降维...")
    tsne = TSNE(n_components=2, perplexity=10, n_iter=5000, random_state=42, n_jobs=-1)
    tsne_results = tsne.fit_transform(final_features_for_tsne)
    print("t-SNE 运行完毕。")

    # --- 准备绘图所需的数据和映射 ---
    df = pd.DataFrame({
        'tsne-1': tsne_results[:, 0], 'tsne-2': tsne_results[:, 1],
        'label': all_labels, 'subject': final_subjects
    })

    # 任务 -> 颜色 映射
    genshin_element_colors = ['#FF4D4D', '#45AAF2', '#46DDC3', '#A955E3', '#A6C938', '#A0E3E8', '#FAB632']
    color_dict = {i: color for i, color in enumerate(genshin_element_colors)}
    df['color'] = df['label'].map(color_dict)

    # 被试 -> 字母 映射
    letter_markers = list(string.ascii_lowercase + string.ascii_uppercase)
    subject_to_letter = {subject_id: letter_markers[i] for i, subject_id in enumerate(selected_subjects)}
    df['letter'] = df['subject'].map(subject_to_letter)

    # --- 步骤 1: 诊断 - 打印出坐标范围和分位数，以帮助您决定手动范围 ---
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
    MANUAL_X_LIMITS = (np.quantile(x_coords, 0.005)-10, np.quantile(x_coords, 0.995)+10)  # <-- 在这里手动设置X轴范围 (最小值, 最大值)
    MANUAL_Y_LIMITS = (np.quantile(y_coords, 0.005)-10, np.quantile(y_coords, 0.995)+10)  # <-- 在这里手动设置Y轴范围 (最小值, 最大值)


    #  创建图表和坐标轴
    fig, ax = plt.subplots(figsize=(7, 6))

    #  遍历DataFrame中的每一行，并用 ax.text() 绘制字母
    print("正在绘制字母标记点...")
    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="绘制点"):
        x, y = row['tsne-1'], row['tsne-2']
        if MANUAL_X_LIMITS[0] <= x <=MANUAL_X_LIMITS[1] and MANUAL_Y_LIMITS[0] <= y <= MANUAL_Y_LIMITS[1]:
            ax.text(x, y, row['letter'], color=row['color'],
                    fontdict={'weight': 'bold', 'size': 13}, ha='center', va='center',alpha=0.5)

    # **应用您手动设置的坐标轴范围**
    ax.set_xlim(MANUAL_X_LIMITS)
    ax.set_ylim(MANUAL_Y_LIMITS)
    # 6. 设置标题和标签
    ax.set_title('', fontsize=17)
    ax.set_xlabel('X', fontsize=16)
    ax.set_ylabel('Y', fontsize=16)

    # 7.不显示刻度
    ax.set_xticks([])
    ax.set_yticks([])
    # 8. 调整布局以避免标签被裁切
    plt.tight_layout()

    # 9. 保存图像
    #    由于没有外部图例，bbox_inches='tight' 不是必需的，但保留也无妨
    plt.savefig(TSNE_SAVE_PATH, dpi=300, bbox_inches='tight')

    print(f"t-SNE 可视化图像已保存至: {TSNE_SAVE_PATH}")