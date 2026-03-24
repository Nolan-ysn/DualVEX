import os
import glob
import numpy as np
import torch
from captum.attr import IntegratedGradients
from nilearn import plotting, image
from scipy.ndimage import zoom,grey_dilation
import nibabel as nib
from nilearn import image
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from matplotlib.colors import LinearSegmentedColormap
from tqdm import tqdm

# 导入模型定义
try:
    from vote import EncoderClassifier, ClassifyConfig
except ImportError:
    print("错误：请确保此脚本与 vote.py 在同一目录下")
    exit()

# ================= 配置区域 =================

MODEL_PATH = "/home/ysn/PycharmProjects/MAE/HCP_7Classify/result/HCP_7Class_MAE_ST_900_fine-tuning_20251205_110214/models/best_model_epoch_16_acc_0.9617.pth"

HCP_DATA_ROOT = "/data3/ysn/HCP_Seven_Tasks/"

REF_2MM_PATH = "/home/amax/cacheNeuroScience/HCP_TMP/AllConCutAndLabel/100206_EMOTION_LR-0_fear.nii" #重采样前的数据，用来做仿射

TEMPLATE_1MM_PATH = "/home/amax/abin/auto_backup.linux_ubuntu_16_64/MNI152_2009_template.nii.gz" #模版文件

# 选项: 'EMOTION', 'GAMBLING', 'LANGUAGE', 'MOTOR', 'RELATIONAL', 'SOCIAL', 'WM'
TARGET_TASK = "WM"

# 聚合样本数量 (取多少个样本的平均值)
NUM_SAMPLES_TO_AVERAGE = 30

OUTPUT_DIR = "/data3/ysn/LPR_HCP/"

DEVICE = "cuda:4" if torch.cuda.is_available() else "cpu"

# 任务映射
TASK_NAMES = ['EMOTION', 'GAMBLING', 'LANGUAGE', 'MOTOR', 'RELATIONAL', 'SOCIAL', 'WM']
LABEL_MAP = {name: i for i, name in enumerate(TASK_NAMES)} 


# ===========================================

def load_model():
    print(f"正在加载模型: {MODEL_PATH}")
    cfg = ClassifyConfig()
    cfg.num_classes = 7
    # 确保 window_size 与训练时一致
    cfg.window_size = 16

    model = EncoderClassifier(cfg).to(DEVICE)

    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    # 兼容不同的保存格式
    state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
    if 'state_dict' in checkpoint and 'encoder.patch_embed.proj.weight' not in state_dict:
        state_dict = checkpoint['state_dict']

    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model


def get_task_files(root_dir, task_name, num_files):
    """获取指定任务的文件列表"""

    all_files = glob.glob(os.path.join(root_dir, "*.npy"))
    task_files = []
    for f in all_files:
        basename = os.path.basename(f)
        if task_name in basename:
            task_files.append(f)

    if len(task_files) == 0:
        raise ValueError(f"未找到包含 {task_name} 的文件，请检查路径或文件名模式。")

    # 随机打乱取前 N 个
    np.random.shuffle(task_files)
    return task_files[:num_files]


def preprocess_hcp_data(npy_path):
    """读取 HCP 数据 并预处理"""
    data = np.load(npy_path)
    data = data[:16] # [16, 80, 96, 80]

    # 转换为 Tensor
    data = torch.from_numpy(data).float()

    # 增加 Batch 维度 -> [1, 16, 80, 96, 80]
    data = data.unsqueeze(0)

    # 归一化 (Instance Norm)
    data = (data - data.mean()) / (data.std() + 1e-6)

    return data.to(DEVICE)


def compute_group_attribution(model, file_list, target_idx):
    """计算一组文件的平均归因图，并在累加前对每个样本进行平滑"""
    ig = IntegratedGradients(model)
    accumulated_attr = None
    count = 0

    print(f"开始计算 {len(file_list)} 个样本的平均 Integrated Gradients (含 FWHM=8 平滑)...")

    for fpath in tqdm(file_list):
        try:
            input_tensor = preprocess_hcp_data(fpath)
            input_tensor.requires_grad = True

            # 1. 计算重要性 [1, 16, 80, 96, 80]
            attr = ig.attribute(input_tensor, target=target_idx, n_steps=20)

            # 2. 取绝对值并转为 numpy
            curr_attr = np.abs(attr.detach().cpu().numpy())  # [1, 16, 80, 96, 80]

            # =================== 单样本平滑逻辑 ===================
            # nilearn.image.smooth_img 需要 NIfTI 对象 (X, Y, Z) 或 (X, Y, Z, T)
            # 我们的数据格式是 (Batch, Time, D, H, W)，需要转换维度顺序

            # 2.1 去掉 Batch 维度 -> [16, 80, 96, 80]
            data_sq = curr_attr[0]

            # 2.2 转置为 NIfTI 标准格式 [X, Y, Z, T] -> [80, 96, 80, 16]
            # 注意：这里 D, H, W 对应 X, Y, Z
            data_transposed = data_sq.transpose(1, 2, 3, 0)

            # 2.3 构建临时 NIfTI 对象
            # 使用单位矩阵 np.eye(4) 作为仿射矩阵。
            # 这意味着我们假设 1个体素 = 1mm。因为您的 patch_size 是 8x8x8 体素，
            # 设置 fwhm=8 就相当于在空间上平滑大约 1 个 patch 的范围，
            temp_nii = nib.Nifti1Image(data_transposed, affine=np.eye(4))

            # 2.4 应用平滑
            smoothed_nii = image.smooth_img(temp_nii, fwhm=8)

            # 2.5 取回数据并还原维度
            smoothed_data = smoothed_nii.get_fdata()  # [80, 96, 80, 16]

            # [X, Y, Z, T] -> [T, X, Y, Z] -> [16, 80, 96, 80]
            smoothed_data = smoothed_data.transpose(3, 0, 1, 2)

            # 恢复 Batch 维度 -> [1, 16, 80, 96, 80]
            curr_attr_smoothed = np.expand_dims(smoothed_data, axis=0)
            # ==========================================================

            # 3. 累加平滑后的结果
            if accumulated_attr is None:
                accumulated_attr = curr_attr_smoothed
            else:
                accumulated_attr += curr_attr_smoothed
            count += 1

        except Exception as e:
            print(f"处理文件 {fpath} 出错: {e}")
            continue

    if count == 0:
        return None

    # 计算平均值
    mean_attr = accumulated_attr / count
    return mean_attr

def visualize_and_save(attr_map_4d, task_name):
    """
    attr_map_4d: [1, 16, 80, 96, 80]
    """
    # 1. 数据降维: [1, 16, 80, 96, 80] -> [80, 96, 80]
    heatmap_3d = np.mean(attr_map_4d[0], axis=0)

    # 2. 加载参考文件 (2mm) 和 模版文件 (1mm)
    try:
        ref_img_2mm = nib.load(REF_2MM_PATH)
        template_img_1mm = nib.load(TEMPLATE_1MM_PATH)
    except FileNotFoundError as e:
        print(f"错误: 无法加载参考文件或模版文件: {e}")
        return

    # 获取原始数据的形状
    orig_shape = ref_img_2mm.shape[:3]
    curr_shape = heatmap_3d.shape  # (80, 96, 80)

    print(f"检测到尺寸差异: 模型输出 {curr_shape} vs 参考文件 {orig_shape}")
    print("正在将热力图还原回原始尺寸...")

    # 计算缩放因子 (Target / Current)
    zoom_factors = [
        orig_shape[0] / curr_shape[0],
        orig_shape[1] / curr_shape[1],
        orig_shape[2] / curr_shape[2]
    ]

    # 使用样条插值将热力图放大回原始尺寸
    # order=1 (线性插值) 通常对热力图足够且平滑
    heatmap_orig_space = zoom(heatmap_3d, zoom_factors, order=1)

    # 3. 创建初始 NIfTI 对象 (在 2mm 空间)
    initial_img = nib.Nifti1Image(heatmap_orig_space, affine=ref_img_2mm.affine)

    # 4. (可选) 确保对齐到 2mm 参考空间
    # 这步是为了防止 heatmap_3d 的 grid 与 ref_img 的 grid 有微小差异
    img_in_2mm = image.resample_to_img(
        source_img=initial_img,
        target_img=ref_img_2mm,
        interpolation='linear'
    )

    # 5. 重采样到 1mm 标准空间 (关键步骤)
    print("正在重采样到 MNI 1mm 空间...")
    img_in_1mm = image.resample_to_img(
        source_img=img_in_2mm,
        target_img=template_img_1mm,
        interpolation='linear'
    )

    data_1mm = img_in_1mm.get_fdata()

    # 5.1 清除 NaN：将重采样产生的 NaN 变为 0
    data_1mm = np.nan_to_num(data_1mm)

    # 5.2 形态学膨胀 (Dilation) - 关键步骤
    # size=(3,3,3) 表示向周围扩展 1 个像素 (1mm)
    # 这会让激活区域稍微变“胖”一点，从而填满模版边缘的缝隙
    # 同时它会取局部最大值，有助于提升整体的视觉亮度
    print("正在进行形态学膨胀以填充边缘...")
    data_dilated = grey_dilation(data_1mm, size=(3, 3, 3))

    # 重新封装为 NIfTI
    img_final = nib.Nifti1Image(data_dilated, img_in_1mm.affine)

    # =================================================================

    # 获取数据的最大值和最小值
    v_max = np.nanmax(data_dilated)
    # 为了背景干净，vmin 设为最大值的 10% 左右，或者直接 0
    v_min = 0

    # 6. 保存最终的 1mm NIfTI 文件
    nii_out = os.path.join(OUTPUT_DIR, f"HCP_{task_name}_1mm_30.nii.gz")
    nib.save(img_in_1mm, nii_out)
    print(f"nii文件已保存: {nii_out}")

    # 7. 可视化 - 玻璃脑 (使用重采样后的 1mm 图像)

    fig = plt.figure(figsize=(14, 5))
    display = plotting.plot_glass_brain(
        img_in_1mm,
        display_mode='lyrz',
        colorbar=True,
        cmap='cold_hot',
        threshold=0.01,
        vmax=v_max,
        plot_abs=False,
        figure=fig
    )
    display.title(f"{task_name}", size=26, color='white', x=0.01, y=0.98)
    cbar = getattr(display, '_cbar', None)
    if cbar:
        cbar.set_ticks([v_min, (v_min + v_max) / 2, v_max])
        cbar.ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        cbar.ax.tick_params(labelsize=28)
    fig.savefig(os.path.join(OUTPUT_DIR, f"HCP_{task_name}_smooth_glass_2.png"), dpi=300)

    plt.close(fig)

    '''
    # 8. 可视化 - 切片图 (背景使用 1mm 模版)
    fig = plt.figure(figsize=(12, 5))
    plotting.plot_stat_map(
        img_in_1mm,
        bg_img=template_img_1mm,  # 使用高清模版作为背景
        title=f"{task_name} Axial Slices",
        display_mode='z',
        cut_coords=8,
        colorbar=True,
        cmap='cold_hot',
        threshold='auto',
        figure=fig
    )
    fig.savefig(os.path.join(OUTPUT_DIR, f"HCP_{task_name}_slices_30.png"), dpi=300)
    plt.close(fig)
    '''

def main():
    # 1. 加载模型
    model = load_model()

    # 2. 获取数据列表
    files = get_task_files(HCP_DATA_ROOT, TARGET_TASK, NUM_SAMPLES_TO_AVERAGE)
    print(f"选取了 {len(files)} 个 {TARGET_TASK} 样本进行平均。")

    # 3. 计算组平均热力图
    target_idx = LABEL_MAP[TARGET_TASK]
    avg_heatmap = compute_group_attribution(model, files, target_idx)

    if avg_heatmap is not None:
        # 4. 可视化
        visualize_and_save(avg_heatmap, TARGET_TASK)
    else:
        print("未能计算热力图。")


if __name__ == "__main__":
    main()