import os
import glob
import numpy as np
import pandas as pd
import nibabel as nib
from scipy.ndimage import zoom
from concurrent.futures import ProcessPoolExecutor, as_completed
import logging

# ================= 配置区域 =================

# 输入NII文件和Event文件所在的根目录
INPUT_ROOT_DIR = "/home/amax/NeuroCyber/OpenNeuroPreproc/ds000244/"

# 输出NPY文件的目录
OUTPUT_NPY_DIR = "/data3/ysn/IBC_for_test/"

# 目标空间分辨率 (D, H, W)
TARGET_SPATIAL_SHAPE = (80, 96, 80)

# 时间窗口长度 (帧数) - 强制取8帧
WINDOW_SIZE = 8

# IBC数据的TR (重复时间)
TR = 2.0

# Block合并阈值 (秒)
# 用于将 Motor/Gambling 等任务中密集的 trial 合并为一个 Block 起始点
BLOCK_MERGE_THRESHOLD = 16.0

# 每个扫描文件最大提取 Block 数
MAX_BLOCKS_PER_SCAN = 6

# 任务与子任务标签的映射配置
TASK_CONFIG = {
    'task-HcpEmotion': ('face', 'EMOTION'),
    'task-HcpGambling': ('punishment', 'GAMBLING'),
    'task-HcpLanguage': ('story', 'LANGUAGE'),
    'task-HcpMotor': ('right_hand', 'MOTOR'),
    'task-HcpRelational': ('relational', 'RELATIONAL'),
    'task-HcpSocial': ('mental', 'SOCIAL'),
    'task-HcpWm': ('2back_place', 'WM')
}

# 并行进程数
NUM_WORKERS = 8

# ===========================================

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def get_block_onsets(event_file_path, target_trial_type):
    """
    读取 .tsv Event文件，提取起始时间并合并Block
    """
    try:
        # 强制使用 tab 分隔符读取 tsv
        df = pd.read_csv(event_file_path, sep='\t')

        # 检查列名，兼容可能的格式差异
        if 'trial_type' not in df.columns:
            # 尝试查找类似的列名
            possible_cols = [c for c in df.columns if 'trial' in c or 'condition' in c]
            if possible_cols:
                logger.warning(
                    f"未找到 'trial_type'，尝试使用 '{possible_cols[0]}' 于 {os.path.basename(event_file_path)}")
                df.rename(columns={possible_cols[0]: 'trial_type'}, inplace=True)
            else:
                logger.error(f"无法识别 Label 列于 {event_file_path}")
                return []

        if isinstance(target_trial_type, list):
            # 如果是列表 (如 WM)，使用 isin 筛选
            target_rows = df[df['trial_type'].isin(target_trial_type)]
        else:
            # 如果是字符串，使用 == 筛选
            target_rows = df[df['trial_type'] == target_trial_type]

        if target_rows.empty:
            return []

        raw_onsets = sorted(target_rows['onset'].tolist())

        # Block检测逻辑：合并相邻过近的 trial
        block_onsets = []
        if raw_onsets:
            block_onsets.append(raw_onsets[0])

            for i in range(1, len(raw_onsets)):
                current_onset = raw_onsets[i]
                prev_onset = raw_onsets[i - 1]

                if (current_onset - prev_onset) > BLOCK_MERGE_THRESHOLD:
                    block_onsets.append(current_onset)

        return block_onsets

    except Exception as e:
        logger.error(f"读取Event文件失败 {event_file_path}: {e}")
        return []


def process_single_subject_task(nii_path, event_path, task_key):
    """
    处理单个被试任务
    """
    try:
        target_trial_type, save_label_name = TASK_CONFIG[task_key]

        # 1. 获取Block起始时间
        onsets_seconds = get_block_onsets(event_path, target_trial_type)

        # 数量限制：如果提取的 Block 超过 6 个，只取前 6 个
        if len(onsets_seconds) > MAX_BLOCKS_PER_SCAN:
            onsets_seconds = onsets_seconds[:MAX_BLOCKS_PER_SCAN]
        if not onsets_seconds:
            return False, f"Skip: 未找到标签 '{target_trial_type}' -> {os.path.basename(event_path)}"

        # 2. 加载 NIfTI 数据
        img = nib.load(nii_path)
        data = img.get_fdata(dtype=np.float32)

        if data.ndim != 4:
            return False, f"Error: 数据维度错误 {data.shape}"

        # 转换为 [T, D, H, W] (假设原始是 X, Y, Z, T)
        data_t_first = data.transpose(3, 0, 1, 2)

        total_frames = data_t_first.shape[0]
        orig_spatial_shape = data_t_first.shape[1:]

        # 3. 计算缩放因子
        zoom_factors = [
            TARGET_SPATIAL_SHAPE[i] / orig_spatial_shape[i] for i in range(3)
        ]

        saved_count = 0
        # 获取文件名主体，去掉后缀
        raw_name = os.path.basename(nii_path).replace('.volreg.nii.gz', '').replace('.nii.gz', '')
        base_name = raw_name.replace(task_key, '').replace('_bold', '').replace('__', '_').strip('_')
        # 4. 遍历每个Block进行切片
        for onset_sec in onsets_seconds:
            # 秒 -> 帧
            start_idx = int(round(onset_sec / TR))

            # 强制取 8 帧，无论 duration 是多少
            end_idx = start_idx + WINDOW_SIZE

            # 边界检查：如果文件末尾不够8帧，则跳过
            if end_idx > total_frames:
                logger.warning(f"窗口越界跳过: {base_name} Start:{start_idx} End:{end_idx} Total:{total_frames}")
                continue

            # 提取时间窗口
            raw_clip = data_t_first[start_idx:end_idx, ...]

            # 空间重采样容器
            resampled_clip = np.zeros((WINDOW_SIZE,) + TARGET_SPATIAL_SHAPE, dtype=np.float32)

            # 逐帧重采样 (2D/3D zoom)
            for t in range(WINDOW_SIZE):
                resampled_clip[t, ...] = zoom(raw_clip[t, ...], zoom_factors, order=1, mode='nearest', prefilter=False)

            # 5. 保存 NPY
            # 格式: sub-01_ses-03_task-HcpEmotion_acq-ap_EMOTION_onset12.npy
            output_filename = f"{base_name}_{save_label_name}_onset{start_idx}.npy"
            output_path = os.path.join(OUTPUT_NPY_DIR, output_filename)

            np.save(output_path, resampled_clip)
            saved_count += 1

        return True, f"提取 {saved_count} 个Block: {raw_name}"

    except Exception as e:
        return False, f"处理异常 {os.path.basename(nii_path)}: {str(e)}"


def find_matched_files(root_dir):
    """
    根据新的命名规则匹配文件
    NII:   ..._bold.volreg.nii.gz
    Event: ..._events.tsv
    """
    tasks = []

    logger.info(f"正在 {root_dir} 中搜索文件...")

    # 递归遍历
    for root, _, files in os.walk(root_dir):
        # 筛选符合后缀的NII文件
        nii_files = [f for f in files if f.endswith('bold.volreg.nii.gz')]

        for nii_file in nii_files:
            # 检查是否是7个目标任务之一
            matched_task_key = None
            for task_key in TASK_CONFIG.keys():
                if task_key in nii_file:
                    matched_task_key = task_key
                    break

            if matched_task_key:
                nii_path = os.path.join(root, nii_file)

                # 构建对应的 Event 文件名
                # 规则：把 _bold.volreg.nii.gz 替换为 _events.tsv
                event_filename = nii_file.replace('_bold.volreg.nii.gz', '_events.tsv')
                event_path = os.path.join(root, event_filename)

                if os.path.exists(event_path):
                    tasks.append((nii_path, event_path, matched_task_key))
                else:
                    logger.warning(f"缺失Event文件: {event_filename} (for {nii_file})")

    return tasks


def main():
    os.makedirs(OUTPUT_NPY_DIR, exist_ok=True)

    task_list = find_matched_files(INPUT_ROOT_DIR)

    if not task_list:
        logger.error("未找到任何匹配的任务文件，请检查路径和文件名格式。")
        return

    logger.info(f"共找到 {len(task_list)} 对文件，开始处理...")

    success_cnt = 0
    fail_cnt = 0

    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = {executor.submit(process_single_subject_task, t[0], t[1], t[2]): t for t in task_list}

        # 使用tqdm显示进度条
        from tqdm import tqdm
        for future in tqdm(as_completed(futures), total=len(task_list), desc="Processing"):
            status, msg = future.result()
            if status:
                success_cnt += 1
                logger.info(msg)
            else:
                fail_cnt += 1
                logger.error(msg)

    logger.info(f"全部完成。成功: {success_cnt}, 失败: {fail_cnt}")


if __name__ == "__main__":
    main()