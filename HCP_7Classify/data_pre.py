import os
import csv
import numpy as np
import pandas as pd
import os
import glob
import nibabel as nib
import numpy as np
from scipy.ndimage import zoom
import matplotlib.pyplot as plt
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import logging



def process_single_file(nii_file_path):
    """
    处理单个 NIfTI 文件：加载、重采样、保存为 NPY

    Args:
        nii_file_path (str): 输入 NIfTI 文件的完整路径。
    Returns:
        tuple: (status, message_or_output_path)
               status: True 表示成功, False 表示失败
               message_or_output_path: 错误信息或成功处理的NPY文件路径
    """
    try:
        # 1. 构建输出路径 (使用传入的相对路径)
        base_filename = os.path.basename(nii_file_path)
        filename_no_ext, _ = os.path.splitext(base_filename)
        # 再次 splitext 以处理 .nii.gz
        if filename_no_ext.endswith('.nii'):
            filename_no_ext, _ = os.path.splitext(filename_no_ext)

        output_npy_path = os.path.join(OUTPUT_NPY_DIR, filename_no_ext +".npy")

        # 如果文件已存在，可以选择跳过
        if os.path.exists(output_npy_path):
            return True, f"Skipped (already exists): {output_npy_path}"

        # 2. 加载 NIFTI 数据
        img_nii = nib.load(nii_file_path)
        data = img_nii.get_fdata(caching='unchanged', dtype=np.float32)

        # 3. 维度检查与调整
        if data.ndim == 3:
            data = data[..., np.newaxis]
            # logger.warning(f"File {nii_file_path} is 3D, adding a time dimension. Shape: {data.shape}")
        elif data.ndim != 4:
            return False, f"Unsupported number of dimensions {data.ndim} for file {nii_file_path}"

        # 4. 转换维度顺序为 [T, D, H, W]
        data_t_first = data.transpose(3, 0, 1, 2)

        T = data_t_first.shape[0]
        orig_spatial_shape = data_t_first.shape[1:]

        # 5. 空间重采样
        resampled_data_t_first = np.zeros((T,) + TARGET_SPATIAL_SHAPE, dtype=np.float32)
        zoom_factors = [
            TARGET_SPATIAL_SHAPE[i] / orig_spatial_shape[i] for i in range(3)
        ]

        for t in range(T):
            frame = data_t_first[t, ...]
            resampled_frame = zoom(frame, zoom_factors, order=1, mode='nearest', prefilter=False)
            resampled_data_t_first[t, ...] = resampled_frame

        # 6. 保存为 NPY 文件
        np.save(output_npy_path, resampled_data_t_first)

        return True, output_npy_path

    except Exception as e:
        return False, f"Error processing {nii_file_path}: {str(e)}"


# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main(input_dir, output_npy_dir, max_workers=None):
    """
    主函数：批量处理NIFTI文件

    Args:
        input_dir (str): 输入目录路径
        output_npy_dir (str): 输出目录路径
        max_workers (int): 最大并行进程数
    """

    label_names = ['fear', 'loss', 'present-story', 'rh', 'relation', 'mental', '2bk-places']

    # 确保输出目录存在
    os.makedirs(output_npy_dir, exist_ok=True)

    # 收集所有NIfTI文件
    nii_files = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            base_filename = os.path.basename(file)
            filename_no_ext, _ = os.path.splitext(base_filename)
            label = filename_no_ext.split('_')[-1]
            if label in label_names and 'RL' in base_filename:
                full_path = os.path.join(root, file)
                nii_files.append(full_path)

    logger.info(f"开始处理 {len(nii_files)} 个文件，使用 {max_workers or '自动'} 个工作进程")

    # 使用进程池处理
    success_count = 0
    failure_count = 0
    skip_count = 0

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        future_to_file = {
            executor.submit(process_single_file, f): f
            for f in nii_files
        }

        # 处理结果
        for future in as_completed(future_to_file):
            file_path = future_to_file[future]
            try:
                status, result = future.result()
                if status:
                    if "Skipped" in result:
                        skip_count += 1
                        logger.info(result)
                    else:
                        success_count += 1
                        logger.info(f"成功处理: {file_path} -> {result}")
                else:
                    failure_count += 1
                    logger.error(result)
            except Exception as e:
                failure_count += 1
                logger.exception(f"处理 {file_path} 时发生未捕获异常: {str(e)}")

    # 打印总结报告
    logger.info("\n===== 处理完成 =====")
    logger.info(f"总文件数: {nii_files}")
    logger.info(f"已存在跳过: {skip_count}")
    logger.info(f"成功处理: {success_count}")
    logger.info(f"失败处理: {failure_count}")


if __name__ == "__main__":
    INPUT_DIR = "/home/amax/cacheNeuroScience/HCP_TMP/AllConCutAndLabel/"
    OUTPUT_NPY_DIR = "/data3/ysn/HCP_Seven_Tasks/"
    TARGET_SPATIAL_SHAPE = (80, 96, 80)  # D, H, W (与 zoom 的输入对应，nibabel加载后是 X,Y,Z)
    NUM_WORKERS = 6
    # 启动处理 (max_workers=None 自动设置进程数)
    main(input_dir=INPUT_DIR, output_npy_dir=OUTPUT_NPY_DIR, max_workers=NUM_WORKERS)