import os
import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from nilearn.image import resample_to_img

from model import MAE_FMRI_Classifier

# 导入 grad-cam 库
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


os.environ["CUDA_VISIBLE_DEVICES"] = "4"


# 定义模型包装器(Adapter / Wrapper)
class ModelWrapper(nn.Module):
    def __init__(self, model, time_steps, in_chans):
        super(ModelWrapper, self).__init__()
        self.model = model
        self.time_steps = time_steps
        self.in_chans = in_chans

    def forward(self, x_5d):
        # x_5d 的形状是 grad-cam 库匹配的 [B, C*T, D, H, W]
        # 现在需要把它变回我们真实模型匹配的 [B, C, T, D, H, W]

        # 获取空间维度
        b, _, d, h, w = x_5d.shape

        # Reshape 回 6D
        x_6d = x_5d.reshape(b, self.in_chans, self.time_steps, d, h, w)

        # 用正确的 6D 张量调用真实模型
        return self.model(x_6d)

print('model is loading...')
original_model = MAE_FMRI_Classifier( time_steps=16, spatial_dim=(80, 96, 80), patch_size=(8, 8, 8),
                 in_chans=1, embed_dim=1024, depth=6, num_heads=8, mlp_ratio=4.0,
                 num_classes=7, pretrained_path=None)

val_dataset_path = "/home/ysn/PycharmProjects/MAE/HCP_7Classify/result/validation_files.npy"
model_path = "/home/ysn/PycharmProjects/MAE/HCP_7Classify/result/HCP_7Class_MAE_900_fine-tuning_20250921_213719/models/best_model_epoch_19_acc_0.9544.pth"


state_dict = torch.load(model_path, map_location='cpu')
original_model.load_state_dict(state_dict, strict=False)

print('load model successfully!')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
original_model.to(device)
original_model.eval()

# --- 创建包装后的模型实例 ---
print("创建包装模型...")
# 将真实模型放入包装器中
model = ModelWrapper(original_model, time_steps=16, in_chans=1)
model.to(device)
model.eval()

# --- Grad-CAM 配置 ---

# 1. 定义目标层：对于ViT，通常选择最后一个Transformer块
# 根据model.py，路径是 model.encoder.vit3d.blocks[-1]
target_layers = [model.model.encoder.vit3d.blocks[-1]]

# 2. 定义 reshape_transform：这是适配ViT的关键
# ViT的输出是 [batch, num_patches, embed_dim]。Grad-CAM需要类似 [batch, channels, H, W] 的格式
# 把 [B, L, E] 变形为 [B, E, D_grid, H_grid, W_grid]
patch_size = (8, 8, 8)
spatial_dim = (80, 96, 80)
grid_size = (spatial_dim[0] // patch_size[0],
             spatial_dim[1] // patch_size[1],
             spatial_dim[2] // patch_size[2])

def reshape_transform_3d(tensor):
    """
    将 [B, L, E] 的张量重塑为 [B, E, D_grid, H_grid, W_grid]
    """
    # tensor shape: [batch_size, num_patches, embedding_dim]
    result = tensor.transpose(1, 2) # -> [B, E, L]
    result = result.reshape(result.size(0),
                            result.size(1),
                            grid_size[0],
                            grid_size[1],
                            grid_size[2])
    return result

# 3. 实例化 GradCAM
cam = GradCAM(model=model,
              target_layers=target_layers,
              reshape_transform=reshape_transform_3d,
              )

val_dataset = np.load(val_dataset_path).tolist()
print(f"数据集大小：{len(val_dataset)}")

save_dir = "/data3/ysn/GradCAM_HCP/"
os.makedirs(save_dir, exist_ok=True)

task_name_to_label_idx = {
    'EMOTION': 0,
    'GAMBLING': 1,
    'LANGUAGE': 2,
    'MOTOR': 3,
    'RELATIONAL': 4,
    'SOCIAL': 5,
    'WM':6 }

EXPECTED_TIME_STEPS = 16
CROPPED_SHAPE = spatial_dim #(80, 96, 80)

ref_2mm_path = "/home/amax/cacheNeuroScience/HCP_TMP/AllConCutAndLabel/100206_EMOTION_LR-0_fear.nii"
template_1mm_path = "/home/amax/abin/auto_backup.linux_ubuntu_16_64/MNI152_2009_template.nii.gz" #(193, 229, 193)


# 1. 定义空间维度和加载参考模板
try:
    print("正在加载参考模板...")
    ref_2mm_template = nib.load(ref_2mm_path)
    template_1mm = nib.load(template_1mm_path)

    ref_shape = ref_2mm_template.shape
    if len(ref_shape) == 4:  # 如果是4D fMRI数据
        print(f"检测到4D参考模板，形状为: {ref_shape}")
        original_spatial_shape = ref_shape[:3]
    elif len(ref_shape) == 3:  # 如果是3D解剖数据
        print(f"检测到3D参考模板，形状为: {ref_shape}")
        original_spatial_shape = ref_shape
    else:
        raise ValueError(f"参考模板的维度不正确（既不是3D也不是4D），形状为: {ref_shape}")

    print(f"提取出的原始空间形状为: {original_spatial_shape}")
    print("模板加载成功。")

except FileNotFoundError as e:
    print(f"错误: 找不到模板文件。请检查路径: {e}")
    exit()


# 2. 根据提取出的空间形状，动计算裁剪坐标
start_x = (original_spatial_shape[0] - CROPPED_SHAPE[0]) // 2
end_x = start_x + CROPPED_SHAPE[0]
start_y = (original_spatial_shape[1] - CROPPED_SHAPE[1]) // 2
end_y = start_y + CROPPED_SHAPE[1]
start_z = (original_spatial_shape[2] - CROPPED_SHAPE[2]) // 2
end_z = start_z + CROPPED_SHAPE[2]


for path in val_dataset[:100]:
    print(f"正在处理: {path}")
    d_full = np.load(path)
    d = d_full[0:EXPECTED_TIME_STEPS,:,:,:]
    # 从文件名中提取标签
    try:
        task_name_str = os.path.basename(path).split('_')[1]  # 例如 'WM'
        target_class_index = task_name_to_label_idx[task_name_str]

    except (IndexError, KeyError) as e:
        print(f"警告：无法提取标签 {path}. 跳过. 错误: {e}")
        continue

    # 原始d的形状是 (Time, H, W, D)，先转换为(1, Time, H, W, D)匹配包装器，再通过包装器增加chan维度
    input_tensor = torch.FloatTensor(d).unsqueeze(0) # Batch, Time, H, W, D

    targets = [ClassifierOutputTarget(target_class_index)]

    # --- 生成 Grad-CAM ---
    # grayscale_cam 的形状是 [batch, D_grid, H_grid, W_grid]
    with autocast():
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)

    # 消除batch维度
    cam_map = grayscale_cam[0, :]  # CAM库自己会上采样到原图像尺寸


    low_res_cam_img = nib.Nifti1Image(cam_map, affine=ref_2mm_template.affine)
    cam_in_2mm_space = resample_to_img(source_img=low_res_cam_img,
                                       target_img=ref_2mm_template,
                                       interpolation='linear')

    #  重采样到 1mm 空间
    resampled_cam_img = resample_to_img(source_img=cam_in_2mm_space,
                                        target_img=template_1mm,
                                        interpolation='linear')

    # 保存为NIFTI图像
    final_cam_data = resampled_cam_img.get_fdata().astype(np.float32)
    final_nifti_img = nib.Nifti1Image(final_cam_data, affine=resampled_cam_img.affine)
    save_filename = os.path.basename(path)[:-4] + f'_CAM_class_{target_class_index}.nii.gz'
    full_save_path = os.path.join(save_dir, save_filename)
    resampled_cam_img.to_filename(full_save_path)
    print(f"成功保存Grad-CAM图到: {full_save_path}")
