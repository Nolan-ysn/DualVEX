import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline

# --- 使用适合发表的图表样式 ---
plt.style.use('seaborn-v0_8-whitegrid')

# --- 数据准备 (与之前相同) ---
epochs_raw = np.array(list(range(1, 31)))
subjects = [20, 50, 100, 200, 900]
loss_scratch = {
    20: [32.6717, 8.2340, 5.2188, 4.2736, 2.7276, 1.7088, 0.7393, 0.1799, 0.0169, 0.0034, 0.0033, 0.0004, 0.0003, 0.0003, 0.0002, 0.0002, 0.0002, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001],
    50: [17.2161, 4.1449, 2.8201, 1.4759, 1.4433, 0.9082, 0.4871, 0.1286, 0.2818, 0.2115, 0.2022, 0.1026, 0.2001, 0.1401, 0.2700, 0.1810, 0.1150, 0.1920, 0.0907, 0.1009, 0.0500, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
    100: [10.9325, 2.5283, 1.0390, 0.2880, 0.2979, 0.3236, 0.0874, 0.3162, 0.1913, 0.4289, 0.0555, 0.1389, 0.0987, 0.0987, 0.1407, 0.1744, 0.0633, 0.0221, 0.1774, 0.0359, 0.0698, 0.0591, 0.0329, 0.1317, 0.0197, 0.0120, 0.0221, 0.0261, 0.0288, 0.0164],
    200: [6.0414, 1.1120, 0.4788, 0.3182, 0.2896, 0.1695, 0.2284, 0.1642, 0.1446, 0.1716, 0.1012, 0.1019, 0.1067, 0.1130, 0.0678, 0.0810, 0.0460, 0.0683, 0.0380, 0.0349, 0.0340, 0.0423, 0.0208, 0.0301, 0.0195, 0.0250, 0.0106, 0.0300, 0.0078, 0.0106],
    900: [2.5245, 0.5277, 0.2240, 0.1202, 0.0765, 0.0581, 0.0509, 0.0396, 0.0378, 0.0316, 0.0273, 0.0265, 0.0256, 0.0227, 0.0211, 0.0186, 0.0173, 0.0177, 0.0155, 0.0150, 0.0151, 0.0127, 0.0132, 0.0119, 0.0108, 0.0097, 0.0107, 0.0093, 0.0089, 0.0078]
}
loss_finetuning = {
    20: [13.6201, 5.1768, 3.6271, 3.4267, 2.9219, 1.6575, 1.2523, 0.9735, 0.7874, 0.8727, 0.4488, 0.2612, 0.2406, 0.2562, 0.0861, 0.4348, 0.3440, 0.5090, 0.3091, 0.0874, 0.1059, 0.0382, 0.0288, 0.1530, 0.1313, 0.1023, 0.0711, 0.0427, 0.0381, 0.0636],
    50: [9.0305, 3.9173, 2.8412, 1.9924, 1.1714, 0.5323, 0.4205, 0.2707, 0.2552, 0.2799, 0.2066, 0.1981, 0.2983, 0.2502, 0.1923, 0.1655, 0.1949, 0.2050, 0.1690, 0.1809, 0.1230, 0.2021, 0.2188, 0.1478, 0.1189, 0.1301, 0.0692, 0.0703, 0.1482, 0.1399],
    100: [5.5899, 2.3995, 1.0780, 0.5186, 0.3973, 0.2544, 0.2713, 0.2081, 0.1848, 0.2791, 0.1249, 0.1735, 0.1401, 0.1617, 0.1731, 0.1070, 0.0933, 0.1513, 0.0660, 0.0685, 0.1122, 0.0721, 0.0709, 0.1019, 0.0601, 0.0508, 0.0969, 0.0327, 0.0482, 0.0330],
    200: [4.4521, 1.2251, 0.5382, 0.3534, 0.2977, 0.2085, 0.1811, 0.1605, 0.1327, 0.1156, 0.0844, 0.1015, 0.0562, 0.0810, 0.0777, 0.0470, 0.0455, 0.0347, 0.0386, 0.0391, 0.0274, 0.0265, 0.0278, 0.0410, 0.0143, 0.0364, 0.0211, 0.0207, 0.0223, 0.0223],
    900: [1.7036, 0.4508, 0.2103, 0.1137, 0.0755, 0.0545, 0.0469, 0.0362, 0.0313, 0.0280, 0.0233, 0.0222, 0.0206, 0.0192, 0.0164, 0.0159, 0.0162, 0.0130, 0.0127, 0.0114, 0.0131, 0.0103, 0.0107, 0.0103, 0.0085, 0.0084, 0.0080, 0.0066, 0.0077, 0.0068]
}

# --- 开始绘图 ---
fig, ax = plt.subplots(figsize=(12, 8))

# 定义颜色方案 (为5个被试量分别指定一种颜色)
colors = plt.get_cmap('tab10').colors[:len(subjects)]

# --- 循环绘制曲线 ---
for i, n_subjects in enumerate(subjects):
    # --- Scratch (虚线) ---
    loss_data_scratch = np.array(loss_scratch[n_subjects])
    # 样条插值平滑
    epochs_smooth = np.linspace(epochs_raw.min(), epochs_raw.max(), 300)
    spl_scratch = make_interp_spline(epochs_raw, loss_data_scratch, k=3)
    loss_smooth_scratch = spl_scratch(epochs_smooth)
    # 绘制
    ax.plot(epochs_smooth, loss_smooth_scratch, color=colors[i], linestyle='--',
            label=f'N={n_subjects} (Scratch)')

    # --- Fine-tuning (实线) ---
    loss_data_finetuning = np.array(loss_finetuning[n_subjects])
    # 样条插值平滑
    spl_finetuning = make_interp_spline(epochs_raw, loss_data_finetuning, k=3)
    loss_smooth_finetuning = spl_finetuning(epochs_smooth)
    # 绘制
    ax.plot(epochs_smooth, loss_smooth_finetuning, color=colors[i], linestyle='-',
            label=f'N={n_subjects} (Fine-tuning)')

# --- 设置图表属性 ---
# 设置Y轴为对称对数刻度 (symlog)
# linthresh=1.0 表示在[-1.0, 1.0]区间内为线性刻度，之外为对数刻度
ax.set_yscale('symlog', linthresh=1.0)
ax.set_ylim(bottom=0) # 确保Y轴从0开始

# 设置标题和坐标轴标签 (英文)
ax.set_title('Training Loss Convergence Comparison', fontsize=18)
ax.set_xlabel('Epoch', fontsize=14)
ax.set_ylabel('Training Loss (Symlog Scale)', fontsize=14)

# 调整刻度字号
ax.tick_params(axis='both', which='major', labelsize=12)

# --- 创建和自定义图例 ---

handles, labels = ax.get_legend_handles_labels()
# 重新排序图例，使得相同颜色（被试量）的线条相邻
order = [i for j in range(len(subjects)) for i in [j, j + len(subjects)]]
# 将图例放置在图内右上角 ('upper right')
ax.legend([handles[idx] for idx in order], [labels[idx] for idx in order],
          title='Subjects (N) & Training', fontsize=12, ncol=2,
          loc='upper right')

# 调整布局
plt.tight_layout()

# 添加网格
ax.grid(True, which="both", ls="--", linewidth=0.5)


# --- 显示和保存图表 ---
plt.savefig('hcp_loss_curves.png', dpi=300)
plt.show()