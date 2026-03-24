import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline
from matplotlib.lines import Line2D

# 此版本X轴使用Epoch * 被试数量，关注处理的总数据量,结果更好看

# --- 使用适合发表的图表样式 ---
plt.style.use('seaborn-v0_8-white')

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
    20: [12.3711, 1.9706, 0.6684, 0.3434, 0.1326, 0.2326, 0.1045, 0.1035, 0.2123, 0.2411, 0.3195, 0.2643, 0.1574, 0.1277, 0.0932, 0.0642, 0.0605, 0.0334, 0.0059, 0.0002, 0.0204, 0.1368, 0.2882, 0.3255, 0.2057, 0.4510, 0.1845, 0.1206, 0.0527, 0.0562],
    50: [6.7845, 0.9343, 0.3354, 0.4338, 0.2679, 0.2972, 0.2315, 0.3344, 0.1818, 0.2280, 0.1345, 0.0809, 0.2360, 0.3612, 0.1791, 0.1200, 0.1579, 0.4071, 0.1344, 0.0793, 0.1086, 0.0576, 0.1189, 0.2014, 0.1192, 0.2465, 0.1167, 0.0447, 0.0651, 0.1098],
    100: [
    3.9464, 0.8343, 0.4058, 0.4377, 0.4064, 0.3122, 0.2205, 0.1979, 0.2666, 0.1187,
    0.2510, 0.2664, 0.1242, 0.1181, 0.2159, 0.1491, 0.1087, 0.1402, 0.0856, 0.1612,
    0.0676, 0.0476, 0.1963, 0.0434, 0.0285, 0.1269, 0.0631, 0.0639, 0.0810, 0.0720
],
    200: [
    2.9101, 0.8186, 0.4583, 0.3397, 0.2663, 0.2502, 0.2081, 0.1572, 0.1530, 0.1476,
    0.1426, 0.0777, 0.1016, 0.0646, 0.0871, 0.0491, 0.0803, 0.0456, 0.0569, 0.0508,
    0.0512, 0.0263, 0.0450, 0.0276, 0.0293, 0.0288, 0.0372, 0.0260, 0.0190, 0.0237
],
    900: [
    1.4061, 0.4127, 0.1885, 0.1105, 0.0817, 0.0653, 0.0522, 0.0462, 0.0419, 0.0368,
    0.0340, 0.0268, 0.0280, 0.0230, 0.0248, 0.0198, 0.0190, 0.0181, 0.0164, 0.0169,
    0.0144, 0.0146, 0.0135, 0.0102, 0.0107, 0.0113, 0.0099, 0.0108, 0.0091, 0.0084
]
}

# --- 开始绘图 ---
fig, ax = plt.subplots(figsize=(7.8, 6))

# 定义颜色方案 (为5个被试量分别指定一种颜色)
colors = plt.get_cmap('tab10').colors[:len(subjects)]

# --- 循环绘制曲线 ---
for i, n_subjects in enumerate(subjects):
    # --- 新的X轴：Epoch * 被试数量 ---
    x_axis_raw = epochs_raw * n_subjects

    # 为平滑曲线生成更密集的x轴坐标
    x_axis_smooth = np.linspace(x_axis_raw.min(), x_axis_raw.max(), 300)

    # --- Scratch (虚线) ---
    loss_data_scratch = np.array(loss_scratch[n_subjects])
    spl_scratch = make_interp_spline(x_axis_raw, loss_data_scratch, k=3)
    loss_smooth_scratch = spl_scratch(x_axis_smooth)
    ax.plot(x_axis_smooth, loss_smooth_scratch, color=colors[i], linestyle='--',linewidth=4,
            label=f'N={n_subjects} (Scratch)')

    # --- Fine-tuning (实线) ---
    loss_data_finetuning = np.array(loss_finetuning[n_subjects])
    spl_finetuning = make_interp_spline(x_axis_raw, loss_data_finetuning, k=3)
    loss_smooth_finetuning = spl_finetuning(x_axis_smooth)
    ax.plot(x_axis_smooth, loss_smooth_finetuning, color=colors[i], linestyle='-',linewidth=4,
            label=f'N={n_subjects} (Fine-tuning)')

# --- 设置图表属性 ---
ax.set_xscale('log')
ax.set_yscale('linear')
ax.set_ylim(0,18)
ax.set_yticks(np.arange(0, 19, 6))

# 设置标题和坐标轴标签 (英文)
ax.set_title('', fontsize=22)
ax.set_xlabel('Cumulative Samples', fontsize=22)
ax.set_ylabel('Training Loss', fontsize=22)

# 调整刻度字号
ax.tick_params(axis='both', which='major', labelsize=21)
ax.tick_params(axis='x', pad=8)

ax.grid(False)
# --- 创建和自定义图例 ---

# 1. 创建代表“被试数量(颜色)”的图例句柄 (实线，但颜色不同)
color_legend_handles = [
    Line2D([0], [0], color=colors[i], lw=4, label=f'N={n}')
    for i, n in enumerate(subjects)
]

# 2. 创建代表“训练方式(线型)”的图例句柄 (灰色，但线型不同)
style_legend_handles = [
    Line2D([0], [0], color='gray', linestyle='--', lw=4, label='Scratch'),
    Line2D([0], [0], color='gray', linestyle='-', lw=4, label='Fine-tuning')
]

# 3. 将两组句柄合并 (可以在中间加个空句柄做分隔，也可以直接连起来)
# 这里直接合并，列表顺序即图例显示顺序
all_handles = color_legend_handles + style_legend_handles

# 4. 创建统一图例
ax.legend(handles=all_handles,
          title='',
          fontsize=20,
          ncol=1,       # 单列显示更清晰，或者设为2
          loc='upper right',
          frameon=False,
          fancybox=False,
          edgecolor='black',
          handlelength=1.8,
          handletextpad=0.3,  # 图标与文字间距
          labelspacing=0.6,  # 行间距
          borderpad=0.2
          )


for line in ax.get_lines():
    line.set_alpha(0.7)

# 调整布局
plt.tight_layout()

# 添加网格
# ax.grid(True, which="both", ls="--", linewidth=0.5)


# --- 显示和保存图表 ---
plt.savefig('hcp_loss_curves_ST.png', dpi=300)
# plt.show()