import matplotlib.pyplot as plt
import numpy as np

# 设置风格，保持整洁
plt.style.use('seaborn-v0_8-white')

# --- 1. 数据录入 ---
subjects = [20, 50, 100, 200, 900]

# full ft
means_ft = np.array([0.409, 0.414, 0.431, 0.436, 0.444])
stds_ft = np.array([0.219, 0.240, 0.226, 0.220, 0.172])
sems_ft = np.array([0.083,0.091,0.086,0.083,0.065])

# Scratch
# 根据您提供的表格提取
means_scratch = np.array([0.242, 0.425, 0.424, 0.424, 0.406])
stds_scratch = np.array([0.127, 0.164, 0.170, 0.134, 0.146])
sems_scratch = np.array([0.086,0.115,0.076,0.088,0.100])

#lora
means_lora = np.array([0.430, 0.407, 0.369, 0.312, 0.2619])
stds_lora = np.array([0.125, 0.152, 0.073, 0.199, 0.219])
sems_lora= np.array([0.085,0.095,0.066,0.113,0.121])
# --- 2. 绘图设置 ---
fig, ax = plt.subplots(figsize=(8, 6))


# --- 3. 绘制折线和误差阴影 ---

#  绘制 Scratch
ax.plot(subjects, means_scratch, marker='D', linestyle='-', linewidth=2.5,
        markersize=8, label='Scratch',alpha=0.7)
ax.fill_between(subjects, means_scratch - sems_scratch, means_scratch + sems_scratch,
                 alpha=0.15, edgecolor=None)

#  绘制 Full FT
ax.plot(subjects, means_ft, marker='o', linestyle='-', linewidth=2.5,
        markersize=8, label='Full FT',alpha=0.7)
ax.fill_between(subjects, means_ft - sems_ft, means_ft + sems_ft,
                alpha=0.15, edgecolor=None)

#  绘制 LoRA
ax.plot(subjects, means_lora, marker='s', linestyle='-', linewidth=2.5,
        markersize=8, label='LoRA',alpha=0.7)
ax.fill_between(subjects, means_lora - sems_lora, means_lora + sems_lora,
                alpha=0.15, edgecolor=None)


# 绘制随机水平线
random_chance = 1.0 / 7.0
ax.axhline(y=random_chance, color='gray', linestyle='--', linewidth=2.5,
           alpha=0.8, zorder=0)
ax.text(x=900, y=random_chance + 0.015, s='Random Chance (14.3%)',
        color='gray', fontsize=18, fontweight='bold',
        ha='right', va='bottom')

# 设置 X 轴为对数坐标
ax.set_xscale('log')
ax.set_xticks(subjects)
ax.set_xticklabels(subjects)

# 设置 Y 轴范围
ax.set_ylim(0.0, 0.6)
ax.set_yticks(np.arange(0.0, 0.61, 0.2))

ax.tick_params(axis='both', which='major', labelsize=22)
ax.tick_params(axis='x', pad=8)

# --- 5. 标签和标题 ---
ax.set_xlabel('Number of Subjects', fontsize=23)
ax.set_ylabel('Accuracy', fontsize=23)

# 添加图例
ax.legend(loc='lower right', frameon=False, fontsize=18,
          bbox_to_anchor=(0.96, 0.00),
          borderaxespad=0.0,
          handletextpad=0.6)

# --- 6. 保存 ---
plt.tight_layout()
plt.savefig('IBC_test_comparison_3.png', dpi=300, bbox_inches='tight')
