import matplotlib.pyplot as plt
import numpy as np

# 使用默认风格，手动调整参数以获得最纯净的科研图表外观
plt.style.use('default')

# --- 数据准备 (保持不变) ---
subjects = ['20', '50', '100', '200', '900']
x = np.arange(len(subjects))
width = 0.35

scratch_mean = np.array([0.344286, 0.749857, 0.826571, 0.905571, 0.953571])
scratch_std = np.array([0.246169, 0.119114, 0.071621, 0.040381, 0.023797])

finetuning_mean = np.array([0.777285714,0.855142857,0.905285714,0.930142857,0.962142857])
finetuning_std = np.array([0.068463407,0.066195238,0.063060062,0.028956453,0.013396872])

# --- 开始绘图 ---
fig, ax = plt.subplots(figsize=(10, 6))

# 1. 绘制柱状图 (带纹理和白色背景)
rects1 = ax.bar(x - width/2, scratch_mean, width,
                yerr=scratch_std,
                label='Training from scratch',
                capsize=4,
                color='white',
                edgecolor='tab:orange',
                hatch='///',
                linewidth=1.2,
                error_kw={'ecolor': 'black', 'elinewidth': 1.2})

rects2 = ax.bar(x + width/2, finetuning_mean, width,
                yerr=finetuning_std,
                label='Pre-trained with fine-tuning',
                capsize=4,
                color='white',
                edgecolor='tab:blue',
                hatch='\\\\\\',
                linewidth=1.2,
                error_kw={'ecolor': 'black', 'elinewidth': 1.2})

# --- 2. 核心修改：设置坐标轴刻度与网格 ---

# A. 彻底去掉网格线 (什么都不写，或者显式关闭)
ax.grid(False)

# B. 设置刻度样式 (模仿参考图风格)
# direction='in': 刻度线朝内
# top=True, right=True: 上边框和右边框也显示刻度
# length=4, width=1: 刻度线的大小
ax.tick_params(axis='both', which='major', direction='in',
               top=True, right=True, length=5, width=1, labelsize=12)

# C. 确保边框是封闭的黑框 (默认就是，但为了保险可以显式设置)
for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_linewidth(1)
    spine.set_color('black')

# --- 3. 标签、标题与范围 ---
ax.set_ylabel('Mean Classification Accuracy', fontsize=14)
ax.set_xlabel('Number of Subjects', fontsize=14)
ax.set_title('', fontsize=16) # 如果论文里不需要图内标题，可以注释掉

ax.set_xticks(x)
ax.set_xticklabels(subjects)

ax.set_ylim(0, 1.1)
ax.set_yticks(np.arange(0, 1.15, 0.1)) # 设置更加细致的Y轴刻度

# 图例设置 (去掉图例边框或者保留，参考图中是有边框的)
ax.legend(fontsize=12, loc='upper left', frameon=True, edgecolor='black', fancybox=False)

plt.tight_layout()

# --- 保存 ---
plt.savefig('HCP_accuracy_bar_chart.png', dpi=300)
# plt.show()