import matplotlib.pyplot as plt
import numpy as np

# 设置风格，保持整洁
plt.style.use('seaborn-v0_8-white')

# --- 1. 数据录入 (根据你提供的表格图片) ---
subjects = [20, 50, 100, 200, 900]
# Mean Accuracy
means = np.array([0.409, 0.414, 0.431, 0.436, 0.444])
# Standard Deviation
stds = np.array([0.119, 0.140, 0.126, 0.120, 0.072])

# --- 2. 绘图设置 ---
fig, ax = plt.subplots(figsize=(8, 6)) # 尺寸可以根据大图E区域调整

# 颜色设置 (使用橙色，呼应子图D中的 Fine-tuning 线条)
line_color = '#2ca02c'
fill_color = '#2ca02c'

# --- 3. 绘制折线和误差阴影 ---
# 绘制主线 (均值)
ax.plot(subjects, means, marker='o', linestyle='-', linewidth=4,
        markersize=10, color=line_color, label='Generalization to IBC')

# 绘制误差范围 (均值 ± 标准差)
# alpha 控制透明度
ax.fill_between(subjects, means - stds, means + stds,
                color=fill_color, alpha=0.2, edgecolor=None)

random_chance = 1.0 / 7.0
ax.axhline(y=random_chance, color='gray', linestyle='--', linewidth=3.5,
           alpha=0.8, label='Random Chance', zorder=0)

# --- 4. 坐标轴美化 ---
# 设置 X 轴为对数坐标，这样 20, 50, 100... 的分布更均匀，与子图 D 保持一致
ax.set_xscale('log')
ax.set_xticks(subjects)
ax.set_xticklabels(subjects) # 强制显示具体数字

# 设置 Y 轴范围
ax.set_ylim(0.0, 0.6)
ax.set_yticks(np.arange(0.0, 0.61, 0.2))

ax.tick_params(axis='both', which='major', labelsize=22)
ax.tick_params(axis='x', pad=8)

# --- 5. 标签和标题 ---
ax.set_xlabel('Number of Subjects', fontsize=24)
ax.set_ylabel('Accuracy', fontsize=24)
# ax.set_title('Generalization Scaling (IBC Dataset)', fontsize=14) # 如果大图有标注E，这里可以不要标题

# 去掉右边和上边的边框 (Spines)，让图更清爽
#ax.spines['right'].set_visible(False)
#ax.spines['top'].set_visible(False)

# 添加图例 (可选)
ax.legend(loc='lower right', frameon=False, fontsize=21)

# --- 6. 保存 ---
plt.tight_layout()
plt.savefig('IBC_test_2.png', dpi=300, bbox_inches='tight')
plt.show()