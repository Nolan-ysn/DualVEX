import matplotlib.pyplot as plt
import numpy as np

plt.style.use('seaborn-v0_8-white')
#该代码是HCP七分类的微调结果准确率图，非motor帧分类

# --- 表格中的数据 ---
subjects = [20, 50, 100, 200, 900]

# 从头训练 (scratch) 的数据
scratch_mean = np.array([0.344286, 0.749857, 0.826571, 0.905571, 0.953571])
scratch_std = np.array([0.246169, 0.119114, 0.071621, 0.040381, 0.023797])
scratch_sem = np.array([0.093, 0.045, 0.027, 0.015, 0.009])

# 预训练后微调 (fine-tuning) 的数据
finetuning_mean = np.array([0.777285714,0.855142857,0.905285714,0.930142857,0.962142857])
finetuning_std = np.array([0.068463407,0.066195238,0.063060062,0.028956453,0.013396872])
finetuning_sem = np.array([0.026, 0.025, 0.024, 0.011, 0.005])

# LoRA的数据
lora_mean = np.array([0.7785,0.8433,0.8624,0.872,0.886])
lora_std = np.array([0.117,0.095,0.058,0.059,0.087])
lora_sem = np.array([0.044, 0.036, 0.022, 0.022, 0.033])


plt.figure(figsize=(8, 6))

# 绘制"从头训练"的折线
plt.plot(subjects, scratch_mean, 'D', label='Scratch', markersize=8, linestyle='-', linewidth=2.5,alpha=0.7)
plt.fill_between(subjects, scratch_mean - scratch_sem, scratch_mean + scratch_sem, alpha=0.15)

# 绘制"预训练微调"的折线
plt.plot(subjects, finetuning_mean, 'o', label='Full FT', markersize=8, linestyle='-', linewidth=2.5,alpha=0.7)
plt.fill_between(subjects, finetuning_mean - finetuning_sem, finetuning_mean + finetuning_sem, alpha=0.15)

# 绘制"冻结预训练参数"的折线
plt.plot(subjects, lora_mean, 's', label='LoRA', markersize=8, linestyle='-', linewidth=2.5,alpha=0.7)
plt.fill_between(subjects, lora_mean - lora_sem, lora_mean + lora_sem, alpha=0.15)

# --- 设置英文标题和标签 ---
plt.title('', fontsize=18)
plt.xlabel('Number of Subjects', fontsize=22)
plt.ylabel('Accuracy', fontsize=22)

# --- 坐标轴和图例 ---
# 将X轴设置为对数刻度，以更好地展示数据
plt.xscale('log')
plt.xticks(subjects, labels=subjects) # 确保X轴上所有的被试数量都作为刻度显示

plt.ylim(0, 1.0)
plt.xticks(subjects, labels=subjects, fontsize=22) # 设置X轴刻度字号
plt.yticks(np.arange(0, 1.1, 0.2),fontsize=22) # 设置Y轴刻度字号
ax = plt.gca()
ax.tick_params(axis='x', pad=8)
# 添加图例以区分线条
plt.legend(loc='lower right', frameon=False, fontsize=22)

# 确保布局紧凑
plt.tight_layout()

# --- 显示和保存图表 ---
plt.savefig('HCP_seven_tasks_accuracy_ST_4.png', dpi=300) # dpi=300 的分辨率适合发表

# plt.show()