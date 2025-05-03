import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 设置风格
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
})

# 读取数据
df = pd.read_csv("./mmd_power_kernel.csv")

# 创建画布
plt.figure(figsize=(10, 5))
ax = sns.barplot(data=df, x='kernel', y='power', hue='scenario', edgecolor='black', linewidth=0.8)

# 去除坐标轴标题
plt.xlabel("")
plt.ylabel("")

# 美化
plt.ylim(0, 1.05)
plt.xticks(rotation=30)
plt.grid(axis='y', linestyle='--', linewidth=0.5)
sns.despine()

# 图例外置
plt.legend(title="Settings", bbox_to_anchor=(1.01, 1), loc='upper left', borderaxespad=0.,fontsize=20,title_fontsize=20)

# 紧凑布局并保存
plt.tight_layout()
plt.savefig("./mmd_power_kernel.pdf", dpi=300, bbox_inches='tight')
