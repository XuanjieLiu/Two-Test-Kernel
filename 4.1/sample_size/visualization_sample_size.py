# import matplotlib.pyplot as plt
# import pandas as pd

# # Load saved CSV
# df_plot = pd.read_csv("./mmd_power_sample_size.csv")

# # Plotting
# plt.figure(figsize=(8, 6))
# plt.rcParams['font.family'] = 'Times New Roman'
# plt.rcParams['xtick.labelsize'] = 20  # 放大X轴刻度字体
# plt.rcParams['ytick.labelsize'] = 20 # 放大Y轴刻度字体

# scenarios = df_plot['scenario'].unique()
# for scenario in scenarios:
#     subset = df_plot[df_plot['scenario'] == scenario]
#     plt.plot(subset['sample_size'], subset['power'], marker='^', label=scenario, markersize=5, linewidth=2)


# plt.xlabel('Sample Size (per distribution)', fontsize=20)
# plt.ylabel('Test Power', fontsize=25)
# plt.legend(title="Settings", fontsize=20, title_fontsize=20)
# plt.grid(True, linestyle='--', alpha=0.5,axis='both', linewidth=2)
# plt.tight_layout()
# plt_path = "./mmd_power_sample_size_plot.pdf"
# plt.savefig(plt_path,dpi=300,bbox_inches='tight')


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.interpolate import make_interp_spline

# Load saved CSV
df_plot = pd.read_csv("./mmd_power_sample_size.csv")

# Plotting setup
plt.figure(figsize=(8, 6))
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['xtick.labelsize'] = 20
plt.rcParams['ytick.labelsize'] = 20

scenarios = df_plot['scenario'].unique()
colors = plt.cm.tab10.colors  # 色彩方案

for idx, scenario in enumerate(scenarios):
    subset = df_plot[df_plot['scenario'] == scenario].sort_values('sample_size')
    x = subset['sample_size'].values
    y = subset['power'].values

    # Spline interpolation for smoothing
    if len(x) >= 4:  # 至少4个点才能拟合三次样条
        x_smooth = np.linspace(x.min(), x.max(), 300)
        spline = make_interp_spline(x, y, k=3)
        y_smooth = spline(x_smooth)
        plt.plot(x_smooth, y_smooth, label=scenario, linewidth=2, color=colors[idx % len(colors)])
    else:
        plt.plot(x, y, marker='^', label=scenario, markersize=5, linewidth=2, color=colors[idx % len(colors)])

# Labels and legend
plt.xlabel('Sample Size (per distribution)', fontsize=20)
plt.ylabel('Test Power', fontsize=25)
plt.legend(title="Settings", fontsize=20, title_fontsize=20)
plt.grid(True, linestyle='--', alpha=0.5, axis='both', linewidth=2)
plt.tight_layout()

# Save
plt_path = "./mmd_power_sample_size_plot.pdf"
plt.savefig(plt_path, dpi=300, bbox_inches='tight')
