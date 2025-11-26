

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import sem

# --------------------------------------------
# File paths
# --------------------------------------------
script_dir = Path(__file__).resolve().parent

NeuralR1       = script_dir / "FilesForAssignment2" / "NeuralResponses_S1.txt"
NeuralR2       = script_dir / "FilesForAssignment2" / "NeuralResponses_S2.txt"
CategoryLabel  = script_dir / "FilesForAssignment2" / "CategoryLabels.txt"
CategoryVector = script_dir / "FilesForAssignment2" / "CategoryVectors.txt"

# --------------------------------------------
# Load data
# --------------------------------------------
neural       = pd.read_csv(NeuralR1)
cat_vectors  = pd.read_csv(CategoryVector)                 # comma-separated
cat_labels = pd.read_csv(
    CategoryLabel,
    sep=" ",
    header=None,
    engine="python",
    quotechar='"',
    skiprows=1
)


animate_row = cat_labels[cat_labels.iloc[:, 1] == "animate"]
inanim_row  = cat_labels[cat_labels.iloc[:, 1] == "inanim"]

animate_col = int(animate_row.iloc[0, 0]) - 1
inanim_col  = int(inanim_row.iloc[0, 0]) - 1


animate_mask   = cat_vectors.iloc[:, animate_col] == 1
inanimate_mask = cat_vectors.iloc[:, inanim_col] == 1


image_means = neural.mean(axis=1)

animate_values   = image_means[animate_mask].values    # 44 values
inanimate_values = image_means[inanimate_mask].values  # 44 values

mean_values = [animate_values.mean(), inanimate_values.mean()]
sem_values  = [sem(animate_values), sem(inanimate_values)]

categories = ["Animate", "Inanimate"]

# Thin bars
bar_width = 0.2

# X positions closer together
x_pos = np.array([0, 0.25])  

plt.figure(figsize=(6,5))
plt.bar(
    x_pos,
    mean_values,
    yerr=sem_values,
    width=bar_width,
    capsize=5,
    color='royalblue',
    edgecolor='black',
    ecolor='red',
    linewidth=1.5
)

plt.axhline(0, color='black', linewidth=1)

plt.xlim(-0.2, 0.5)

plt.xticks(x_pos, categories)
plt.ylabel("Average voxel response")
plt.title("Average fMRI Response by Object Category")
plt.tight_layout()
# plt.show()


def paired_t_test(x, y):
  """
  Voer t test uit
  """
  x = np.asarray(x)
  y = np.asarray(y)
  
  difference = x - y
  
  # Mean and standard deviation of differences
  m = difference.mean()
  s = difference.std(ddof=1)  # use ddof=1 for sample standard deviation
  
  # Number of observations
  N = len(difference)
  
  # Compute t-value
  t_stat = m / (s / np.sqrt(N))
  
  # Degrees of freedom
  df = N - 1
  
  return t_stat, df

# Example usage with animate_values and inanimate_values arrays
t_val, df = paired_t_test(animate_values, inanimate_values)
print("Paired t-test result:")
print("t =", t_val)
print("df =", df)

"""
Plot 1B
"""


# Compute voxel-wise differences
animate_voxel_avg   = neural[animate_mask].mean(axis=0)  # 100 voxels
inanimate_voxel_avg = neural[inanimate_mask].mean(axis=0)  # 100 voxels
voxel_diff = animate_voxel_avg - inanimate_voxel_avg
voxel_diff_20 = voxel_diff[:20]

# X positions
x_pos = np.arange(20)  # 0 to 19

# Plot bars with no spacing
plt.figure(figsize=(10,5))
plt.bar(x_pos, voxel_diff_20, width=1.0, color='royalblue', edgecolor='black')
plt.axhline(0, color='black', linewidth=1)  # y=0 line
plt.ylabel("Response Amplitude")
plt.xlabel("Voxel")
plt.title("Animate minus Inanimate")

# X-axis ticks every 5 voxels
plt.xticks(ticks=[0,5,10,15,19], labels=['0','5','10','15','20'])
plt.tight_layout()
plt.show()