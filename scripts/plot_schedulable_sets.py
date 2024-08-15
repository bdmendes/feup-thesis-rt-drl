data = [
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    0.998,
    1.0,
    0.996,
    0.992,
    0.982,
    0.972,
    0.928,
    0.922,
    0.85,
    0.782,
    0.748,
    0.664,
    0.62,
    0.522,
    0.458,
    0.386,
    0.316,
    0.258,
    0.174,
    0.168,
    0.082,
    0.05,
    0.04,
    0.042,
    0.018,
    0.014,
    0.004,
    0.002,
    0.002,
    0.0,
    0.002,
    0.002,
    0.0,
]
data = [x * 100 for x in data]
data = data[5:]
data = data[:-5]

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

num_runnables = np.arange(60, 360, step=10)

plt.rcParams.update({"font.size": 18})
plt.rcParams["axes.linewidth"] = 1.5
plt.rcParams["lines.linewidth"] = 1.5
plt.figure(figsize=(10, 6))
sns.lineplot(x=num_runnables, y=data, marker="o")
plt.xlabel("number of runnables")
plt.ylabel("schedulable sets (%)")
plt.tight_layout()
plt.savefig("results/schedulable_sets.pdf")
