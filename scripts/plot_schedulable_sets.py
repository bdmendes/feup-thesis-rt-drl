data_03 = [
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    0.999,
    0.999,
    0.997,
    0.987,
    0.99,
    0.982,
    0.982,
    0.975,
    0.976,
    0.973,
    0.955,
    0.944,
    0.916,
    0.918,
    0.907,
    0.872,
    0.856,
    0.81,
    0.792,
    0.756,
    0.73,
    0.722,
    0.711,
    0.683,
    0.648,
    0.651,
    0.616,
    0.589,
    0.548,
    0.504,
    0.468,
    0.464,
    0.419,
    0.388,
    0.381,
    0.344,
    0.339,
    0.316,
    0.302,
    0.313,
    0.3,
    0.253,
    0.244,
    0.233,
    0.204,
    0.191,
    0.167,
    0.136,
    0.17,
    0.13,
    0.124,
    0.121,
    0.118,
    0.098,
    0.072,
    0.064,
    0.055,
    0.06,
    0.041,
    0.052,
    0.046,
    0.034,
    0.039,
    0.038,
    0.022,
    0.022,
    0.016,
    0.021,
    0.02,
    0.018,
    0.023,
    0.011,
    0.004,
    0.006,
    0.012,
    0.008,
    0.004,
    0.006,
    0.006,
    0.007,
    0,
    0.001,
    0.002,
    0.001,
    0,
    0.001,
    0,
    0,
    0.001,
    0.001,
    0,
    0.001,
    0.002,
    0,
    0.001,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0.001,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
]

data_06 = [
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    0.999,
    0.998,
    1,
    0.999,
    0.997,
    0.996,
    0.997,
    0.991,
    0.995,
    0.987,
    0.978,
    0.982,
    0.965,
    0.958,
    0.951,
    0.95,
    0.937,
    0.909,
    0.898,
    0.889,
    0.872,
    0.864,
    0.834,
    0.819,
    0.779,
    0.773,
    0.746,
    0.71,
    0.694,
    0.673,
    0.643,
    0.614,
    0.591,
    0.563,
    0.56,
    0.545,
    0.49,
    0.508,
    0.441,
    0.42,
    0.409,
    0.383,
    0.355,
    0.339,
    0.301,
    0.278,
    0.244,
    0.24,
    0.22,
    0.235,
    0.203,
    0.196,
    0.172,
    0.155,
    0.135,
    0.111,
    0.116,
    0.113,
    0.099,
    0.111,
    0.084,
    0.07,
    0.059,
    0.059,
    0.048,
    0.051,
    0.046,
    0.041,
    0.032,
    0.038,
    0.037,
    0.028,
    0.022,
    0.015,
    0.015,
    0.011,
    0.013,
    0.007,
    0.013,
    0.013,
    0.008,
    0.012,
    0.007,
    0.003,
    0.005,
    0.005,
    0.004,
    0.004,
    0.004,
    0.007,
    0.005,
    0.001,
    0,
    0.004,
    0.001,
    0,
    0.002,
    0.001,
    0,
    0.001,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0.001,
    0.001,
    0,
    0,
    0,
    0,
    0,
    0.001,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
]

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

num_runnables = np.arange(1, 150)

plt.rcParams.update({"font.size": 24})
plt.tight_layout()
plt.figure(figsize=(10, 10))
sns.lineplot(x=num_runnables, y=data_03, marker="o")
plt.xlabel("number of runnables")
plt.ylabel("schedulable sets (%)")
plt.show()