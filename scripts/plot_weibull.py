weibull1_file = "out/weibull_sample1.txt"
weibull_file = "out/weibull_sample.txt"


def load_samples(file):
    with open(file, "r") as f:
        return [int(float(x) / 1000) for x in f.readlines()]


weibull1 = load_samples(weibull1_file)
weibull = load_samples(weibull_file)

import matplotlib.pyplot as plt

# bar plot of weibull1
plt.hist(weibull1, label="weibull1")
plt.legend(loc="upper right")
plt.title("Weibull1")
plt.xlabel("Time (s)")
plt.ylabel("Frequency")
plt.show()

# bar plot of weibull
plt.hist(weibull, label="weibull")
plt.legend(loc="upper right")
plt.title("Weibull")
plt.xlabel("Time (s)")
plt.ylabel("Frequency")
plt.show()
