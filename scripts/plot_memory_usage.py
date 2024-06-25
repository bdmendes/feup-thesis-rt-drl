from genericpath import isfile
from os import listdir
import matplotlib.pyplot as plt


for i in [1, 3]:
    mem_data = []
    with open(f"out/memory_usage_{i}.txt", "r") as f:
        lines = f.readlines()
        for line in lines:
            parts = line.split(" ")
            mem_data.append(int(parts[0]) / 1000000)

    # plot memory usage
    plt.plot(mem_data, label=f"{i} hidden layers")
    plt.legend()
    plt.xlabel("Activation")
    plt.ylabel("Memory usage (MB)")

plt.savefig("out/memory_usage.png")
