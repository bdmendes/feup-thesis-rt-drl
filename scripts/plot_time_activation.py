from matplotlib import pyplot as plt

for i in [1, 3]:
    time_data = []
    with open(f"out/activation_times_{i}.txt", "r") as f:
        lines = f.readlines()
        for line in lines:
            parts = line.split(" ")
            time_data.append(int(parts[0]))

    # plot memory usage
    plt.plot(time_data, label=f"{i} hidden layers")
    plt.legend()
    plt.xlabel("Activation")
    plt.ylabel("Time (μs)")

plt.savefig("out/activation_times.png")
