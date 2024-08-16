from matplotlib import pyplot as plt

# Initialize a dictionary to hold time data for each case
time_data_dict = {1: [], 3: []}

# Read and store the time data for each hidden layer configuration
for i in [1, 3]:
    with open(f"results/activation_times_{i}.txt", "r") as f:
        lines = f.readlines()
        for line in lines:
            parts = line.split(" ")
            time_data_dict[i].append(float(parts[0]) / 1000)

plt.rcParams.update({"font.size": 24})

# Create the box plot
plt.figure(figsize=(10, 10))
plt.boxplot(
    [time_data_dict[1], time_data_dict[3]], labels=["1 hidden layer", "3 hidden layers"]
)

# Set labels and title
plt.xlabel("number of hidden layers")
plt.ylabel("time (ms)")

plt.tight_layout()

# Save the figure
plt.savefig("results/activation_times.png")
