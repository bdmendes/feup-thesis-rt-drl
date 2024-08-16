from matplotlib import pyplot as plt

# Initialize a dictionary to hold time data for each case
time_data_dict = {1: [], 3: []}

# Read and store the time data for each hidden layer configuration
for i in [1, 3]:
    with open(f"results/activation_times_{i}.txt", "r") as f:
        lines = f.readlines()
        for line in lines:
            parts = line.split(" ")
            time_data_dict[i].append(int(parts[0]))

plt.rcParams.update({"font.size": 16})

# Create the box plot
plt.figure(figsize=(10, 6))
plt.boxplot(
    [time_data_dict[1], time_data_dict[3]], labels=["1 hidden layer", "3 hidden layers"]
)

# Set labels and title
plt.xlabel("Number of Hidden Layers")
plt.ylabel("Time (μs)")

plt.tight_layout()

# Save the figure
plt.savefig("results/activation_times.png")
