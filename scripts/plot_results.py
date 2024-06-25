from genericpath import isfile
from os import listdir
import matplotlib.pyplot as plt

TOTAL_TRIALS = 50


def collect_data(lines):
    parts = lines[0].split(" ")
    reward = int(float(parts[2][:-1]))
    mode_changes = int(parts[7][:-1])
    task_kills = int(parts[15][:-1])
    task_starts = int(parts[18][:-1])
    return (reward, mode_changes, task_kills, task_starts)


placebo_data = []
best_data = []

for i in range(0, TOTAL_TRIALS):
    trial_data = []
    dir_path = f"results/out_{i}"
    files = [f"{dir_path}/{f}" for f in listdir(dir_path) if isfile(f"{dir_path}/{f}")]
    for file in files:
        with open(file, "r") as f:
            lines = f.readlines()
            data = collect_data(lines)
            if "placebo" in file:
                placebo_data.append(data)
            else:
                trial_data.append(data)
    # find trial with best reward
    best_data.append(max(trial_data, key=lambda x: x[0]))

for i in range(0, TOTAL_TRIALS):
    print(f"Trial {i} - Placebo: {placebo_data[i]}, Best: {best_data[i]}")


def plot_data(index: int, label: str):
    # remove outliers: trials where placebo vs best is more than 100% different
    placebo_data_filtered = []
    best_data_filtered = []
    for i in range(0, TOTAL_TRIALS):
        is_outlier = False
        for j in range(0, 4):
            if abs(placebo_data[i][j] / max(best_data[i][j], 1)) > 20:
                is_outlier = True
                break
        if not is_outlier:
            placebo_data_filtered.append(placebo_data[i])
            best_data_filtered.append(best_data[i])
        else:
            print(f"Trial {i} is an outlier")
    print(f"Filtered {TOTAL_TRIALS - len(placebo_data_filtered)} outliers\n")

    placebo_data_filtered = placebo_data_filtered[:25]
    best_data_filtered = best_data_filtered[:25]

    y_placebo = [x[index] for x in placebo_data_filtered]
    x_placebo = list(range(len(y_placebo)))

    y_best = [x[index] for x in best_data_filtered]
    x_best = list(range(len(y_best)))

    # Use bigger width for better visibility
    plt.figure(figsize=(10, 5))

    # Use bigger font
    plt.rcParams.update({"font.size": 12})

    # Clear the current figure
    plt.clf()

    # Plot lines for placebo data
    plt.plot(x_placebo, y_placebo, marker="o", label="Placebo")

    # Plot lines for best data
    plt.plot(x_best, y_best, marker="s", label="Reactive")

    # Add labels and legend
    plt.xlabel("Task set")
    plt.ylabel(label)  # Add y-label for better understanding
    plt.legend()

    # Show all trials on x-axis
    plt.xticks(range(len(x_placebo)), range(len(x_placebo)))

    # Save the plot to a file
    plt.savefig(f"results/{label}.png")


plot_data(0, "Reward")
plot_data(1, "Mode Changes")
plot_data(2, "Task Kills")
plot_data(3, "Task Starts")
