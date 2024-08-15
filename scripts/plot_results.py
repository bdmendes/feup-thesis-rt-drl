from genericpath import isfile
from os import listdir
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

TOTAL_TRIALS = 50


def collect_data(lines):
    res = []
    for line in lines:
        parts = line.split(" ")
        reward = int(float(parts[2][:-1]))
        mode_changes = int(parts[7][:-1])
        task_kills = int(parts[15][:-1])
        task_starts = int(parts[18][:-1])
        res.append([reward, mode_changes, task_kills, task_starts])
    return res


placebo_data = []
best_data = []

for i in range(0, TOTAL_TRIALS):
    trial_data = []
    dir_path = f"results_multiple/out_{i}"
    files = [f"{dir_path}/{f}" for f in listdir(dir_path) if isfile(f"{dir_path}/{f}")]
    for file in files:
        with open(file, "r") as f:
            lines = f.readlines()
            data = collect_data(lines[1:])
            if "placebo" in file:
                placebo_data.append(data)
            else:
                trial_data.append(data)

    # find trial with best reward and its index
    best_trial = max(trial_data, key=lambda x: sum(a[0] for a in x))
    best_trial_index = trial_data.index(best_trial)

    best_data.append(best_trial)

for i in range(0, TOTAL_TRIALS):
    print(f"Trial {i} - Placebo: {placebo_data[i]}, Best: {best_data[i]}")


def plot_data(index: int, label: str):
    # Prepare data for seaborn
    def prepare_data(ranges, label):
        """Convert the list of lists into a DataFrame for seaborn"""
        data = []
        for i, trial in enumerate(ranges):
            # Only take the first value in each list
            value = trial[0] if trial else None
            data.append({"Trial": f"{i+1}", "Value": value, "Type": label})
        return pd.DataFrame(data)

    y_placebo = [[entry[index] for entry in trial] for trial in placebo_data]
    y_best = [[entry[index] for entry in trial] for trial in best_data]

    # Create DataFrames
    placebo_df = prepare_data(y_placebo, "Placebo")
    best_df = prepare_data(y_best, "With Agent")

    # Combine DataFrames
    combined_df = pd.concat([placebo_df, best_df])

    # Create the line plot
    plt.figure(figsize=(16, 10))
    sns.lineplot(
        x="Trial",
        y="Value",
        hue="Type",
        data=combined_df,
        palette={"Placebo": "red", "With Agent": "green"},
        marker="o",  # Add markers to distinguish the points
    )

    # Add titles and labels
    plt.xlabel("Task Set")
    plt.ylabel(label)
    plt.legend(title="Scenario")

    # Display the plot
    plt.tight_layout()

    # Make font bigger
    plt.rc("axes", labelsize=18)
    plt.rc("xtick", labelsize=18)
    plt.rc("ytick", labelsize=18)
    plt.rc("legend", fontsize=18)
    plt.rc("legend", title_fontsize=18)

    # Save the plot to a file
    plt.savefig(f"results/{label}.png")


plot_data(0, "Reward")
plot_data(1, "Mode Changes")
plot_data(2, "Task Kills")
plot_data(3, "Task Starts")
