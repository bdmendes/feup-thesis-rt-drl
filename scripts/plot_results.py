from genericpath import isfile
from os import listdir
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

TOTAL_TRIALS = 50
placebo_data_150 = []
best_data_150 = []
placebo_data_250 = []
best_data_250 = []


def read():
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

    for path in ["results_150", "results_250"]:
        for i in range(0, TOTAL_TRIALS):
            trial_data = []
            dir_path = f"{path}/out_{i}"
            files = [
                f"{dir_path}/{f}"
                for f in listdir(dir_path)
                if isfile(f"{dir_path}/{f}")
            ]
            for file in files:
                with open(file, "r") as f:
                    lines = f.readlines()
                    data = collect_data(lines[1:])
                    placebo_data = (
                        placebo_data_150 if "150" in dir_path else placebo_data_250
                    )
                    if "placebo" in file:
                        placebo_data.append(data)
                    else:
                        trial_data.append(data)

            # find trial with best reward and its index
            best_trial = max(trial_data, key=lambda x: sum(a[0] for a in x))
            best_trial_index = trial_data.index(best_trial)

            best_data = best_data_150 if "150" in dir_path else best_data_250
            best_data.append(best_trial)

    for i in range(0, TOTAL_TRIALS):
        print(f"Trial {i} - Placebo: {placebo_data[i]}, Best: {best_data[i]}")


def plot_data(index: int, label: str):
    # Prepare data for seaborn
    def prepare_data(ranges, label, runnable_count):
        """Convert the list of lists into a DataFrame for seaborn"""
        data = []
        for i, trial in enumerate(ranges):
            for value in trial:
                data.append(
                    {
                        "Trial": f"{i+1}",
                        "Value": value,
                        "Type": label,
                        "Runnables": runnable_count,
                    }
                )
        return pd.DataFrame(data)

    # Data extraction
    y_placebo_150 = [[entry[index] for entry in trial] for trial in placebo_data_150]
    y_best_150 = [[entry[index] for entry in trial] for trial in best_data_150]
    y_placebo_250 = [[entry[index] for entry in trial] for trial in placebo_data_250]
    y_best_250 = [[entry[index] for entry in trial] for trial in best_data_250]

    # Create DataFrames
    placebo_df_150 = prepare_data(y_placebo_150, "AMC+", 150)
    best_df_150 = prepare_data(y_best_150, "Enhanced", 150)
    placebo_df_250 = prepare_data(y_placebo_250, "AMC+", 250)
    best_df_250 = prepare_data(y_best_250, "Enhanced", 250)

    # Combine DataFrames
    combined_df = pd.concat([placebo_df_150, best_df_150, placebo_df_250, best_df_250])

    # Create the box plot
    plt.figure(figsize=(10, 10))
    sns.boxplot(
        x="Runnables",  # x-axis shows the number of runnables (150 and 250)
        y="Value",  # y-axis shows the distribution of values
        hue="Type",  # hue creates the subcolumns for AMC+ and Enhanced
        data=combined_df,
        palette={"AMC+": "red", "Enhanced": "green"},
    )

    # Add titles and labels
    plt.xlabel("Number of Runnables")
    plt.ylabel(label)
    plt.legend(title="Schedule")

    # Display the plot
    plt.tight_layout()

    # Make font bigger
    plt.rc("axes", labelsize=18)
    plt.rc("xtick", labelsize=18)
    plt.rc("ytick", labelsize=18)
    plt.rc("legend", fontsize=18)
    plt.rc("legend", title_fontsize=18)

    # Save the plot to a file
    plt.savefig(f"results/{label}_comparison.png")


read()
plot_data(0, "Reward")
plot_data(1, "Mode Changes")
plot_data(2, "Task Kills")
plot_data(3, "Task Starts")
