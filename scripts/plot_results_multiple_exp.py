from genericpath import isfile
from os import listdir
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

TOTAL_TRIALS = 10


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


def collect_hiperparams(line):
    parts = [part.split(":")[-1] for part in line.split(";")]
    hidden_sizes = len(eval(parts[0]))
    sample_batch_size = int(parts[1])
    activation_function = parts[2]
    return [hidden_sizes, sample_batch_size, activation_function]


placebo_data = []
best_data = []
best_hiperparams = []

for i in range(0, TOTAL_TRIALS):
    trial_data = []
    trial_hipers = []
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
                trial_hipers.append(collect_hiperparams(lines[0]))

    # find trial with best reward and its index
    best_trial = max(trial_data, key=lambda x: sum(a[0] for a in x))
    best_trial_index = trial_data.index(best_trial)

    best_data.append(best_trial)
    best_hiperparams.append(trial_hipers[best_trial_index])

for i in range(0, TOTAL_TRIALS):
    print(f"Trial {i} - Placebo: {placebo_data[i]}, Best: {best_data[i]}")


def plot_data(index: int, label: str):
    # Prepare data for seaborn
    def prepare_data(ranges, label):
        """Convert the list of lists into a DataFrame for seaborn"""
        data = []
        for i, trial in enumerate(ranges):
            for value in trial:
                data.append({"Trial": f"{i+1}", "Value": value, "Type": label})
        return pd.DataFrame(data)

    y_placebo = [[entry[index] for entry in trial] for trial in placebo_data]
    y_best = [[entry[index] for entry in trial] for trial in best_data]

    # Create DataFrames
    placebo_df = prepare_data(y_placebo, "Placebo")
    best_df = prepare_data(y_best, "With Agent")

    # Combine DataFrames
    combined_df = pd.concat([placebo_df, best_df])

    # Create the box plot
    plt.figure(figsize=(16, 10))
    sns.boxplot(
        x="Trial",
        y="Value",
        hue="Type",
        data=combined_df,
        palette={"Placebo": "red", "With Agent": "green"},
    )

    # Add titles and labels
    plt.xlabel("Task Set")
    plt.ylabel(label)
    plt.legend(title="Scenario")

    # Display the plot
    plt.tight_layout()

    # make font bigger
    plt.rc("axes", labelsize=18)
    plt.rc("xtick", labelsize=18)
    plt.rc("ytick", labelsize=18)
    plt.rc("legend", fontsize=18)
    plt.rc("legend", title_fontsize=18)

    # Save the plot to a file
    plt.savefig(f"results/{label}.png")


def plot_best_hiperparameters():
    from collections import Counter

    # Extract the values for each hyperparameter
    hidden_sizes = [hp[0] for hp in best_hiperparams]
    batch_sizes = [hp[1] for hp in best_hiperparams]
    activation_functions = [hp[2] for hp in best_hiperparams]

    # Count the occurrences of each value
    hidden_sizes_count = Counter(hidden_sizes)
    batch_sizes_count = Counter(batch_sizes)
    activation_functions_count = Counter(activation_functions)

    def save_bar_chart(data, title, xlabel, ylabel, filename):
        plt.figure(figsize=(6, 6))
        plt.rcParams.update({"font.size": 30})
        plt.bar(data.keys(), data.values())
        # plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.xticks(list(data.keys()))
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()

    # Save each graph to disk
    save_bar_chart(
        hidden_sizes_count,
        "Hidden Sizes",
        "Hidden Size",
        "Count",
        "results/hidden_sizes_count.png",
    )
    save_bar_chart(
        batch_sizes_count,
        "Batch Sizes",
        "Batch Size",
        "Count",
        "results/batch_sizes_count.png",
    )
    save_bar_chart(
        activation_functions_count,
        "Activation Functions",
        "Activation Function",
        "Count",
        "results/activation_functions_count.png",
    )


plot_best_hiperparameters()
plot_data(0, "Reward")
plot_data(1, "Mode Changes")
plot_data(2, "Task Kills")
plot_data(3, "Task Starts")
