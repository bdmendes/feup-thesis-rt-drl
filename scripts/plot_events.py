import matplotlib.pyplot as plt


def collect_mode_changes(file):
    mode_changes = []
    with open(file) as f:
        for line in f:
            parts = line.split()
            mode_changes.append(int(parts[4][:-1]))
    return mode_changes


def collect_task_kills(file):
    task_kills = []
    with open(file) as f:
        for line in f:
            parts = line.split()
            task_kills.append(int(parts[7]))
    return task_kills


mode_changes_25 = collect_mode_changes("out/changes_kills_25.txt")
task_kills_25 = collect_task_kills("out/changes_kills_25.txt")

mode_changes_100 = collect_mode_changes("out/changes_kills_100.txt")
task_kills_100 = collect_task_kills("out/changes_kills_100.txt")

# plot mode changes for 25 and 100 tasks
plt.plot(mode_changes_25, label="Task set with 25 runnables")
plt.plot(mode_changes_100, label="Task set with 100 runnables")
plt.xlabel("Generation Tentative")
plt.ylabel("Mode Changes")
plt.legend()
plt.savefig("out/mode_changes.png")

# bar plot task kills for 25 and 100 tasks
plt.clf()
plt.plot(task_kills_25, label="Task set with 25 runnables")
plt.plot(task_kills_100, label="Task set with 100 runnables")
plt.xlabel("Generation Tentative")
plt.ylabel("Task Cancellations")
plt.legend()
plt.savefig("out/task_kills.png")
