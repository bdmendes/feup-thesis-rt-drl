import shutil
import subprocess
import os

# Define the directories
out_dir = "out"
results_dir = "results"

# Create the results directory if it doesn't exist
if not os.path.exists(results_dir):
    os.makedirs(results_dir)


# Function to clear a directory
def clear_directory(directory):
    if os.path.exists(directory):
        shutil.rmtree(directory)
    os.makedirs(directory)


# Function to run the cargo command
def run_cargo():
    command = ["cargo", "run", "--release"]
    process = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )

    # Read output line by line as it becomes available
    for line in iter(process.stdout.readline, ""):
        print(line.strip())

    # Ensure the process has finished
    process.stdout.close()
    process.wait()

    # Check if there was any error output
    stderr = process.stderr.read()
    if stderr:
        print("stderr:", stderr.strip())

    return process.returncode == 0


# Function to copy the output directory
def copy_out_directory(try_index):
    destination_dir = os.path.join(results_dir, f"out_{try_index}")
    shutil.copytree(out_dir, destination_dir)


# Main loop to execute the process 50 times
for try_index in range(2):
    clear_directory(out_dir)

    if run_cargo():
        copy_out_directory(try_index)
    else:
        print(f"Cargo run failed on try {try_index + 1}")
        break

print("Process completed.")
