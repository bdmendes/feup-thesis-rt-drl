import psutil
import subprocess
import time


def start_process_and_monitor_memory(command, check_interval, duration, output_file):
    # Start the process
    process = subprocess.Popen(command, shell=True)
    pid = process.pid
    print(f"Started process with PID: {pid}")

    with open(output_file, "w") as file:
        file.write(f"Started process with PID: {pid}\n")
        file.write("Time (s)\tMemory Usage (KB)\n")

        # Monitor the memory usage
        start_time = time.time()
        try:
            while time.time() - start_time < duration:
                if psutil.pid_exists(pid):
                    process_info = psutil.Process(pid)
                    memory_info = process_info.memory_info()
                    elapsed_time = time.time() - start_time
                    memory_usage_kb = memory_info.rss / 1024
                    log_line = f"{elapsed_time:.2f}\t{memory_usage_kb:.2f}\n"
                    print(
                        f"Time: {elapsed_time:.2f}s | Memory Usage: {memory_usage_kb:.2f} KB"
                    )
                    file.write(log_line)
                else:
                    print(f"Process with PID {pid} has terminated.")
                    break
                time.sleep(check_interval)
        except psutil.NoSuchProcess:
            print(f"Process with PID {pid} has terminated.")


if __name__ == "__main__":
    command = "cargo run --release"
    # clear out directory
    subprocess.run("rm -rf out", shell=True)
    # create out directory
    subprocess.run("mkdir out", shell=True)
    start_process_and_monitor_memory(
        command, check_interval=0.1, duration=120, output_file="out/memory_usage.txt"
    )
