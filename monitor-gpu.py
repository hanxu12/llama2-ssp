import subprocess
import time

def get_gpu_usages():
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'], capture_output=True, text=True)
        gpu_usages = result.stdout.strip().split('\n')
        return gpu_usages
    except subprocess.CalledProcessError as e:
        print("Failed to query GPU usage: ", e)
        return []

def monitor_gpu_usage(filename="gpu_usage_log.txt", interval=1, duration=60):
    end_time = time.time() + duration
    with open(filename, "w") as file:
        while time.time() < end_time:
            gpu_usages = get_gpu_usages()
            if gpu_usages:
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                for i, gpu_usage in enumerate(gpu_usages):
                    log_entry = f"{timestamp} - GPU {i} Usage: {gpu_usage}%\n"
                    print(log_entry, end='')
                    file.write(log_entry)
                file.flush()
            time.sleep(interval)
    print("Monitoring complete.")

if __name__ == "__main__":
    monitor_gpu_usage()
