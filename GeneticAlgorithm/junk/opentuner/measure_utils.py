import subprocess, time

def measure_runtime(binary):
    start = time.perf_counter()
    try:
        subprocess.run([binary], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except:
        return float("inf")
    end = time.perf_counter()
    return end - start
