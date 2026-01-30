import random
import subprocess
import time
import os
import statistics

# --- CONFIGURATION ---
# The number of times to run the compiled program to gather a stable measurement.
# N=10 is the recommended minimum for robust statistics.
N_RUNS = 1

# The name of the benchmark file to compile
BENCHMARK_SOURCE = "benchmarks/matmul.cpp"

# Compiler executable – now using Clang/LLVM instead of Homebrew GCC
COMPILER_EXE = "clang++"
# --- END CONFIGURATION ---


def translate_chromosome_to_flags(chromosome, core_flag_list):
    """
    Translates the binary chromosome (list of 0s/1s) into a list of
    compiler flag strings (e.g., ['-funroll-loops', '-fno-inline-functions']).

    NOTE: The core_flag_list must contain only the 'base' flag names
          (e.g., 'funroll-loops', not '-fno-funroll-loops').
    """

    # 1. Start with the base optimization level
    flags = ["-O3"]

    # 2. Iterate through the chromosome and apply the binary decision
    for gene, flag_name in zip(chromosome, core_flag_list):
        # We assume the flag list contains names like "funroll-loops"
        if gene == 1:
            # Gene is ON: enable the flag
            flags.append(f"-f{flag_name}")
        else:
            # Gene is OFF: disable the flag
            flags.append(f"-fno-{flag_name}")

    return flags


def compile_program(source_file, flags, output="a.out"):
    """
    Compiles the source file with the given flags using Clang/LLVM.
    On macOS, clang++ already knows how to find the system SDK, so
    we do not need the -isysroot hack required by Homebrew GCC.
    """

    # No SDK detection needed for clang++
    cmd = [COMPILER_EXE] + flags + [source_file, "-o", output]

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        # Optionally log warnings from stderr if you want
        if result.stderr:
            # print("Compiler warnings:\n", result.stderr)
            pass
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n--- Compilation Failed ---")
        print(f"Command: {' '.join(cmd)}")
        print(f"Error: {e.stderr.strip()}")
        print(f"--------------------------\n")
        return False


def run_program_and_time(executable="./a.out"):
    """
    Runs the compiled program once and returns the elapsed wall-clock time.
    """
    if not os.path.exists(executable):
        return float("inf")  # Return infinite time if binary is missing

    try:
        start = time.perf_counter()

        subprocess.run(
            [executable],
            check=True,          # Raise error on non-zero exit code (crash)
            capture_output=True, # Suppress stdout/stderr
            timeout=10           # Guard against infinite loops
        )

        end = time.perf_counter()
        return end - start

    except subprocess.CalledProcessError:
        return float("inf")
    except subprocess.TimeoutExpired:
        return float("inf")


def get_fitness_score(chromosome, core_flag_list):
    """
    The main fitness function: compiles and runs the program N times,
    returning the median time in seconds.
    """
    source_dir = os.path.dirname(BENCHMARK_SOURCE)
    executable = os.path.join(source_dir, "matmul.out")
    flags = translate_chromosome_to_flags(chromosome, core_flag_list)

    # 1. Compile
    if not compile_program(BENCHMARK_SOURCE, flags, executable):
        return float("inf")

    # 2. Run and time N_RUNS times
    runtimes = []
    for _ in range(N_RUNS):
        run_time = run_program_and_time(executable)
        if run_time == float("inf"):
            if os.path.exists(executable):
                os.remove(executable)
            return float("inf")
        runtimes.append(run_time)

    # 3. Clean up
    if os.path.exists(executable):
        os.remove(executable)

    # 4. Return median runtime
    return statistics.median(runtimes)


def get_baseline_runtime():
    """
    Calculates the baseline performance using only the -O3 flag.
    Returns the median runtime (in seconds).
    """
    print("\n--- Establishing -O3 Baseline Performance (clang++) ---")

    source_dir = os.path.dirname(BENCHMARK_SOURCE)
    executable = os.path.join(source_dir, f"baseline_O3_{random.getrandbits(16)}.out")

    flags = ["-O3"]

    if not compile_program(BENCHMARK_SOURCE, flags, executable):
        print("ERROR: Baseline compilation failed.")
        return float("inf")

    runtimes = []
    for i in range(N_RUNS):
        run_time = run_program_and_time(executable)
        if run_time == float("inf"):
            print("ERROR: Baseline execution failed.")
            if os.path.exists(executable):
                os.remove(executable)
            return float("inf")
        runtimes.append(run_time)
        print(f"  -> Run {i+1:02d}/{N_RUNS}: {run_time:.4f}s")

    if os.path.exists(executable):
        os.remove(executable)

    median_runtime = statistics.median(runtimes)
    print(f"Baseline Median Runtime (O3, clang++): {median_runtime:.4f}s")
    print("------------------------------------------------")
    return median_runtime
