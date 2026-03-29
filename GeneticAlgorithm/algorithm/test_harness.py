import random
import subprocess
import time
import os
import statistics

# --- CONFIGURATION ---
# Number of times to run the compiled program for a stable measurement
N_RUNS = 5

# Compiler executable
COMPILER_EXE = "g++-15"
# --- END CONFIGURATION ---


def translate_chromosome_to_flags(chromosome, core_flag_list):
    """
    Translates a binary chromosome into GCC flag strings.
    gene=1 → -fflag, gene=0 → -fno-flag
    """
    flags = ["-O3"]
    for gene, flag_name in zip(chromosome, core_flag_list):
        if gene == 1:
            flags.append(f"-f{flag_name}")
        else:
            flags.append(f"-fno-{flag_name}")
    return flags


def compile_program(source_file, flags, output="a.out", extra_sources=None, extra_includes=None, defines=None):
    """
    Compiles the source file with the given flags.
    Includes macOS SDK path handling for g++-15 on Mac.

    extra_sources  : additional .c/.cpp files to compile alongside (e.g. polybench.c)
    extra_includes : list of -I include paths (e.g. PolyBench utilities/)
    """
    try:
        sdk = subprocess.check_output(
            ["xcrun", "--sdk", "macosx", "--show-sdk-path"]
        ).decode().strip()
        sdk_flags = ["-isysroot", sdk]
    except subprocess.CalledProcessError:
        print("Warning: Could not find macOS SDK path.")
        sdk_flags = []

    include_flags = [f"-I{p}" for p in (extra_includes or [])]
    sources       = [source_file] + (extra_sources or [])

    define_flags = [f"-D{d}" for d in (defines or [])]
    cmd = [COMPILER_EXE] + sdk_flags + flags + include_flags + define_flags + sources +["-o", output]

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n--- Compilation Failed ---")
        print(f"Command: {' '.join(cmd)}")
        print(f"Error: {e.stderr.strip()}")
        print(f"--------------------------\n")
        return False


def run_program_and_time(executable="./a.out", polybench=False):
    """
    Runs the compiled program once and returns elapsed time in seconds.

    polybench=True  : reads time printed to stdout by PolyBench (-DPOLYBENCH_TIME)
    polybench=False : measures wall-clock time externally with perf_counter
    """
    if not os.path.exists(executable):
        return float("inf")

    try:
        if polybench:
            result = subprocess.run(
                [executable],
                check=True,
                capture_output=True,
                text=True,
                timeout=10
            )
            return float(result.stdout.strip())
        else:
            start = time.perf_counter()
            subprocess.run(
                [executable],
                check=True,
                capture_output=True,
                timeout=30
            )
            return time.perf_counter() - start

    except subprocess.CalledProcessError:
        return float("inf")
    except subprocess.TimeoutExpired:
        return float("inf")
    except ValueError:
        # stdout wasn't a valid float (polybench mode)
        return float("inf")


def get_fitness_score(chromosome, core_flag_list, benchmark_path,
                      polybench=False, extra_sources=None, extra_includes=None, 
                      defines=None):
    """
    Compiles and runs the benchmark N_RUNS times, returning the median time.

    benchmark_path  : path to the benchmark source file
    polybench       : True if benchmark prints its own time to stdout
    extra_sources   : additional source files needed (e.g. polybench.c)
    extra_includes  : additional include paths (e.g. PolyBench utilities/)
    """
    benchmark_name = os.path.splitext(os.path.basename(benchmark_path))[0]
    source_dir     = os.path.dirname(benchmark_path)
    executable     = os.path.join(source_dir, f"{benchmark_name}.out")
    flags          = translate_chromosome_to_flags(chromosome, core_flag_list)

    # 1. Compile
    if not compile_program(benchmark_path, flags, executable, extra_sources, extra_includes, defines):
        return float("inf")

    # 2. Run N_RUNS times
    runtimes = []
    for _ in range(N_RUNS):
        run_time = run_program_and_time(executable, polybench=polybench)
        if run_time == float("inf"):
            if os.path.exists(executable):
                os.remove(executable)
            return float("inf")
        runtimes.append(run_time)

    # 3. Clean up
    if os.path.exists(executable):
        os.remove(executable)

    # 4. Return median
    return statistics.median(runtimes)


def get_baseline_runtime(benchmark_path, polybench=False,
                         extra_sources=None, extra_includes=None, defines=None):
    """
    Calculates the baseline -O3 performance for a given benchmark.
    Returns the median runtime in seconds.

    benchmark_path  : path to the benchmark source file
    polybench       : True if benchmark prints its own time to stdout
    extra_sources   : additional source files (e.g. polybench.c)
    extra_includes  : additional include paths (e.g. PolyBench utilities/)
    """
    print(f"\n--- Establishing -O3 Baseline: {benchmark_path} ---")

    source_dir = os.path.dirname(benchmark_path)
    executable = os.path.join(source_dir, f"baseline_O3_{random.getrandbits(16)}.out")
    flags      = ["-O3"]

    if not compile_program(benchmark_path, flags, executable, extra_sources, extra_includes, defines):
        print("ERROR: Baseline compilation failed.")
        return float("inf")

    runtimes = []
    for i in range(N_RUNS):
        run_time = run_program_and_time(executable, polybench=polybench)
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
    print(f"Baseline Median ({benchmark_path}): {median_runtime:.4f}s")
    print("---------------------------------------------------")
    return median_runtime