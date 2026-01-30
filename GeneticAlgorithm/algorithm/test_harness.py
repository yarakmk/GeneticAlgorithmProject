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
BENCHMARK_SOURCE = "benchmarks/matmul.cpp" # You must create this file later!
# Compiler executable (verified to be GCC 15 on your system)
COMPILER_EXE = "g++-15" 
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
    Compiles the source file with the given flags. Includes macOS SDK path
    handling for g++-15 on Mac.
    """
    
    # Attempt to get the macOS SDK path needed for g++-15/Homebrew compatibility
    try:
        sdk = subprocess.check_output(
            ["xcrun", "--sdk", "macosx", "--show-sdk-path"]
        ).decode().strip()
        sdk_flags = ["-isysroot", sdk]
    except subprocess.CalledProcessError:
        print("Warning: Could not find macOS SDK path. Trying compilation without it.")
        sdk_flags = []
    
    cmd = [COMPILER_EXE] + sdk_flags + flags + [source_file, "-o", output]

    try:
        # Use subprocess.run for better error logging
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        # Check for empty output indicating success (no errors)
        if result.stderr:
            # If the compiler output warnings to stderr, we can optionally log them
            pass
        return True
    except subprocess.CalledProcessError as e:
        # Compilation failed: print the full command and error output for debugging
        print(f"\n--- Compilation Failed ---")
        print(f"Command: {' '.join(cmd)}")
        print(f"Error: {e.stderr.strip()}")
        print(f"--------------------------\n")
        return False


def run_program_and_time(executable="./a.out"):
    """
    Runs the compiled program once and returns the elapsed wall-clock time.
    """
    # Check if the executable exists
    if not os.path.exists(executable):
        return float("inf") # Return infinite time if binary is missing

    try:
        # We use time.perf_counter() for wall-clock time as it's the simplest 
        # reliable method in Python for measuring real-world execution time.
        start = time.perf_counter()
        
        # Run the executable, ensuring any benchmark output goes to /dev/null
        # so it doesn't skew timing or fill up logs.
        subprocess.run(
            [executable], 
            check=True,  # Raise error on non-zero exit code (crashes)
            capture_output=True, # Suppress stdout/stderr
            timeout=10 # Add a timeout guard (e.g., 10 seconds max)
        )
        
        end = time.perf_counter()
        return end - start

    except subprocess.CalledProcessError:
        # Program crashed or returned an error code
        return float("inf") 
    except subprocess.TimeoutExpired:
        # Program took too long
        return float("inf")


def get_fitness_score(chromosome, core_flag_list):
    """
    The main fitness function: compiles and runs the program N times, 
    returning the median time in seconds.
    """
    source_dir = os.path.dirname(BENCHMARK_SOURCE)
    executable = os.path.join(source_dir, f"matmul.out")
    flags = translate_chromosome_to_flags(chromosome, core_flag_list)

    # 1. Compile
    if not compile_program(BENCHMARK_SOURCE, flags, executable):
        # Compiler failed: very bad fitness score
        return float("inf")

    # 2. Run and Time N_RUNS times
    runtimes = []
    for _ in range(N_RUNS):
        # Get one run time
        run_time = run_program_and_time(executable)
        
        if run_time == float("inf"):
            # If a crash/error occurs during any run, discard the entire set
            os.remove(executable) # Clean up the failed binary
            return float("inf")

        runtimes.append(run_time)

    # 3. Clean up the compiled binary (optional, but good practice)
    os.remove(executable)
    
    # 4. Return the Median (MOST CRUCIAL STEP for stability)
    return statistics.median(runtimes)

def get_baseline_runtime():
    """
    Calculates the baseline performance using only the -O3 flag.
    Returns the median runtime (in seconds).
    """
    print("\n--- Establishing -O3 Baseline Performance ---")
    
    # 1. Get the directory of the source file (e.g., "benchmarks")
    source_dir = os.path.dirname(BENCHMARK_SOURCE) 
    
    # 2. Construct the full relative path for the executable with a unique name
    executable = os.path.join(source_dir, f"baseline_O3_{random.getrandbits(16)}.out")
    
    # 3. Define the flags (only -O3)
    flags = ["-O3"]
    
    # 4. Compile
    if not compile_program(BENCHMARK_SOURCE, flags, executable):
        print("ERROR: Baseline compilation failed.")
        return float("inf")

    # 5. Run and Time N_RUNS times
    runtimes = []
    for i in range(N_RUNS):
        run_time = run_program_and_time(executable)
        if run_time == float("inf"):
            print("ERROR: Baseline execution failed.")
            if os.path.exists(executable): os.remove(executable)
            return float("inf")
        runtimes.append(run_time)
        print(f"  -> Run {i+1:02d}/{N_RUNS}: {run_time:.4f}s")


    # 6. Clean up the compiled binary
    if os.path.exists(executable): os.remove(executable)
    
    # 7. Return the Median
    median_runtime = statistics.median(runtimes)
    print(f"Baseline Median Runtime (O3): {median_runtime:.4f}s")
    print("------------------------------------------------")
    return median_runtime