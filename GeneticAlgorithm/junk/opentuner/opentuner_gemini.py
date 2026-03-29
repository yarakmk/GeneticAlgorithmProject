import opentuner
from opentuner import ConfigurationManipulator
from opentuner import EnumParameter
from opentuner import MeasurementInterface
from opentuner import Result
import subprocess
import time
import statistics
import json
import os
import warnings
import logging

from sqlalchemy.exc import SAWarning
warnings.filterwarnings('ignore', category=SAWarning)

def get_sdk_path():
    """
    Retrieves the macOS SDK path using xcrun.
    This is often required for g++-15 on macOS to find standard headers.
    """
    try:
        result = subprocess.run(["xcrun", "--show-sdk-path"], capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except Exception:
        return None
    
def load_flags_from_file(filename):
    """
    Reads flags and filters out known problematic flags that cause
    conflicts or require specific values in GCC 15.
    """
    cleaned_flags = []
    # THE BLACKLIST: Specifically targeting your recent errors
    BLACKLIST = [
    # # Math conflicts
    # "associative-math",
    # "reciprocal-math",
    # "unsafe-math-optimizations",
    # "fast-math",
    # # Header/Standard Library breakers (causes postypes.h errors)
    # "exceptions",
    # "threadsafe-statics",
    # "rtti",
    # "short-enums", # BREAKS ABI: makes enums smaller than int, library mismatch
    # "builtin", # BREAKS HEADERS: standard library relies on builtins
    # "common",
    # "non-call-exceptions",
    # "use-cxa-atexit",

    # # Windows/Platform specific
    # "dllexport", "dllimport", "pe-aligned-commons",
    # # Requires parameters (=N)
    # "tree-parallelize-loops",
    # "min-function-alignment",
    # # Hardware/Version specific causing unrecognized option errors
    # "fuse-ops-with-volatile-access",
    # "speculatively-call-stored-functions",
    # "dep-fusion",
    # "lto-toplevel-asm-heuristics",
    # "live-patching",
    # "var-tracking",
    # "caller-saves",
    # "delayed-branch"
    ]

    if not os.path.exists(filename):
        print(f"Error: {filename} not found.")
        return ["unroll-loops", "tree-vectorize"]

    print(f"Loading and filtering flags from: {os.path.abspath(filename)}")
    with open(filename, 'r') as f:
        for line in f:
            flag = line.strip().lower()
            if not flag or flag.startswith('#') or flag.startswith('-O'):
                continue
            
            # Clean the flag name for checking
            core_name = flag[2:] if flag.startswith('-f') else flag
            
            # CHECK 1: Is it in the blacklist?
            if any(forbidden in core_name for forbidden in BLACKLIST):
                continue
                
            # CHECK 2: Does it contain an equal sign? (Needs a number)
            if '=' in core_name:
                continue
                
            cleaned_flags.append(core_name)
            
    return list(set(cleaned_flags))

# The full list of flags you want OpenTuner to explore
FULL_FLAG_LIST = load_flags_from_file("CFSCA_flags.txt")
MACOS_SDK_PATH = get_sdk_path()

class GCCFlagsTuner(MeasurementInterface):
    def __init__(self, *args, **kwargs):
        self.benchmark = kwargs.pop('benchmark')
        self.iteration = 0
        super(GCCFlagsTuner, self).__init__(*args, **kwargs)

    def manipulator(self):
        manipulator = ConfigurationManipulator()
        for flag in FULL_FLAG_LIST:
            manipulator.add_parameter(EnumParameter(flag, [True, False]))
        return manipulator

    def execute_benchmark(self, flags):
        """Helper to handle compilation and execution for any set of flags."""
        os.makedirs("./tmp", exist_ok=True)
        bin_path = "./tmp/tmp_bin"
        
        # Build command: Base g++-15 + provided flags
        compile_cmd = ["g++-15"] + flags
        if MACOS_SDK_PATH:
            compile_cmd += ["-isysroot", MACOS_SDK_PATH]
        compile_cmd += [self.benchmark, "-o", bin_path]

        try:
            # Clean up old binary to ensure fresh results
            if os.path.exists(bin_path):
                os.remove(bin_path)

            proc = subprocess.run(compile_cmd, capture_output=True, timeout=30, text=True)
            if proc.returncode != 0:
                return float('inf')
            
            runtimes = []
            for _ in range(3):
                start = time.perf_counter()
                subprocess.run([bin_path], check=True, capture_output=True, timeout=20)
                runtimes.append(time.perf_counter() - start)
            
            return statistics.median(runtimes)
        except Exception:
            return float('inf')

    def run(self, desired_result, input, limit):
        self.iteration += 1
        cfg = desired_result.configuration.data
        
        # --- ENFORCE DEPENDENCIES ---

        # 1. STRONG DEPENDENCIES: If Parent is OFF, Child MUST be OFF
        strong_deps = {
            'loop-unroll-and-jam': ['forward-propagate'],
            'unroll-loops': ['peel-loops'],
            'modulo-sched': ['modulo-sched-allow-regmoves'],
            'schedule-insns': ['sched-pressure'],
            'ipa-cp': ['ipa-vrp']
        }
        for parent, children in strong_deps.items():
            if not cfg.get(parent, False):
                for child in children:
                    cfg[child] = False

        # 2. WEAK DEPENDENCIES: If Parent is ON, Child MUST be ON
        weak_deps = {
            'auto-profile': ['peel-loops', 'gcse-after-reload', 'tree-loop-distribute-patterns'],
            'schedule-insns': [
                'sched-interblock', 'sched-spec-load', 'sched-group-heuristic', 
                'sched-critical-path-heuristic', 'sched-spec-insn-heuristic', 
                'sched-dep-count-heuristic', 'sched-rank-heuristic'
            ]
        }
        for parent, children in weak_deps.items():
            if cfg.get(parent, False):
                for child in children:
                    cfg[child] = True

        # 3. SYNERGISTIC: If either is ON, both MUST be ON (or both OFF)
        # Usually, this means if the tuner picks one, we force the other to match.
        if cfg.get('gcse-sm', False) or cfg.get('gcse-lm', False):
            cfg['gcse-sm'] = True
            cfg['gcse-lm'] = True


        # Build command from GA suggestions
        active_flags = ["-O3"]
        for flag in FULL_FLAG_LIST:
            if cfg.get(flag, False):
                active_flags.append(f"-f{flag}")
            else:
                active_flags.append(f"-fno-{flag}")
        
        res_time = self.execute_benchmark(active_flags)
        
        # Update progress
        if res_time < self.best_time:
            self.best_time = res_time
            status = "⭐ NEW BEST!"
        else:
            status = ""

        print(f"[{self.iteration:03d}] Current: {res_time:.4f}s | Best: {self.best_time:.4f}s {status}")
        return Result(time=res_time)

    def save_final_config(self, configuration):
        data = configuration.data if hasattr(configuration, 'data') else configuration
        best_flags = [f"-f{flag}" if data.get(flag) else f"-fno-{flag}" for flag in FULL_FLAG_LIST]
        os.makedirs("config", exist_ok=True)
        with open("config/best_flags.json", "w") as f:
            json.dump(best_flags, f, indent=4)
        print(f"\n[Done] Saved best flags to config/best_flags.json")

def run_ot_search(benchmark_path, timeout_seconds):
    from opentuner.api import TuningRunManager
    arg_parser = opentuner.default_argparser()
    args = arg_parser.parse_args(["--stop-after", str(timeout_seconds)])
    
    interface = GCCFlagsTuner(args, benchmark=benchmark_path)

    # --- INITIAL BASELINE CALCULATION ---
    print("-" * 50)
    print(f"STEP 1: Calculating Baseline (-O3 only)...")
    baseline_time = interface.execute_benchmark(["-O3"])
    
    if baseline_time == float('inf'):
        print("CRITICAL ERROR: Baseline -O3 failed to compile or run!")
        return

    interface.best_time = baseline_time 
    print(f"BASELINE RESULT: {baseline_time:.4f}s")
    print("-" * 50)
    # ------------------------------------

    api = TuningRunManager(interface, args)
    
    print(f"Starting search on {len(FULL_FLAG_LIST)} valid flags...")
    if MACOS_SDK_PATH:
        print(f"Using SDK path: {MACOS_SDK_PATH}")
    
    start_wall_clock = time.time()
    try:
        desired_result = api.get_next_desired_result()
        while desired_result is not None:
            if (time.time() - start_wall_clock) > timeout_seconds:
                break
            result = interface.run(desired_result, 0, 0)
            api.report_result(desired_result, result)
            desired_result = api.get_next_desired_result()
    except KeyboardInterrupt:
        pass

    best_cfg = api.get_best_configuration()
    if best_cfg:
        interface.save_final_config(best_cfg)

if __name__ == "__main__":
    run_ot_search("benchmarks/montecarlo.cpp", timeout_seconds=5400)