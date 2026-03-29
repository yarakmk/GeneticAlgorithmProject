from opentuner import ConfigurationManipulator, EnumParameter, MeasurementInterface, Result
import json
import subprocess
import time, os, statistics
from compile_utils import compile_program
from measure_utils import measure_runtime

class GCCFlagTuner(MeasurementInterface):

    def manipulator(self):
        manip = ConfigurationManipulator()
        
        # load all flags
        flags = json.load(open("all_gcc_flags.json"))["flags"]

        for f in flags:
            manip.add_parameter(
                EnumParameter(f, ["on", "off"])
            )

        return manip
    
    #converts OpenTuner’s parameters into real GCC flags.
    def config_to_flag_list(self, cfg):
        flags = ["-O3"]  # Base level

        for flagname, setting in cfg.items():
            if setting == "on":
                flags.append(f"-f{flagname}")
            else:
                flags.append(f"-fno-{flagname}")

        return flags
    
    def run(self, desired_config, input, limit):
    
        flags = self.config_to_flag_list(desired_config.configuration.data)
        exe = "benchmarks/matmul_opentuner.out"

        if not compile_program("benchmarks/matmul.cpp", flags, exe):
            return Result(time=float("inf"))

        time_taken = measure_runtime(exe)
        return Result(time=time_taken)