import subprocess
import os

def compile_program(source_file, flags, output):
    sdk = subprocess.check_output(["xcrun", "--sdk", "macosx", "--show-sdk-path"]).decode().strip()
    cmd = ["g++-15", "-isysroot", sdk] + flags + [source_file, "-o", output]

    try:
        subprocess.run(cmd, check=True, stderr=subprocess.STDOUT)
        return True
    except subprocess.CalledProcessError:
        return False
