#! /usr/bin/python3

r'''###############################################################################
###################################################################################
#
#	Fast Parallel Extract Python Module
#	Version 1.0
#
#	Project Los Angeles
#
#	Tegridy Code 2025
#
#   https://github.com/Tegridy-Code/Project-Los-Angeles
#
###################################################################################
###################################################################################
#
#   Copyright 2025 Project Los Angeles / Tegridy Code
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
#
###################################################################################
###################################################################################
#
#   Critical packages
#
#   !sudo apt update -y
#   !sudo apt install -y p7zip-full
#   !sudo apt install -y pigz
#
###################################################################################
###################################################################################
#
#   Basic use example
#
#   import fast_parallel_extract
#
#   fast_parallel_extract.fast_parallel_extract()
#
###################################################################################
'''

###################################################################################

import os
import shutil
import subprocess
import time

###################################################################################

def fast_parallel_extract(archive_path='./Godzilla-MIDI-Dataset/Godzilla-MIDI-Dataset-CC-BY-NC-SA.tar.gz', 
                          output_dir='./Godzilla-MIDI-Dataset/', 
                          pigz_procs=256
                         ):
    
    print('=' * 70)
    print('Extracting...')

    start_time = time.time()

    if not os.path.isfile(archive_path):
        raise FileNotFoundError(f"The archive file '{archive_path}' does not exist.")

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    if shutil.which("pigz") is None:
        raise EnvironmentError("The 'pigz' package is not installed or is not found in the system's PATH.")

    command = [
        "tar",
        "-I", f"pigz -p {pigz_procs}",
        "-xvf", archive_path,
        "-C", output_dir
    ]

    with open("/dev/null", "w") as devnull:
        subprocess.run(command, stdout=devnull, stderr=subprocess.STDOUT)

    end_time = time.time()
    execution_time = end_time - start_time

    print('Done!')
    print('=' * 70)
    print(f"Extraction took {execution_time / 60} minutes")
    print('=' * 70)

###################################################################################
# This is the end of the Fast Parallel Extract Python Module
###################################################################################