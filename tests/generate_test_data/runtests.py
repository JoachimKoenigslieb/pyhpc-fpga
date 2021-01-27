# Generate test data and answers for numpy equivalent functions

import glob 
import os
import numpy as np
import shutil
from copy import copy
import subprocess


os.environ["XILINX_XRT"] = "/opt/xilinx/xrt"
os.environ["PATH"] = """/opt/xilinx/xrt/bin:/tools/Xilinx/Vitis/2020.1/bin:/tools/Xilinx/Vitis/2020.1/gnu/microblaze/lin/bin:/tools/Xilinx/Vitis/2020.1/gnu/arm/lin/bin:/tools/Xilinx/Vitis/2020.1/gnu/microblaze/linux_toolchain/lin64_le/bin:/tools/Xilinx/Vitis/2020.1/gnu/aarch32/lin/gcc-arm-linux-gnueabi/bin:/tools/Xilinx/Vitis/2020.1/gnu/aarch32/lin/gcc-arm-none-eabi/bin:/tools/Xilinx/Vitis/2020.1/gnu/aarch64/lin/aarch64-linux/bin:/tools/Xilinx/Vitis/2020.1/gnu/aarch64/lin/aarch64-none/bin:/tools/Xilinx/Vitis/2020.1/gnu/armr5/lin/gcc-arm-none-eabi/bin:/tools/Xilinx/Vitis/2020.1/tps/lnx64/cmake-3.3.2/bin:/tools/Xilinx/Vitis/2020.1/cardano/bin:/tools/Xilinx/Vivado/2020.1/bin:/tools/Xilinx/DocNav:/opt/xilinx/xrt/bin:/tools/Xilinx/Vitis/2020.1/bin:/tools/Xilinx/Vitis/2020.1/gnu/microblaze/lin/bin:/tools/Xilinx/Vitis/2020.1/gnu/arm/lin/bin:/tools/Xilinx/Vitis/2020.1/gnu/microblaze/linux_toolchain/lin64_le/bin:/tools/Xilinx/Vitis/2020.1/gnu/aarch32/lin/gcc-arm-linux-gnueabi/bin:/tools/Xilinx/Vitis/2020.1/gnu/aarch32/lin/gcc-arm-none-eabi/bin:/tools/Xilinx/Vitis/2020.1/gnu/aarch64/lin/aarch64-linux/bin:/tools/Xilinx/Vitis/2020.1/gnu/aarch64/lin/aarch64-none/bin:/tools/Xilinx/Vitis/2020.1/gnu/armr5/lin/gcc-arm-none-eabi/bin:/tools/Xilinx/Vitis/2020.1/tps/lnx64/cmake-3.3.2/bin:/tools/Xilinx/Vitis/2020.1/cardano/bin:/tools/Xilinx/Vivado/2020.1/bin:/tools/Xilinx/DocNav:/home/joachim/.local/bin:/home/joachim/bin:/home/joachim/anaconda3/bin:/home/joachim/anaconda3/condabin:/home/joachim/.local/bin:/home/joachim/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin:/home/joachim/.dotnet/tools:/home/joachim/.dotnet/tools:/home/joachim/bin:/home/joachim/.npm-packages/bin:/home/joachim/bin:/home/joachim/.npm-packages/bin"""
os.environ["LD_LIBRARY_PATH"] = "/opt/xilinx/xrt/lib:/opt/xilinx/xrt/lib:"
os.environ["PYTHONPATH"] = "/opt/xilinx/xrt/python:/opt/xilinx/xrt/python:"
os.environ["XCL_EMULATION_MODE"] = "hw_emu"
os.environ["LIBRARY_PATH"] = "/usr/lib/x86_64-linux-gnu"


PATH = '/home/joachim/Documents/speciale/vitis/veros-port/tests'

files = glob.glob('../*')
files.remove('../generate_test_data')
files.remove('../gtsv')
files.remove('../shared')

files = [file.strip('../') for file in files]

for test_file in files:
    os.chdir(f'{PATH}/{test_file}')
    print(f'Running test for {test_file}....')
    proc = subprocess.run(['./hw_emu_host'], stdout=subprocess.PIPE, text=True, stderr=subprocess.PIPE)
    pout = proc.stdout.split('\n')
    magic_ind = [i for i, line in enumerate(pout) if line.startswith('checksum')][0]
    result = pout[magic_ind:magic_ind+2] 
    print(' '.join(result))
    print('----------')



