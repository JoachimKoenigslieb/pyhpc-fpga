kernels.xclbin: emconfig.json gtsv_kernel.xo sqrt_kernel.xo add.xo sub.xo mult.xo div.xo vmax.xo
	v++ -t sw_emu --config design.cfg -l -g -o kernels.xclbin gtsv_kernel.xo sqrt_kernel.xo add.xo sub.xo mult.xo div.xo vmax.xo

emconfig.json:
	emconfigutil --platform xilinx_u200_xdma_201830_2

host: ../src/host.cpp
	g++ -o host ../src/host.cpp -I/opt/xilinx/xrt/include/ -I/tools/Xilinx/Vivado/2020.1/include/ -I../src/xcl2/ -L/opt/xilinx/xrt/lib/ -L/usr/local/lib -L/usr/lib -lOpenCL -lcnpy -lz -lpthread -lrt -lstdc++ -std=c++14 -w -O0 -g

gtsv_kernel.xo: ../src/gtsv_kernel.cpp
	v++ -t sw_emu --config design.cfg -c -k gtsv -I/tools/Xilinx/Vivado/2020.1/include/ -I../src/xf_solver/L2/include/ -I../src/ -o gtsv_kernel.xo ../src/gtsv_kernel.cpp

sqrt_kernel.xo: ../src/sqrt_kernel.cpp
	v++ -t sw_emu --config design.cfg -c -k vsqrt -I/tools/Xilinx/Vivado/2020.1/include/ -I../src/xf_solver/L2/include/ -I../src/ -o sqrt_kernel.xo ../src/sqrt_kernel.cpp

add.xo: ../src/vadd_kernel.cpp
	v++ -t sw_emu --config design.cfg -c -k vadd -I/tools/Xilinx/Vivado/2020.1/inclide/ -I../src/ -o add.xo ../src/vadd_kernel.cpp

sub.xo: ../src/vsub_kernel.cpp
	v++ -t sw_emu --config design.cfg -c -k vsub -I/tools/Xilinx/Vivado/2020.1/inclide/ -I../src/ -o sub.xo ../src/vsub_kernel.cpp

mult.xo: ../src/vmult_kernel.cpp
	v++ -t sw_emu --config design.cfg -c -k vmult -I/tools/Xilinx/Vivado/2020.1/inclide/ -I../src/ -o mult.xo ../src/vmult_kernel.cpp

div.xo: ../src/vdiv_kernel.cpp
	v++ -t sw_emu --config design.cfg -c -k vdiv -I/tools/Xilinx/Vivado/2020.1/inclide/ -I../src/ -o div.xo ../src/vdiv_kernel.cpp

vmax.xo: ../src/vmax_kernel.cpp
	v++ -t sw_emu --config design.cfg -c -k vmax -I/tools/Xilinx/Vivado/2020.1/inclide/ -I../src/ -o vmax.xo ../src/vmax_kernel.cpp

clean:
	rm -f *.log
	rm -f *_summary
	rm -f *.info
	rm -f *.xo
	rm -rf ./_x
