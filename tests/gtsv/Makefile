KERNEL_DIR := ./kernels
OBJ_DIR := ./objects
TARGET := hw_emu

KERNEL_INCLUDES := -I../shared/xcl2/  -I./xf_solver/L2/include/

HOST_INCLUDES := -I/opt/xilinx/xrt/include/ -I/tools/Xilinx/Vivado/2020.1/include/ -I../shared/xcl2/
HOST_LIBS  := -L/opt/xilinx/xrt/lib/ -L/usr/local/lib -L/usr/local/include -L/usr/lib -lOpenCL -lcnpy -lz -lpthread -llapack -lrt -lstdc++ -std=c++14 -w -O0 -g

SOURCES := $(wildcard ${KERNEL_DIR}/*.cpp)
OBJECTS := $(patsubst %.cpp,%.xo,$(addprefix $(TARGET)_objects/,$(notdir $(SOURCES))))

$(TARGET)_kernels.xclbin: $(OBJECTS)
	v++ -t $(TARGET) --config design.cfg -l -g -o $@ $^

$(TARGET)_objects/%.xo: $(KERNEL_DIR)/%.cpp
	v++ -t $(TARGET) --config design.cfg -c -k $(basename $(notdir $<)) $(KERNEL_INCLUDES) -o $@ $<

host: host.cpp
	g++ -o $(TARGET)_host $< ${HOST_INCLUDES} ${HOST_LIBS} -g

clean:
	rm -f *.log
	rm -f *_summary
	rm -f *.info
	rm -rf ./_x
	rm -rf ./.run
	rm -rf ./.Xil
	rm ./objects_hw_emu/*compile_summary
	rm ./objects_sw_emu/*compile_summary
