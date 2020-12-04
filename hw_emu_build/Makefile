SRC_DIR := ../hw_emu_src
KERNEL_DIR := ${SRC_DIR}/kernels
OBJ_DIR := ./objects
TARGET := hw_emu
KERNEL_INCLUDES := -I/tools/Xilinx/Vivado/2020.1/include/ -I${SRC_DIR}/xf_solver/L2/include/ 
HOST_INCLUDES := -I/tools/Xilinx/Vivado/2020.1/include/ -I/tools/xilinx/xrt/include/ -I${SRC_DIR}/xcl2/
KERNEL_LIBS := -L/opt/xilinx/xrt/lib/ -lOpenCL -lz -lpthread -lrt -lstdc++ -std=c++14 -w -O0 -g
HOST_LIBS  := -L/opt/xilinx/xrt/lib/ -L/usr/local/lib -L/usr/lib -lOpenCL -lcnpy -lz -lpthread -lrt -lstdc++ -std=c++14 -w -O0 -g

SOURCES := $(wildcard ${KERNEL_DIR}/*.cpp)
OBJECTS := $(patsubst %.cpp,%.xo,$(addprefix $(TARGET)_objects/,$(notdir $(SOURCES))))

$(info $$var is [${OBJECTS}])

$(TARGET)_kernels.xclbin: $(OBJECTS) 
	v++ -t $(TARGET) --config design.cfg -l -g -o $@ $^

$(TARGET)_objects/%.xo: $(KERNEL_DIR)/%.cpp
	v++ -t $(TARGET) --config design.cfg -c -k $(basename $(notdir $<)) $(KERNEL_INCLUDES) ${KERNEL_LIBS} -o $@ $<

host: $(SRC_DIR)/host.cpp
	g++ -o host $< ${HOST_INCLUDES} ${HOST_LIBS} -g

clean:
	rm -f *.log
	rm -f *_summary
	rm -f *.info
	rm -rf ./_x
	rm -rf ./.run
	rm -rf ./.Xil
	rm ./objects_hw_emu/*compile_summary
	rm ./objects_sw_emu/*compile_summary
