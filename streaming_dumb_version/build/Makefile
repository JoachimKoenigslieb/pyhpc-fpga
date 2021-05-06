SRC_DIR := ../src
KERNEL_DIR := ${SRC_DIR}/kernels
OBJ_DIR := ./objects

TARGET := hw_emu
PLATFORM := xilinx_u200_xdma_201830_2 

KERNEL_INCLUDES := -I/tools/Xilinx/Vivado/2020.1/include/ -I${SRC_DIR}/includes/

HOST_INCLUDES := -I/opt/xilinx/xrt/include/ -I/tools/Xilinx/Vivado/2020.1/include/ -I${SRC_DIR}/includes/

HOST_LIBS  := -L/opt/xilinx/xrt/lib/ -lOpenCL -lz -lpthread -lrt -lstdc++ -std=c++14 -w -O0 -g 

SOURCES := $(wildcard ${KERNEL_DIR}/*.cpp)
OBJECTS := $(patsubst %.cpp,%.xo,$(addprefix $(TARGET)_objects/,$(notdir $(SOURCES))))

all: host ${TARGET}_kernels.xclbin

$(TARGET)_kernels.xclbin: $(OBJECTS) 
	v++ -t $(TARGET) --platform ${PLATFORM} -l --config design.cfg -g -o $@ $^

$(TARGET)_objects/%.xo: $(KERNEL_DIR)/%.cpp
	v++ -t $(TARGET) --platform ${PLATFORM} --config design.cfg -g -c -k $(basename $(notdir $<)) $(KERNEL_INCLUDES) -o $@ $<

emconfig.json:
	emconfigutil --platform ${PLATFORM}

host: $(SRC_DIR)/host.cpp 
	g++ -o host $^ ${HOST_INCLUDES} ${HOST_LIBS} -g

clean:
	rm -f *.log
	rm -f *_summary
	rm -f *.info
	rm -rf ./_x
	rm -rf ./.run
	rm -rf ./.Xil
	rm -f ./objects_hw_emu/*compile_summary
	rm -f ./objects_sw_emu/*compile_summary
	rm -f *.wdb
	rm -f *.protoinst
	rm -f *.wcfg
