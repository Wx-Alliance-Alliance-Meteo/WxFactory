
CXX ?= g++
NVCC ?= nvcc
CXXFLAGS ?= -shared -std=c++11 -fPIC
NVCCFLAGS ?= -shared -std=c++11 -Xcompiler -fPIC
INCLUDE := $(shell python3 -m pybind11 --includes)

CXXFLAGS += $(INCLUDE)
NVCCFLAGS += $(INCLUDE)

# EXT_SUFFIX := $(shell python3-config --extension-suffix)
EXT_SUFFIX := .so

LIB_DIR := lib/pde
CPP_OUT := $(LIB_DIR)/interface_c$(EXT_SUFFIX)
CUDA_OUT := $(LIB_DIR)/interface_cuda$(EXT_SUFFIX)

COMMON_HEADERS := $(wildcard pde/*.hpp) $(wildcard pde/kernels/*.hpp) $(wildcard pde/kernels/*.h)

.PHONY: all clean cpp cuda

all: cpp cuda
cpp: $(CPP_OUT)
cuda: $(CUDA_OUT)

$(LIB_DIR):
	@mkdir -pv $(LIB_DIR)

$(CPP_OUT): pde/interface.cpp $(COMMON_HEADERS) $(LIB_DIR)
	$(CXX) $(CXXFLAGS) $< -o $@

$(CUDA_OUT): pde/interface.cu $(COMMON_HEADERS) $(LIB_DIR)
	$(NVCC) $(NVCCFLAGS) $< -o $@

clean:
	rm -rf $(LIB_DIR)

clean-cpp:
	rm -rf $(CPP_OUT)

clean-cuda:
	rm -rf $(CUDA_OUT)
	

# c++ -shared -std=c++11 -fPIC $(python3 -m pybind11 --includes) interface.cpp -o interface_c$(python3-config --extension-suffix)
# nvcc -shared -std=c++11 -Xcompiler -fPIC $(python3 -m pybind11 --includes) interface.cu -o interface_cuda$(python3-config --extension-suffix)
