CC=g++
PYBIND_INCLUDES=$(shell echo `python -m pybind11 --includes`)
PYTHON_SUFFIX=$(shell echo `python3-config --extension-suffix`)
CPP_FLAGS=-O3 -Wall -std=c++14 -fopenmp -fipa-sra
PYBIND_FLAGS =-shared -fPIC
 
all: ans$(PYTHON_SUFFIX)
 
ans$(PYTHON_SUFFIX): ans.cpp compressor.cpp decompressor.cpp
	$(CC) $(CPP_FLAGS) $(PYBIND_FLAGS) $(PYBIND_INCLUDES) $^ -o $@
	mv ans$(PYTHON_SUFFIX) ../../lib_wrappers/mans
