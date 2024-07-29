CC=g++
PYBIND_INCLUDES=$(shell echo `python -m pybind11 --includes`)
PYTHON_SUFFIX=$(shell echo `python3-config --extension-suffix`)
CPP_FLAGS=-O3 -Wall -std=c++14 -fopenmp -fipa-sra
PYBIND_FLAGS =-shared -fPIC
 
all: ec_direct$(PYTHON_SUFFIX)
 
ec_direct$(PYTHON_SUFFIX): py_ec_lib_direct.cpp ec_lib_direct.cpp
	$(CC) $(CPP_FLAGS) $(PYBIND_FLAGS) $(PYBIND_INCLUDES) $^ -o $@
	mv ec_direct$(PYTHON_SUFFIX) ../../lib_wrappers/direct
