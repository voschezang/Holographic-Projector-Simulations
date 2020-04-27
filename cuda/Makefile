EXE := run
.PHONY: $(EXE)

build:
	# make add-to-path
	# nvcc -o main main.cu
	# -ftz=true // flush dernormal to zero
	#  -ftz=true -prec-div=false -prec-sqrt=false
	# /usr/local/cuda-10.2/bin/nvcc -o main main.cu -l curand
	# /usr/local/cuda-10.2/bin/nvcc -o main main.cu -l curand -std=c++11
	nvcc -o $(EXE) main.cu -l curand -std=c++11

build-run:
	make build && make run

run:
	./$(EXE)

# CUDA
init-path:
	export PATH=/usr/local/cuda-10.1/bin${PATH:+:${PATH}}

profile:
	make build
	nvprof ./$(EXE)

vprofile:
	make build
	nvvp ./$(EXE)

setup-profiler:
	modprobe nvidia NVreg_RestrictProfilingToAdminUsers=0

add-to-path:
	echo 'export PATH=/usr/local/cuda-10.1/bin${PATH:+:${PATH}}'
