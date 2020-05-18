C := nvcc # e.g. /usr/local/cuda-10.2/bin/nvcc
EXE := run
DIR := ../tmp
ZIP := ../tmp/out.zip
HyperParams = EXAMPLE='"text"'
# TODO add stream size, kernel size, x distribution, Ny, Nz
.PHONY: $(EXE)

build:
	# -ftz=true // flush dernormal to zero
	#  -ftz=true -prec-div=false -prec-sqrt=false
	$(C) -o $(EXE) main.cu -l curand -std=c++11 -D$(HyperParams)

build-run:
	make build && make run

run:
	./$(EXE)
	make zip


zip:
	rm -f $(ZIP)
	cd $(DIR) && zip $(ZIP) out.json x_amp.dat x_phase.dat u.dat

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
