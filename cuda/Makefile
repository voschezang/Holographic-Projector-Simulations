# full abs path is required for ssh
C := /usr/local/cuda-10.2/bin/nvcc
EXE := run
DIR := ../tmp
ZIP := ../tmp/out.zip
REL_OUTPUT_FILES := *.dat
HyperParams := KERNEL_SIZE='8'
# TODO add stream size, kernel size, x distribution, Ny, Nz
.PHONY: $(EXE)

build:
	# -ftz=true // flush dernormal to zero
	#  -ftz=true -prec-div=false -prec-sqrt=false
	# TODO check -O2 vs. -O3 performance
	$(C) -o $(EXE) main.cu -l curand -std=c++11 -D$(HyperParams)

debug:
	$(C) -o $(EXE) main.cu -l curand -std=c++11 -D$(HyperParams) -g
	make run

build-run:
	make build && make run

run:
	make cleanup-output
	./$(EXE)
	make zip

zip:
	rm -f $(ZIP)
	cd $(DIR) && zip -v $(ZIP) out.json $(REL_OUTPUT_FILES) &> /dev/null

cleanup-output:
	rm -rf $(DIR)/$(REL_OUTPUT_FILES)

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
