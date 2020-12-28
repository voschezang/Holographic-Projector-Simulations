FLAGS := -l curand -l cublas -std=c++14
ARCH_FLAGS := -arch=compute_70 -code=sm_70
EXE := run
DIR := ../tmp
ZIP := ../tmp/out.zip
REL_OUTPUT_FILES := *.dat
.PHONY: $(EXE) test

build:
	nvcc -o $(EXE) main.cu $(FLAGS) $(ARCH_FLAGS)
	mkdir -p $(DIR)

debug:
	nvcc -o $(EXE) main.cu $(FLAGS) $(ARCH_FLAGS) -g -G
	echo $(EXE)
	cuda-gdb

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

init-path:
	export PATH=/usr/local/cuda-11.0/bin${PATH:+:${PATH}}

profile:
	make build
	nvprof --print-gpu-trace ./$(EXE)

vprofile:
	make build
	nvvp ./$(EXE)

setup-profiler:
	modprobe nvidia NVreg_RestrictProfilingToAdminUsers=0

add-to-path:
	echo 'export PATH=/usr/local/cuda-10.1/bin${PATH:+:${PATH}}'
