C := nvcc # e.g. /usr/local/cuda-10.2/bin/nvcc
EXE := run
ZIP := ../tmp/out.zip
.PHONY: $(EXE)

build:
	# -ftz=true // flush dernormal to zero
	#  -ftz=true -prec-div=false -prec-sqrt=false
	$(C) -o $(EXE) main.cu -l curand -std=c++11

build-run:
	make build && make run

run:
	./$(EXE)
	make zip


zip:
	rm -f $(ZIP)
	zip $(ZIP) ../tmp/out.txt

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
