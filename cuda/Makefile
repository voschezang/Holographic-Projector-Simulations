# full abs path is required for ssh # TODO not if sourcing .profile?
# C := /usr/local/cuda-10.2/bin/nvcc
# C := /usr/local/cuda-11.0/bin/nvcc
# GCC := /cvmfs/sft.cern.ch/lcg/releases/gcc/8.3.0.1/x86_64-centos7/bin
# GCC := /cvmfs/sft.cern.ch/lcg/releases/gcc/9.2.0/x86_64-centos7/bin
#-ccbin=$(GCC)
# last performance measurement: -O2 was 3% faster than -O3, with 1024^2 x 1024^2 datapoints (0.80 vs 0.77 TFLOPS)
FLAGS := -l curand -l cublas -std=c++14
ARCH_FLAGS := -arch=compute_70 -code=sm_70
EXE := run
DIR := ../tmp
ZIP := ../tmp/out.zip
REL_OUTPUT_FILES := *.dat
# TODO add stream size, kernel size, x distribution, Ny, Nz
.PHONY: $(EXE) test

build:
	# -ftz=true // flush dernormal to zero
	#  -ftz=true -prec-div=false -prec-sqrt=false
	# -g debug
	# -D$(HyperParams)
	nvcc -o $(EXE) main.cu $(FLAGS) $(ARCH_FLAGS)

debug:
	nvcc -o $(EXE) main.cu $(FLAGS) $(ARCH_FLAGS)
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
	# export PATH=$(GCC)${PATH:+:${PATH}}

test:
	nvcc -o test test.cu $(FLAGS) -DNORM_3D='norm3d_host' && ./test
	nvcc -o test test_gpu.cu $(FLAGS) && ./test
	rm test;

test-phase-projector:
	# params should be: 1440 hd nonrand screen, no PROJECT_PHASE=0, 1pt, offset 4mm 6mm 10mm, non
	make build-run
	diff ../tmp/x0_amp.dat   ../tmp/test/phase_1440_hd_nonrand_1pt-4-6-10mm/x0_amp.dat
	diff ../tmp/x0_phase.dat ../tmp/test/phase_1440_hd_nonrand_1pt-4-6-10mm/x0_phase.dat
	diff ../tmp/y0_amp.dat   ../tmp/test/phase_1440_hd_nonrand_1pt-4-6-10mm/y0_amp.dat
	diff ../tmp/y0_phase.dat ../tmp/test/phase_1440_hd_nonrand_1pt-4-6-10mm/y0_phase.dat

test-phase-projector-reset:
	cp ../tmp/x0_amp.dat   ../tmp/test/phase_1440_hd_nonrand_1pt-4-6-10mm/
	cp ../tmp/x0_phase.dat ../tmp/test/phase_1440_hd_nonrand_1pt-4-6-10mm/
	cp ../tmp/y0_amp.dat   ../tmp/test/phase_1440_hd_nonrand_1pt-4-6-10mm/
	cp ../tmp/y0_phase.dat ../tmp/test/phase_1440_hd_nonrand_1pt-4-6-10mm/

test-nonphase-projector:
	# params should be: 1440 hd nonrand screen, no PROJECT_PHASE=0, 1pt, offset 4mm 6mm 10mm, non
	make build-run
	diff ../tmp/x0_amp.dat   ../tmp/test/nonphase_1440_hd_nonrand_1pt-4-6-10mm/x0_amp.dat
	diff ../tmp/x0_phase.dat ../tmp/test/nonphase_1440_hd_nonrand_1pt-4-6-10mm/x0_phase.dat
	diff ../tmp/y0_amp.dat   ../tmp/test/nonphase_1440_hd_nonrand_1pt-4-6-10mm/y0_amp.dat
	diff ../tmp/y0_phase.dat ../tmp/test/nonphase_1440_hd_nonrand_1pt-4-6-10mm/y0_phase.dat

test-nonphase-projector-reset:
	cp ../tmp/x0_amp.dat   ../tmp/test/nonphase_1440_hd_nonrand_1pt-4-6-10mm/
	cp ../tmp/x0_phase.dat ../tmp/test/nonphase_1440_hd_nonrand_1pt-4-6-10mm/
	cp ../tmp/y0_amp.dat   ../tmp/test/nonphase_1440_hd_nonrand_1pt-4-6-10mm/
	cp ../tmp/y0_phase.dat ../tmp/test/nonphase_1440_hd_nonrand_1pt-4-6-10mm/

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
