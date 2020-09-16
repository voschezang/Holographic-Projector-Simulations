# Holographic-Projector

High performance simulations of holographic projectors for GPU's.

### Project structure

`/cuda` contains C++/CUDA code.

`/py` contains Python code, and does not require a GPU.

Additionally it contains some experimental jupyter notebooks (deprecated).


### Overview of CUDA code

The main kernel to compute superpositions is `superposition::per_kernel`, 
which repeatedly calls `superposition::phasor_displacement`.

The main files are `cuda/main.cu`, which calls `cuda/functions.cu` to compute superposition transformations.
Specifically a brute force transformation `transform_full()` and a Monte Carlo version `transform()` which is less accurate but potentially faster.
The are a number of variants of `SuperpositionPerBlockHelper` macros which are used in combination with `superposition_per_block_helper()` functions.
They allow the GPU geometry to be included as templates, which is required for CUB library functions.



