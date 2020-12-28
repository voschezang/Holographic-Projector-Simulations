# Holographic-Projector

High performance simulations of holographic projectors for GPU's.

### Project structure

- `/cuda` contains C++/CUDA code. Additionally the CUDA libraries cuBLAS, thrust and CUB are used.

- `/py` contains Python code, and does not require a GPU.


### Overview of CUDA code

The main kernel to compute superpositions is `superposition::per_block`, 
which repeatedly calls `superposition::phasor_displacement`.

The main files are `cuda/main.h,.cu`, in which functions declared in `cuda/transform.cu` are called to compute superposition transformations:

- a brute force transformation `transform_full()`
- a Monte Carlo version `transform_MC()` which is less accurate but potentially faster

The are a number of variants of `SuperpositionPerBlockHelper` macros which are used in combination with `superposition_per_block_helper()` functions.
They allow the GPU geometry to be included as templates, which is required for CUB library functions.

Additionally `cuda/macros.h` contain macros and constants.
The file `cuda/hyper_params.h` will be removed.
