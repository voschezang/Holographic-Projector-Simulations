# Holographic-Projector

High performance simulations of [holographic](https://en.wikipedia.org/wiki/Holography) projectors for GPU's, using [CUDA](https://docs.nvidia.com/cuda/).
More info can be found on [wikipedia](https://en.wikipedia.org/wiki/Wave_interference).

### Project structure

- `/cuda` contains C++/CUDA code. Additionally the CUDA libraries cuBLAS, thrust and CUB are used.

- `/matlab` contains Matlab scripts, that can be used to run simulations.

- `/py` contains legacy Python code, and does not require a GPU.

<img src='img_readme/True_MC.png' alt='Holographic Projection Example'>


### Setup & Usage

Compile the CUDA program using `make build`.
This creates a CLI application `cuda/run` that can compute superpositions. 
It computes superpositions for each `target` position and w.r.t. all `source` positions.

The source and target datasets can be generated using the CLI application, or using an external program.
For example, `matlab/projector.m` can be used to generate the necessary dataset files.

The distributions that are computed are the projector distribution and the projection distribution.
To use external data, use the flag `-f {directory}` to incdicate the name of the directory that contains the dataset files.
These files should be binary arrays for double-precision floating-point values (Little-endian encoding by default) and should be named as follows.
- `x0_amp.dat, x0_phase.dat`
- `u0.dat` (for the source data)
- `v0.dat` (for the _projector_ target positions)
- `w0.dat` (for the _projection_ target positions) - this file is only used if the boolean flag `-F` is supplied.

---



<img src='img_readme/1pt.png'>

<img src='img_readme/5pt.png'>





### Overview of CUDA code

The main kernel to compute superpositions is `superposition::per_block`, 
which repeatedly calls `superposition::phasor_displacement`.

The main files are `cuda/main.h,.cu`, in which functions declared in `cuda/transform.cu` are called to compute superposition transformations:

- a brute force transformation `transform_full()`

The are a number of variants of `SuperpositionPerBlockHelper` macros which are used in combination with `superposition_per_block_helper()` functions.
They allow the GPU geometry to be included as templates, which is required for CUB library functions.

Additionally `cuda/macros.h` contain macros and constants.

---

Various tests are included in `cuda/test.cu` and `cuda/test_gpu.cu`. 
There are no tests written for the MC estimators (`transform()`).

