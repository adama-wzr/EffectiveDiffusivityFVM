# EffectiveDiffusivityFVM

This repository is dedicated to the simulation of effective diffusivity in 2D structures via the Finite Volume Method (FVM). This approach was designed for maximum efficiency when generating large datasets for machine learning applications, thus uses the pixel resolution of the image as the base mesh for the simulation. Below is basic information on how to compile and run this code. For more detailed information about the code itself, refer to the documentation pdf. For more information regarding the computational model, refer to the publication (in preparation).

This repository includes one code version only: it is meant to run on a CUDA-capable GPU. However, if there is a need for another mode, please contact one of the authors as it is easy to make changes (GPU vs. CPU) as they only affect the solver, not the numerical model.

# Table of Contents

1. [Requirements](#requirements)
2. [GPU Compilation](#gpu-compilation)

## Requirements

This list reflects what we tested on and can confirm that runs properly, but older versions might work. Might work with other compilers as well.
- NVIDIA Compute capability >= 8.6
- CUDA >= 11.0
- gcc >= 11.0
- [stb_image](https://github.com/nothings/stb) latest version

The code has been tested on Ubuntu >= 20.04, Windows 10 and 11. The HPC version has only been tested on Rocky Linux 8.7.

## GPU Compilation

With the NVIDIA suite installed properly and already added to the path, also assuming all required files are in the same folder.

```bash
nvcc Perm2D.cu
```
