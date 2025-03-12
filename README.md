# EffectiveDiffusivityFVM

This repository is dedicated to the simulation of effective diffusivity in 2D and 3D structures via the Finite Volume Method (FVM). This approach was designed for maximum efficiency when generating large datasets for machine learning applications, thus uses the pixel resolution of the image as the base mesh for the simulation. Below is basic information on how to compile and run this code. 

For more detailed information about the code itself, refer to the documentation [pdf](https://github.com/adama-wzr/EffectiveDiffusivityFVM/blob/ExperimentalBranch/Deff2DGPU/Effective%20Diffusivity%20Documentation.pdf).

# Table of Contents

1. [Requirements](#requirements)
2. [GPU Compilation](#gpu-compilation)
3. [Required Files](#required-files)
4. [How to Cite](#how-to-cite)
5. [Authors](#code-authors)
6. [Documentation](#documentation)
7. [Acknowledgements](#acknowledgements)
8. [Upcoming Changes](#upcoming-changes)

## Requirements

This list reflects what we tested on and can confirm that runs properly, but older versions might work. Might work with other compilers as well.
- NVIDIA Compute capability >= 8.6
- CUDA >= 11.5
- gcc >= 11.4
- C++17 or newer
- [stb_image](https://github.com/nothings/stb) any recent version

The code has been tested on Ubuntu >= 20.04, Windows 10 and 11, and on Rocky Linux 8.7.

## GPU Compilation

With the NVIDIA suite installed properly and already added to the path, also assuming all required files are in the same folder. There might be different requirements loading the OpenMP library. On Windows, use the following:

```bash
nvcc -Xcompiler -openmp main.cu
```

If getting errors related to std library, that is likely due to multiple C++ versions being present. Add the following flag to compilation:

```bash
nvcc -std=c++17 -Xcompiler -openmp main.cu
```
This is mainly an issue on Windows. Any version that is C++17 or more recent should work.

Sometimes, the code may fail to launch the kernel for the GPU. There are multiple reasons why that might be the case. If the drivers are up-to-date and the kernel are still not launching, specifying the architecture of the GPU will generally solve the problem:

```bash
nvcc -std=c++17 -Xcompiler -openmp -arch=sm_XX main.cu
```

where we replace the "XX" by the compute capability of the GPU (i.e. compute capability 5.2 would be `-arch=sm_52`). TO verify your compute capability, check here [CUDA GPUs - Compute Capability](https://developer.nvidia.com/cuda-gpus).

## Required Files

All these files have to be in the same folder (or in the path for compilation/run).

- 2D grayscale .jpg image, 3D stack, or 3D structure saved as .csv
- Main Deff2D file (.cpp or .cu)
- Helper Deff2D file (.h or .cuh)
- input.txt
- stb_image.h

## How to Cite

Please cite one of the relevant publications shown below (more coming):

- Sarabandi, A., Adam, A., & Li, X. (2024). Influence of Electrolyte Saturation on the Performance of Li-O2 Batteries. ACS Applied Materials and Interfaces. https://doi.org/10.1021/acsami.4c12168

## Code Authors

- Main developer: Andre Adam (The University of Kansas)
    - [ResearchGate](https://www.researchgate.net/profile/Andre-Adam-2)
    - [GoogleScholar](https://scholar.google.com/citations?hl=en&user=aP_rDkMAAAAJ)
    - [GitHub](https://github.com/adama-wzr)
- Advisor: Dr. Xianglin Li (Washingtion University in St. Louis)
    - [Website](https://xianglinli.wixsite.com/mysite)
    - [GoogleScholar](https://scholar.google.com/citations?user=8y0Vd8cAAAAJ&hl=en)

 ## Documentation

The documentation pdf is a more in-depth source on the mathematical formulation and code implementation, while also providing technical insight on how to run and modify the code included in this repository.

## Acknowledgements

This work wouldn't be possible without the computational time awarded as part of the following grants:

This work used Expanse(GPU) at SDSC through allocations MAT210014 and MAT230071 from the Advanced Cyberinfrastructure Coordination Ecosystem: Services & Support (ACCESS) program, which is supported by National Science Foundation grants #2138259, #2138286, #2138307, #2137603, and #2138296.

## Upcoming Changes

Changes will be coming to this repository soon. A new branch for development will come out and start implementing some of the features described below, and they will only be committed to this folder once the changes are stable. For now, the folder with the 2D Effective Diffusivity code will remain unchanged.

The list of new changes and new capabilities to the code will be added as follows (the order might change, but this is a tentative implementation guide):

1. Expanded more general derivation of equations will allow for time dependent solutions.
2. Mass generation/destruction will be implemented.
3. CPU version with multi-threading.
4. 3D version of the code will be available for 3D structures (with either a stack of 2D images or a csv file encoding in 3D).
5. A multi-GPU version with HPC resources in mind for large-scale simulations.
6. More flexible boundary conditions.
7. Simulation mode for calculating tortuosity.

Additionally, some experimental features might come in the near future. At this time, I cannot provide a reasonable estimate of when these will be implemented (or if they will at all). In no particular order:

- GUI:
  - The actual GUI is a way to facilitate and/or guide the generation of the input file and calling the appropriate code versions based on user input.
  - The GUI won't be necessary. While code can run from the GUI itself, the code will also run without the GUI.
- Higher-order discretization methods.
- For the GPU code, I will try and implement some of the most recent cuBLAS and cuSPARSE solvers.
- For the CPU code, implementation of [scaLAPACK](https://www.netlib.org/scalapack/) for solving the sparse systems. I will test those versus the already existing solvers, so we will see what works best.
  - The GPU code in 3D, with the same solver as the CPU version, can be hundreds of times faster than the CPU counterpart. In other words, with parallel computing, a mid-range GPU (like a GeFORCE RTX 3070) can be as fast as 100 CPUs (also with the bold assumption of 100% efficiency on the parallel CPU code). Therefore, the development will first focus on GPU and multi-GPU code as opposed to CPU.
- More meshing options.
  - The current meshing approach is very rudimentary. For estimating bulk properties of the domain, the meshing does not seem to be a problem. However, the local accuracy might be sacrificed in some cases. I will try to add some methods for mesh refinement in some locations.
