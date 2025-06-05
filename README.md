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
- [stb_image](https://github.com/nothings/stb) any recent version.
    - Only `stb_image.h` is necessary, you don't need to build the whole project.

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

where we replace the "XX" by the compute capability of the GPU (i.e. compute capability 5.2 would be `-arch=sm_52`). To verify your compute capability, check here [CUDA GPUs - Compute Capability](https://developer.nvidia.com/cuda-gpus).

**New Function Added:** I am in the process of refactoring the code as we make progress towards a more modular package. Therefore, some of the new code development is made as part of submodules to the main source code. The compilation of the main code did not change! For the new code, use the cmake file to build it.

While VisualStudio is recommended for building on Windows, if you do chose to build from the command line use the following:

1. Create a new folder, preferably titled `build`, and open the terminal from this folder.
2. Run `cmake ..` inside the folder.
3. Run `cmake --build . --config Release` inside the folder.
4. On Windows, this will create a folder `Release` with the executables inside.
5. On Linux, this should just create the executables in the build folder.


Some of the refactoring is still roughly incomplete, and lacking proper documentation. This is intentinal: updates to the documentation and official tested capabilities will always accompany publications. If you want to use the package in the meanwhile, please reach out to Andre Adam or Dr. Xianglin Li, whose contant information are included in the [Authors](#code-authors) section.

## Required Files

All these files have to be in the same folder (or in the path for compilation/run).

- 2D grayscale .jpg image, 3D stack, or 3D structure saved as .csv
- Main Deff2D file (.cpp or .cu)
- Helper Deff2D file (.h or .cuh)
- input.txt
- stb_image.h

## Publications

There currently isn't a consolidated publication for this package. Please cite one (or more) of the relevant publications below:

- Sarabandi, A., **Adam, A.**, & Li, X. (2024). Influence of Electrolyte Saturation on the Performance of Li-O2 Batteries. ACS Applied Materials and Interfaces. https://doi.org/10.1021/acsami.4c12168

## Code Authors

- Main developer: Andre Adam (The University of Kansas)
    - [ResearchGate](https://www.researchgate.net/profile/Andre-Adam-2)
    - [GoogleScholar](https://scholar.google.com/citations?hl=en&user=aP_rDkMAAAAJ)
    - [GitHub](https://github.com/adama-wzr)
- Advisor: Dr. Xianglin Li (Washingtion University in St. Louis)
    - [Website](https://xianglinli.wixsite.com/mysite)
    - [GoogleScholar](https://scholar.google.com/citations?user=8y0Vd8cAAAAJ&hl=en)

 ## Documentation

The documentation pdf is a more in-depth source on the mathematical formulation and code implementation, while also providing technical insight on how to run and modify the code included in this repository. The documentation only covers the old code in the Deff2D_GPU folder, which was part of the associated publication. A more comprehensive documentation folder will acompany major updates, version releases, and/or new publications.

If there are any questions in the meanwhile, please feel free to reach out to one of the authors.

## Acknowledgements

This work wouldn't be possible without the computational time awarded as part of the following grants:

This work used Expanse(GPU) at SDSC through allocations MAT210014 and MAT230071 from the Advanced Cyberinfrastructure Coordination Ecosystem: Services & Support (ACCESS) program, which is supported by National Science Foundation grants #2138259, #2138286, #2138307, #2137603, and #2138296.

## Upcoming Changes

Changes will be coming to this repository soon. A new branch for development will come out and start implementing some of the features described below, and they will only be committed to this folder once the changes are stable. For now, the folder with the 2D Effective Diffusivity code will remain unchanged.

Recent Added Changes:
1. Simulation mode for tortuosity in 2D and 3D has been implemented.
2. CPU version with multi-threading currently implemented. It will be separated from the GPU code in future updates.
3. Source term has been addded to equations, but is not currently used in any simulations.
4. 3D versions are already available.
5. Flexible boundary conditions are available.
6. Transient model (2D + 1D) is currently available, seems to work well.
7. Initial concentration distributions have been implemented for transient simulation, but not yet for the other methods.
8. Migration has been added (to 2D + 1D model), lacks documentation and validation at the moment.

Upcoming changes (in no particular order):
- Time discretizations (3D + 1D).
    - Crank-Nicolson method.
- More output options (some concentration and flux distributions have been implemented already).
- File-system re-arrangement (the singular helper file is getting busy, this will be a major change).
    - This is next in line, but I have to figure out how to make the code run on a PC without CUDA.
- tiff-based outputs for flux and concentration mapping (maybe).
- Multi-GPU version for HPC use (coming soon, using CUDA Cooperative Groups).
    - multi-GPU with asynchronous execution is not super efficient, works well but struggles with large number of GPUs.
    - Page locked memory is a must, but might not work well for older GPUs. 
- More solver options (different simulations have different needs).

Additionally, some experimental features might come in the near future. At this time, I cannot provide a reasonable estimate of when these will be implemented (or if they will at all). In no particular order:

- GUI:
  - Running code on the GUI has significant impacts in terms of the overall efficiency. For now, if I do make a GUI at some point, it will mainly handle the input file generation, and not the actual simulation.
  - Work has started on a gen-config style GUI. This GUI will help users configure the code. The code execution will remain a command-line only execution style, as this is primarily aimed for Linux/HPC resources.
- Higher-order discretization methods.
- For the GPU code, I will try and implement some of the most recent cuBLAS and cuSPARSE solvers.
- For the CPU code, implementation of [scaLAPACK](https://www.netlib.org/scalapack/) for solving the sparse systems. I will test those versus the already existing solvers, so we will see what works best.
  - The GPU code in 3D, with the same solver as the CPU version, can be hundreds of times faster than the CPU counterpart. In other words, with parallel computing, a mid-range GPU (like a GeFORCE RTX 3070) can be as fast as 100 CPUs (also with the bold assumption of 100% efficiency on the parallel CPU code). Therefore, the development will first focus on GPU and multi-GPU code as opposed to CPU.
- More meshing options.
  - The current meshing approach is very rudimentary. For estimating bulk properties of the domain, the meshing does not seem to be a problem. However, the local accuracy might be sacrificed in some cases. I will try to add some methods for mesh refinement in some locations.
