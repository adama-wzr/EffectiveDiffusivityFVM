# EffectiveDiffusivityFVM

This repository is dedicated to the simulation of effective diffusivity in 2D structures via the Finite Volume Method (FVM). This approach was designed for maximum efficiency when generating large datasets for machine learning applications, thus uses the pixel resolution of the image as the base mesh for the simulation. Below is basic information on how to compile and run this code. For more detailed information about the code itself, refer to the documentation pdf.

This repository includes one code version only: it is meant to run on a CUDA-capable GPU. However, if there is a need for another mode, please contact one of the authors as it is easy to make changes (GPU vs. CPU) as they only affect the solver, not the numerical model.

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
- [stb_image](https://github.com/nothings/stb) latest version

The code has been tested on Ubuntu >= 20.04, Windows 10 and 11, and on Rocky Linux 8.7.

## GPU Compilation

With the NVIDIA suite installed properly and already added to the path, also assuming all required files are in the same folder.

```bash
nvcc Perm2D.cu
```
## Required Files

All these files have to be in the same folder (or in the path for compilation/run).

- 2D grayscale .jpg image.
- Main Deff2D file (.cpp or .cu)
- Helper Deff2D file (.h or .cuh)
- input.txt
- stb_image.h

## How to Cite

Publication is in preparation at the moment. If you need to use this code and there is no publication available yet, contact one of the authors.

## Code Authors

- Main developer: Andre Adam (The University of Kansas)
    - [ResearchGate](https://www.researchgate.net/profile/Andre-Adam-2)
    - [GoogleScholar](https://scholar.google.com/citations?hl=en&user=aP_rDkMAAAAJ)
    - [GitHub](https://github.com/adama-wzr)
- Advisor: Dr. Xianglin Li (Washingtion University in St. Louis)
    - [Website](https://xianglinli.wixsite.com/mysite)
    - [GoogleScholar](https://scholar.google.com/citations?user=8y0Vd8cAAAAJ&hl=en)

 ## Documentation

The publication is an excellent source of basic information on the formulation and validation. The documentation pdf is a more in-depth source on the mathematical formulation and code implementation, while also providing technical insight on how to run and modify the code included in this repository.

## Acknowledgements

This work wouldn't be possible without the computational time awarded as part of the following grants:

This work used Expanse(GPU) at SDSC through allocations MAT210014 and MAT230071 from the Advanced Cyberinfrastructure Coordination Ecosystem: Services & Support (ACCESS) program, which is supported by National Science Foundation grants #2138259, #2138286, #2138307, #2137603, and #2138296.

## Upcoming Changes

At this moment, there is no timeline of when these changes might come through. However, below is a list of changes that have been discussed and will be implemented sometime in the future. This list is in no particular order.

- 3D version of the code.
- CPU alternatives to the GPU code.
- HPC version with both CPU and GPU acceleration.
- GUI:
  - The actual GUI is a way to facilitate and/or guide the generation of the input file and calling the appropriate code versions based on user input.
  - The GUI won't be necessary. While code can run from the GUI itself, the code will also run without the GUI.
- Higher accuracy discretization schemes.
