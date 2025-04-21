# This file obviously doesn't work, but we will make it one day, trust
cmake_minimum_required(VERSION 3.10 FATAL_ERROR)

# version information--------------------------
project(EffectiveDiffusivityFVM LANGUAGES CXX CUDA VERSION 1.0)

# Include directories -------------------------
include_directories(
    EffectiveDiffusivityFVM PUBLIC
    # "${PROJECT_SOURCE_DIR}"
    "${PROJECT_SOURCE_DIR}/src_GPU"
    "${PROJECT_SOURCE_DIR}/src"
    # "${PROJECT_SOURCE_DIR}/tests"
    # "${PROJECT_SOURCE_DIR}/src_cuda"
)

# Different methods are needed for Linux based systems and Windows
# I don't know anyone with a Mac, so no build available for that

if (WIN32)
    enable_language(CUDA)
endif (WIN32)

# Set Project Sources ----------------------------------------

set(PROJECT_SOURCES
    src_GPU/main.cu
)

# Add Executable --------------------------------------------

add_executable(
    EffectiveDiffusivityFVM
    ${PROJECT_SOURCES}
)

# Compilation options to ensure this works with CUDA + OpenMP
set_property(TARGET EffectiveDiffusivityFVM PROPERTY CUDA_SEPARABLE_COMPILATION ON)
set_property(TARGET EffectiveDiffusivityFVM PROPERTY CUDA_ARCHITECTURES 89)
target_compile_features(EffectiveDiffusivityFVM PUBLIC cxx_std_17)

# add submodels with target link executable


set(TEST_SOURCES
    src/SubModels/TransientSweepSimulatedDiffusimetry/TSSD.cu
)

add_executable(
    TSSD
    ${TEST_SOURCES}
)

# Compilation options to ensure this works with CUDA + OpenMP
set_property(TARGET TSSD PROPERTY CUDA_SEPARABLE_COMPILATION ON)
set_property(TARGET TSSD PROPERTY CUDA_ARCHITECTURES 89)
target_compile_features(TSSD PUBLIC cxx_std_17)
