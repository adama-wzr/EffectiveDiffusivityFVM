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
    "${PROJECT_SOURCE_DIR}/src/SubModels/Migration"
    "${PROJECT_SOURCE_DIR}/src/SubModels/TransientSweepSimulatedDiffusimetry"
    "${PROJECT_SOURCE_DIR}/lib/"
    # "${PROJECT_SOURCE_DIR}/tests"
    # "${PROJECT_SOURCE_DIR}/src_cuda"
)

# Different methods are needed for Linux based systems and Windows
# I don't know anyone with a Mac, so no build available for that

if (WIN32)
    enable_language(CUDA)
endif (WIN32)

if (UNIX)
    find_package(OpenMP REQUIRED)
    find_package(CUDA REQUIRED)
    set(CMAKE_CUDA_COMPILER "nvcc")
endif (UNIX)

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

# target_link_libraries(TPMS-Characterization PRIVATE OpenMP::OpenMP_CXX)

if (UNIX)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=-fopenmp")

    target_link_libraries(EffectiveDiffusivityFVM PRIVATE OpenMP::OpenMP_CXX)
endif (UNIX)



# add submodels with target link executable
include_directories(
    TSSD PUBLIC
    # "${PROJECT_SOURCE_DIR}"
    "${PROJECT_SOURCE_DIR}/src_GPU"
    "${PROJECT_SOURCE_DIR}/src"
    "${PROJECT_SOURCE_DIR}/src/SubModels/Migration"
    "${PROJECT_SOURCE_DIR}/src/SubModels/TransientSweepSimulatedDiffusimetry"
    "${PROJECT_SOURCE_DIR}/lib/"
    # "${PROJECT_SOURCE_DIR}/tests"
    # "${PROJECT_SOURCE_DIR}/src_cuda"
)

set(TEST_SOURCES
    src/SubModels/TransientSweepSimulatedDiffusimetry/TSSD.cu
)

add_executable(
    TSSD
    ${TEST_SOURCES}
)

if (UNIX)
    target_link_libraries(TSSD PRIVATE OpenMP::OpenMP_CXX)
endif(UNIX)

# Compilation options to ensure this works with CUDA + OpenMP
set_property(TARGET TSSD PROPERTY CUDA_SEPARABLE_COMPILATION ON)
set_property(TARGET TSSD PROPERTY CUDA_ARCHITECTURES 86 89)
target_compile_features(TSSD PUBLIC cxx_std_17)
