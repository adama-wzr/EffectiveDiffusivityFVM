#ifndef _SOLVE2D
#define _SOLVE2D

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "cuda_runtime.h"
#include "cuda.h"
#include <omp.h>

#include <helper.cuh>   // why are datastructures in helper.h????

// kernels and stuff are also on helper.cuh for no reason

#include <datastructures.cpp>

#define CHECK_CUDA(func)                                               \
    {                                                                  \
        cudaError_t status = (func);                                   \
        if (status != cudaSuccess)                                     \
        {                                                              \
            printf("CUDA API failed at line %d with error: %s (%d)\n", \
                   __LINE__, cudaGetErrorString(status), status);      \
            return EXIT_FAILURE;                                       \
        }                                                              \
    }


/*

    Kernels:

*/

// 2D GPU Jacobi-SOR

__global__ void JI_SOR2D_kernel_PBx(
    double *A,
    double *x,
    double *b,
    double *xNew,
    long int nElements,
    int nCols,
    int nRows)
{
    unsigned int myIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int myRow = myIdx / nCols;
    int myCol = myIdx - myRow * nCols;
    double w = 2.0 / 3.0;

    if (myIdx < nElements)
    {
        double sigma = 0;
        for (int j = 1; j < 5; j++)
        {
            if (A[myIdx * 5 + j] != 0)
            {
                if (j == 1)
                {
                    if(myCol == 0)
                    {
                        // Periodic west
                        sigma += A[myIdx * 5 + j] * x[myRow * nCols + nCols - 1];
                    }
                    else
                    {
                        // Normal west
                        sigma += A[myIdx * 5 + j] * x[myIdx - 1];
                    }
                }
                else if (j == 2)
                {
                    if(myCol == nCols - 1)
                    {
                        // Periodic East
                        sigma += A[myIdx * 5 + j] * x[myRow * nCols + 0];
                    }
                    else
                    {
                        // Normal east
                        sigma += A[myIdx * 5 + j] * x[myIdx + 1];
                    }
                }
                else if (j == 3)
                {
                    sigma += A[myIdx * 5 + j] * x[myIdx + nCols];
                }
                else if (j == 4)
                {
                    sigma += A[myIdx * 5 + j] * x[myIdx - nCols];
                }
            }
        }
        xNew[myIdx] = (1.0 - w) * x[myIdx] + w / A[myIdx * 5 + 0] * (b[myIdx] - sigma);
    }
}
    

/*

    Overly specific solvers:

*/

int JI2D_PBx_GPU(double *Coeff,
                 double *RHS,
                 double *Concentration,
                 double *d_Coeff,
                 double *d_RHS,
                 double *d_Conc,
                 double *d_ConcTemp,
                 options *opts,
                 meshInfo *mesh)
{
    /*
        Function JI2D_PBx_GPU:
        Inputs:
            - pointer to coefficient matrix array
            - pointer to RHS matrix array
            - pointer to Concentration distribution array
            - pointer to device coefficient matrix
            - pointer to device right-hand side array
            - pointer to device concentration array
            - pointer to device temporary concentration array storage
            - pointer to options struct
            - pointer to mesh struct
        Outputs:
            - None

        This function will manage the host-device interactions for the Jacobi Iteration method
        in 2D, with a standard over-relaxation applied. The function will manage data transfers,
        convergence criteria, and kernel coordination. Periodic BC's in x-direction.
    */

    long int iterCount = 0;
    int threads_per_block = 128;
    int numBlocks = mesh->nElements / threads_per_block + 1;

    double pctChange = 1;
    int iterToCheck = 100;

    // copy arrays into GPU

    CHECK_CUDA(cudaMemcpy(d_Conc, Concentration,
                          sizeof(double) * mesh->nElements, cudaMemcpyHostToDevice));

    CHECK_CUDA(cudaMemcpy(d_ConcTemp, Concentration,
                          sizeof(double) * mesh->nElements, cudaMemcpyHostToDevice));

    CHECK_CUDA(cudaMemcpy(d_RHS, RHS,
                          sizeof(double) * mesh->nElements, cudaMemcpyHostToDevice));

    CHECK_CUDA(cudaMemcpy(d_Coeff, Coeff,
                          sizeof(double) * mesh->nElements * 5, cudaMemcpyHostToDevice));

    // Create Array to store temp Conc

    double *TempConc = (double *)malloc(sizeof(double) * mesh->nElements);

    memcpy(TempConc, Concentration, sizeof(double) * mesh->nElements);

    // start the main loop

    while (iterCount < opts->MAX_ITER && pctChange > opts->ConvergeCriteria)
    {
        // call kernel

        JI_SOR2D_kernel_PBx<<<numBlocks, threads_per_block>>>(d_Coeff, d_ConcTemp, d_RHS, d_Conc,
                                                          mesh->nElements, mesh->numCellsX, mesh->numCellsY);
        // check convergence

        if (iterCount % iterToCheck == 0 && iterCount != 0)
        {
            // copy array from device to host
            CHECK_CUDA(cudaMemcpy(Concentration, d_Conc, sizeof(double) * mesh->nElements, cudaMemcpyDeviceToHost));

            // compare
            double sum = 0;
            long int count = 0;

            for (int i = 0; i < mesh->nElements; i++)
            {
                if (Concentration[i] != 0)
                {
                    sum += fabs((Concentration[i] - TempConc[i]) / Concentration[i]);
                    count++;
                }
            }
            // calculate the change
            pctChange = sum / count;
            // copy memory to temp conc
            memcpy(TempConc, Concentration, sizeof(double) * mesh->nElements);
        }

        if (opts->SteadyStateFlag && iterCount %  100000 == 0)
        {
            printf("Iter %ld, Conv %1.3e, Target %1.3e\n", iterCount, pctChange, opts->ConvergeCriteria);
        }

        // update d_Conc = d_ConcTemp

        CHECK_CUDA(cudaMemcpy(d_ConcTemp, d_Conc, sizeof(double) * mesh->nElements, cudaMemcpyDeviceToDevice));

        // increment
        iterCount++;
    }

    // copy the solution

    CHECK_CUDA(cudaMemcpy(Concentration, d_ConcTemp,
                          sizeof(double) * mesh->nElements, cudaMemcpyDeviceToHost));

    // store info to print

    mesh->conv = pctChange;
    mesh->iterCount = iterCount;

    // free memory

    free(TempConc);

    return 0;
}


int JI2D_TransientUpdate_PBx(
             double     *RHS,
             double     *Concentration,
             double     *d_Coeff,
             double     *d_RHS,
             double     *d_Conc,
             double     *d_ConcTemp,
             options    *opts,
             meshInfo   *mesh)
{
    /*
        Function JI2D_TransientUpdate_PBx:
        Inputs:
            - pointer to RHS matrix array
            - pointer to Concentration distribution array
            - pointer to device coefficient matrix
            - pointer to device right-hand side array
            - pointer to device concentration array
            - pointer to device temporary concentration array storage
            - pointer to options struct
            - pointer to mesh struct
        Outputs:
            - None

        This function will manage the host-device interactions for the Jacobi Iteration method
        in 2D, with a standard over-relaxation applied. The function will manage data transfers,
        convergence criteria, and kernel coordination. The difference between this one and JI2D_SOR
        is that this one has less memory transfers, as a lot of the information is already in the GPU.
    */

    long int iterCount = 0;
    int threads_per_block = 128;
    int numBlocks = mesh->nElements / threads_per_block + 1;

    double pctChange = 1;
    int iterToCheck = 100;

    // Update RHS on GPU 

    CHECK_CUDA(cudaMemcpy(d_RHS, RHS,
                          sizeof(double) * mesh->nElements, cudaMemcpyHostToDevice));

    // Create Array to store temp Conc

    double *TempConc = (double *)malloc(sizeof(double) * mesh->nElements);

    memcpy(TempConc, Concentration, sizeof(double) * mesh->nElements);

    // start the main loop

    while (iterCount < opts->MAX_ITER && pctChange > opts->ConvergeCriteria)
    {
        // call kernel

        JI_SOR2D_kernel_PBx<<<numBlocks, threads_per_block>>>(d_Coeff, d_ConcTemp, d_RHS, d_Conc,
                                                          mesh->nElements, mesh->numCellsX, mesh->numCellsY);
        // check convergence

        if (iterCount % iterToCheck == 0 && iterCount != 0)
        {
            // copy array from device to host
            CHECK_CUDA(cudaMemcpy(Concentration, d_Conc, sizeof(double) * mesh->nElements, cudaMemcpyDeviceToHost));

            // compare
            double sum = 0;
            long int count = 0;

            for (int i = 0; i < mesh->nElements; i++)
            {
                if (Concentration[i] != 0)
                {
                    sum += fabs((Concentration[i] - TempConc[i]) / Concentration[i]);
                    count++;
                }
            }
            // calculate the change
            pctChange = sum / count;
            // copy memory to temp conc
            memcpy(TempConc, Concentration, sizeof(double) * mesh->nElements);
        }

        // update d_Conc = d_ConcTemp

        CHECK_CUDA(cudaMemcpy(d_ConcTemp, d_Conc, sizeof(double) * mesh->nElements, cudaMemcpyDeviceToDevice));

        // increment
        iterCount++;
    }

    // copy the solution

    CHECK_CUDA(cudaMemcpy(Concentration, d_ConcTemp,
                          sizeof(double) * mesh->nElements, cudaMemcpyDeviceToHost));

    // store info to print

    mesh->conv = pctChange;
    mesh->iterCount = iterCount;

    // free memory

    free(TempConc);

    return 0;
}



#endif