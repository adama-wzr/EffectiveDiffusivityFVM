/*

License information:

MIT License

Copyright (c) 2025 Andre Adam

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.


Code contributors:

Andre Adam. 

Last Update: 
01/24/2025

*/

#ifndef _HELPER
#define _HELPER

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <vector>
#include <iostream>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#include <stdbool.h>
#include <fstream>
#include <cfloat>
#include <set>
#include <string>
#include "cuda_runtime.h"
#include "cuda.h"
#include <omp.h>

// CUDA CHECK ERROR

#define CHECK_CUDA(func)                                                       \
{                                                                              \
    cudaError_t status = (func);                                               \
    if (status != cudaSuccess) {                                               \
        printf("CUDA API failed at line %d with error: %s (%d)\n",             \
               __LINE__, cudaGetErrorString(status), status);                  \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}

// Data structure definitions

typedef struct
{
    double *DC;              // array with diffusion coefficients
    unsigned char *DC_TH;    // Upper limit threshold for phase differentiation when reading jpg's
    int numDC;               // number of diffusion coefficients
    int MeshIncreaseX;       // Mesh refinement in x-direction
    int MeshIncreaseY;       // Mesh refinement in y-direction
    int MeshIncreaseZ;       // Mesh refinement in the z-direction
    double CLeft;            // Concentration of trace species in left boundary
    double CRight;           // Concentration of trace species in right boundary
    long int MAX_ITER;       // Max iterations
    double ConvergeCriteria; // Convergence Criteria
    char *inputFilename;     // Input filename
    int printOut;            // Flag to print output or not
    char *outputFilename;    // Output filename
    int printCmap;           // print concentration map (true/false) flag
    char *CMapName;          // Concentration map name
    int verbose;             // verbose flag
    int BatchFlag;           // Batch flag
    int NumImgBatch;         // Number of images in the batch
    char SteadyStateFlag;    // steady state simulation or time dependent ?
    int height;              // height in number of pixels
    int width;               // width in number of pixels
    int depth;               // depth in number of pixels
    int nD;                  // number of dimensions
    int nThreads;            // number of threads
    char inputType;          // Input format for 3D simulations (0 default .csv, 1 is stack)
    int useGPU;              // Use GPU or not?
    int nGPU;                // number of GPUs
} options;

typedef struct
{
    int Width;
    int Height;
    int Depth;
    double time;
    unsigned char *target_data;
    double deff;
    bool PathFlag;
    double conv;
} simulationInfo;

typedef struct
{
    int numCellsX;
    int numCellsY;
    int numCellsZ;
    long int nElements;
    double dx;
    double dy;
    double dz;
} meshInfo;

// Define coords for Flood Fill

typedef std::tuple<int, int, int> coord;

typedef std::pair<int, int> coordPair;

/*

    GPU Kernels

*/

// 3D GPU SOR

__global__ void JI_SOR3D_kernel(
    double *A,
    double *x,
    double *b,
    double *xNew,
    long int nElements,
    int nCols,
    int nRows)
{
    unsigned int myIdx = blockIdx.x * blockDim.x + threadIdx.x;
    double w = 2.0 / 3.0;

    if (myIdx < nElements)
    {
        double sigma = 0;
        for (int j = 1; j < 7; j++)
        {
            if (A[myIdx * 7 + j] != 0)
            {
                if (j == 1)
                {
                    sigma += A[myIdx * 7 + j] * x[myIdx - 1];
                }
                else if (j == 2)
                {
                    sigma += A[myIdx * 7 + j] * x[myIdx + 1];
                }
                else if (j == 3)
                {
                    sigma += A[myIdx * 7 + j] * x[myIdx + nCols];
                }
                else if (j == 4)
                {
                    sigma += A[myIdx * 7 + j] * x[myIdx - nCols];
                }
                else if (j == 5)
                {
                    sigma += A[myIdx * 7 + j] * x[myIdx + nCols * nRows];
                }
                else if (j == 6)
                {
                    sigma += A[myIdx * 7 + j] * x[myIdx - nCols * nRows];
                }
            }
        }
        xNew[myIdx] = (1.0 - w) * x[myIdx] + w / A[myIdx * 7 + 0] * (b[myIdx] - sigma);
    }
}

// 2D GPU Jacobi-SOR

__global__ void JI_SOR2D_kernel(
    double *A,
    double *x,
    double *b,
    double *xNew,
    long int nElements,
    int nCols,
    int nRows)
{
    unsigned int myIdx = blockIdx.x * blockDim.x + threadIdx.x;
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
                    sigma += A[myIdx * 5 + j] * x[myIdx - 1];
                }
                else if (j == 2)
                {
                    sigma += A[myIdx * 5 + j] * x[myIdx + 1];
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

    Functions handling user input:

*/


int printOptions(options* opts)
{
    /*
		Function printOptions:
        Inputs:
            - pointer to opts struct
        Outputs:
            - None
        
        The function is only called when Verbose = true. It will
        print the user entered options to the command line. For saving cmd output
        into a text file, please do so externally.
	*/

    printf("--------------------------------------\n\n");
    printf("Current selected options:\n\n");
    printf("--------------------------------------\n");
    printf("Number of Dimensions: %d\n", opts->nD);
    printf("InputType = %d\n", opts->inputType);
    if(opts->BatchFlag)
    {
        printf("Running a bacth of size = %d\n", opts->NumImgBatch);
    } else
    {

        // Options related to input type

        if(opts->inputType == 0)
        {
            printf("Input Method = csv\n");
            printf("Filename = %s\n", opts->inputFilename);
            printf("Structure Width  = %d\n", opts->width);
            printf("Structure Height = %d\n", opts->height);
            if(opts->nD == 3) printf("Structure Depth  = %d\n", opts->depth);
            printf("Diffusion Coefficients:\n");
            for(int i = 0; i < opts->numDC; i++)
            {
                printf("D%d = %1.3e\n",i+1, opts->DC[i]);
            }

            // If nD = 2 make sure to set input type to 2
        }
        else if(opts->inputType == 1)
        {
            printf("Input Method = jpg stack\n");
            printf("Stack Size = %d\n", opts->depth);
            printf("Diffusion Coefficients and Processing Thresholds:\n");
            for(int i = 0; i < opts->numDC; i++)
            {
                printf("D%d = %1.3e\n",i+1, opts->DC[i]);
                printf("D_TH%d = %d\n", i+1, opts->DC_TH[i]);
            }
        }
        else if(opts->inputType == 2)
        {
            printf("Filename = %s\n",opts->inputFilename);
            printf("Diffusion Coefficients and Processing Thresholds:\n");
            for(int i = 0; i < opts->numDC; i++)
            {
                printf("D%d = %1.3e\n",i+1, opts->DC[i]);
                printf("D_TH%d = %d\n", i+1, opts->DC_TH[i]);
            }
        }

        // check steady-state flag

        if(opts->SteadyStateFlag == 1)
        {
            printf("Steady-State Simulation Mode Selected\n");
        }
        else if(opts->SteadyStateFlag == 0)
        {
            printf("Time-dependent Simulation Selected\n");
            printf("Time discretization:\n");
            /*
                Add here different time discretization requirements
            */
        }

        // mesh amplificaiton

        printf("Mesh Refine X = %d\n", opts->MeshIncreaseX);
        printf("Mesh Refine Y = %d\n", opts->MeshIncreaseY);
        if (opts->nD == 3)
        {
            printf("Mesh Refine Z = %d\n", opts->MeshIncreaseZ);
        }

        // convergence

        printf("Max. Iterations: %ld\n", opts->MAX_ITER);
        printf("Convergence: %1.3e\n", opts->ConvergeCriteria);

        // options related to output printing

        if(opts->printCmap == 1)
        {
            printf("CMAP Name: %s\n", opts->CMapName);
        }
        if(opts->printOut == 1)
        {
            printf("Output File Name: %s\n", opts->outputFilename);
        }
        
        // Options related to multi-threading/GPU
        
        if(opts->useGPU == 1)
        {
            printf("Using %d GPU(s)\n", opts->nGPU);
        }
        else
        {
            printf("Number of Threads = %d\n", opts->nThreads);
        }
        
    }

    printf("--------------------------------------\n\n");
    return 0;
}


void readInputGeneral(char* FileName, options* opts){

	/*
		readInputGeneral Function:
		Inputs:
			- FileName: pointer to where the input file name is stored.
			- struct options: pass a struct with the options.
		Outputs: None

		Function reads the input file and stores the options in the opts struct.
	*/

    // initiate necessary variables for input reading
    std::string myText;
    
    char tempC[1000];
	double tempD;
	char tempFilenames[1000];
	std::ifstream InputFile(FileName);

    // initiate arrays for storing names of input/output files

    opts->inputFilename  =   (char *)malloc(1000*sizeof(char));
    opts->outputFilename =   (char*)malloc(1000*sizeof(char));
	opts->CMapName       =   (char*)malloc(1000*sizeof(char));
    
    // variables for reading diffusion coefficients (DC) and the thresholds (DC_TH)
    
    opts->numDC = 0;
    char tempDC[20];
    char tempDC_TH[20];
    int DC_read = 0;
    int DC_TH_read = 0;

    // Default values set here

    opts->MeshIncreaseX = 1;
    opts->MeshIncreaseY = 1;
    opts->MeshIncreaseZ = 1;

    opts->BatchFlag = 0;
    opts->inputType = 0;

    opts->nThreads = 1;

    opts->useGPU = 0;
    opts->nGPU = 1;

    /*
    --------------------------------------------------------------------------------

    If anybody has a better idea of how to parse inputs please let me know.
    Eventually I'm hoping the GUI will replace a lot of this code.

    --------------------------------------------------------------------------------
    */


    while(std::getline(InputFile, myText))
    {
        sscanf(myText.c_str(), "%s %lf", tempC, &tempD);
        if(strcmp(tempC, "nD:") == 0)
        {
            opts->nD = (int)tempD;
        }
        else if(strcmp(tempC, "numDC:") == 0)
        {
            opts->numDC = (int)tempD;
            // allocate the space in memory
            opts->DC = (double *)malloc(opts->numDC*sizeof(double));
            opts->DC_TH = (unsigned char *)malloc(opts->numDC*sizeof(char));
            // set memory
            memset(opts->DC, 0, opts->numDC*sizeof(double));
            memset(opts->DC_TH, 0, opts->numDC*sizeof(char));
            DC_read++;
            DC_TH_read++;
        }
        else if(strcmp(tempC, tempDC) == 0)
        {
            opts->DC[DC_read - 1] = tempD;
            DC_read++;
        }
        else if(strcmp(tempC, tempDC_TH) == 0)
        {
            opts->DC_TH[DC_TH_read - 1] = (unsigned char)tempD;
            DC_TH_read++;
        }
        else if(strcmp(tempC, "MeshAmpX:") == 0)
        {
            opts->MeshIncreaseX = (int)tempD;
        }
        else if(strcmp(tempC, "MeshAmpY:") == 0)
        {
            opts->MeshIncreaseY = (int)tempD;
        }
        else if(strcmp(tempC, "MeshAmpZ:") == 0)
        {
            opts->MeshIncreaseZ = (int)tempD;
        }
        else if(strcmp(tempC, "InputName:") == 0)
        {
            sscanf(myText.c_str(), "%s %s", tempC, tempFilenames);
	 		strcpy(opts->inputFilename, tempFilenames);
        }
        else if(strcmp(tempC, "OutputName:") == 0)
        {
	 		sscanf(myText.c_str(), "%s %s", tempC, tempFilenames);
	 		strcpy(opts->outputFilename, tempFilenames);
        }
        else if(strcmp(tempC, "printCMap:") == 0){
	 		opts->printCmap = (int)tempD;
	 	}
        else if(strcmp(tempC, "CMapName:") == 0)
        {
	 		sscanf(myText.c_str(), "%s %s", tempC, tempFilenames);
	 		strcpy(opts->CMapName, tempFilenames);
	 	}
        else if(strcmp(tempC, "Convergence:") == 0)
        {
	 		opts->ConvergeCriteria = tempD;
	 	}
        else if(strcmp(tempC, "MaxIter:") == 0)
        {
	 		opts->MAX_ITER = (long int)tempD;
	 	}
        else if(strcmp(tempC, "Verbose:") == 0)
        {
	 		opts->verbose = (int)tempD;
	 	}
        else if(strcmp(tempC, "RunBatch:") == 0)
        {
	 		opts->BatchFlag = (int)tempD;
	 	}
        else if(strcmp(tempC, "NumImages:") == 0)
        {
	 		opts->NumImgBatch = (int)tempD;
        }
        else if(strcmp(tempC, "CL:") == 0)
        {
            opts->CLeft = tempD;
        }
        else if(strcmp(tempC, "CR:") == 0)
        {
            opts->CRight = tempD;
        }
        else if(strcmp(tempC, "SS:") == 0)
        {
            opts->SteadyStateFlag = (char)tempD;
        }
        else if(strcmp(tempC, "width:") == 0)
        {
            opts->width = (int)tempD;
        }
        else if(strcmp(tempC, "height:") == 0)
        {
            opts->height = (int)tempD;
        }
        else if(strcmp(tempC, "depth:") == 0)
        {
            opts->depth = (int)tempD;
        }
        else if(strcmp(tempC, "inputType:") == 0)
        {
            opts->inputType = (char)tempD;
        }
        else if(strcmp(tempC, "printOutput:") == 0)
        {
            opts->printOut = (int)tempD;
        }
        else if(strcmp(tempC,"nThreads:") == 0)
        {
            opts->nThreads = (int)tempD;
        }
        else if(strcmp(tempC,"useGPU:") == 0)
        {
            opts->useGPU = (int)tempD;
        }
        else if(strcmp(tempC, "nGPU:") == 0)
        {
            opts->nGPU = (int)tempD;
        }

        // Update the number of expected diffusion coefficients and thresholding for image
        // processing

        if (DC_read <= opts->numDC)     sprintf(tempDC, "D%d:", DC_read);
        if (DC_TH_read <= opts->numDC)  sprintf(tempDC_TH, "D_TH%d:", DC_TH_read);
    }
    return;
}

int readCSV3D(options* opts, char* simObject)
{
    /*
        Function readCSSV3D:
        Inputs:
            - pointer to options data structure
            - pointer to simObject array, where the structure will be saved
                with the appropriate flags.
        Output:
            - None
        
        The function will populate the simObject array according to the data in the
        input .csv file.

    */
    // read structure
    int height, width, depth;
    long int nElements;

    height = opts->height;
    width = opts->width;
    depth = opts->depth;
    nElements = height*width*depth;

    // declare arrays to hold coordinates for all specified phases

    int *x = (int *)malloc(sizeof(int)*nElements);
    int *y = (int *)malloc(sizeof(int)*nElements);
    int *z = (int *)malloc(sizeof(int)*nElements);
    int *phase = (int *)malloc(sizeof(int)*nElements);

    // Read structure file

    FILE *target_data;

    target_data = fopen(opts->inputFilename, "r");

    // check if file exists

    if (target_data == NULL){
        fprintf(stderr, "Error reading file. Exiting program.\n");
        return 1;
    }

    char header[20];

    fscanf(target_data, "%c,%c,%c,%s", &header[0], &header[1], &header[2], &header[3]);

    // if (opts->verbose) printf("Header = %s\n", header);      // debug mainly

    // read coordinates from input file

    size_t count = 0;

    while (fscanf(target_data, "%d,%d,%d,%d", &x[count], &y[count], &z[count], &phase[count]) == 4)
    {
        count++;
    }

    long int index = 0;

    for(long int i = 0; i<count; i++)
    {
        index = z[i]*height*width + y[i]*width + x[i];
        simObject[index] = phase[i];    // the diffusivities later are assigned based on this number
    }

    // memory management

    free(x);
    free(y);
    free(z);
    free(phase);

    return 0;
}


int readImg2D(options* opts, meshInfo* mesh, char*& simObject)
{
    /*
        Function readImg2D:
        Inputs:
            - Pointer to options struct.
            - Pointer to mesh struct.
            - Pointer to simObject array, where image will be stored temporarily.
        Outputs:
            - None

        Function will read the target grayscale image for 2D simulation.

        NOTE: Notation char*& is only valid in C++, not in C.
    */
    // read image and store data

    int nChannels;
    unsigned char* target_data = stbi_load(opts->inputFilename, &opts->width,
                                &opts->height, &nChannels, 1);

    // Terminate if n channel != 1

    if(nChannels != 1)
    {
        printf("Number of Channels of input image != 1\n");
        printf("Exiting with error\n");
        return 1;
    }

    // store image size information

    mesh->numCellsX = opts->width*opts->MeshIncreaseX;
    mesh->numCellsY = opts->height*opts->MeshIncreaseY;
    mesh->numCellsZ = 1;
    mesh->nElements = mesh->numCellsX*mesh->numCellsY;

    // dynamically allocate the simObject array given the data

    simObject = (char *)malloc(sizeof(char)*mesh->nElements);

    // apply the simple thresholding and separate different phases

    for(int row = 0; row < mesh->numCellsY; row++)
    {
        for(int col = 0; col < mesh->numCellsX; col++)
        {
            // Account for mesh amplification
            int targetRow = row / opts->MeshIncreaseY;
            int targetCol = col / opts->MeshIncreaseX;
            int targetIdx = targetRow*opts->width + targetCol;

            // thresholding
            for(int p = 0; p < opts->numDC; p++)
            {
                if (target_data[targetIdx] <= opts->DC_TH[p])
                {
                    simObject[row*mesh->numCellsX + col] = p;
                    break;
                }
            }
        }
    }

    // Memory management

    free(target_data);

    return 0;
}

/*

    Auxiliary Functions:

*/

double WeightedHarmonicMean(double w1, double w2, double x1, double x2)
{
    /*
        WeightedHarmonicMean Function:
        Inputs:
            - w1: weight of the first number
            - w2: weight of the second number
            - x1: first number of the mean
            - x2: second number of the mean
        Outputs:
            - H: weighted harmonic mean
        
        The function will calculate the weighted harmonic mean of two numbers, x1 and x2,
        subject to weights w1 and w2.
    */
    double H = (w1 + w2)/(w1/x1 + w2/x2);
    return H;
}

/*

    Setting DC's and BC's:

*/


int SetDC2D(options* opts, meshInfo* mesh, double* DC, char* simObject)
{
    /*
        Function SetDC2D:
        Inputs:
            - pointer to options struct
            - pointer to mesh struct
            - pointer to DC, an array where the diffusion coefficients will be stored
            - pointer to simObject, where structure and phase information was originally stored.
        Outputs:
            - None.
        The function will set the diffusion coefficient on DC, according to the 
        phases given by the simObject array and user options.
    */

    for(int i = 0; i < mesh->numCellsY; i++)
    {
        for(int j = 0; j < mesh->numCellsX; j++)
        {
            // determine local phase
            int localPhase = simObject[i*mesh->numCellsX + j];
            // store diffusion coefficient of the local phase
            DC[i*mesh->numCellsX + j] = opts->DC[localPhase];
        }
    }
    return 0;
}

int SetDC3D(options* opts, meshInfo* mesh, double* DC, char* simObject)
{
    /*
        Function SetDC3D:
        Inputs:
            - pointer to options struct
            - pointer to mesh struct
            - pointer to DC, an array where the diffusion coefficients will be stored
            - pointer to simObject, where structure and phase information was originally stored.
        Outputs:
            - None.
        The function will set the diffusion coefficient on DC, according to the 
        phases given by the simObject array and user options.
    */

    for(int k = 0; k<mesh->numCellsZ; k++)
    {
        for(int i = 0; i<mesh->numCellsY; i++)
        {
            for(int j = 0; j<mesh->numCellsX; j++)
            {
                // index for original array
                int targetRow = i / opts->MeshIncreaseY;
                int targetCol = j / opts->MeshIncreaseX;
                int targetSlice = k / opts->MeshIncreaseZ;
                int targetIndex = targetSlice*opts->height*opts->width 
                            + targetRow*opts->width + targetCol;
                // index for array with meshAmp
                int index = k*mesh->numCellsX*mesh->numCellsY
                            +i*mesh->numCellsX + j;
                // Identify phase and diffusion coefficient
                int localPhase = simObject[targetIndex];
                double localDC = opts->DC[localPhase];
                // Store data
                DC[index] = localDC;
            }
        }
    }

    return 0;
}


int SetBC_DeffSetup2D(options* opts, meshInfo* mesh, int* BC, double* BC_Value)
{
    /*
        Function SetBC_DeffSetup2D:
        Inputs:
            - pointer to options struct
            - pointer to mesh struct
            - pointer to BC classification array
            - pointer to BC_value array (value of BC for Neumann or Dirichlet)
        Output:
            - None
        
        The function will classify the entire BC array with 2 Dirichlet conditions on the left
        and right, while all other boundaries will be set to zero flux Neumann boundaries.
    */
    // Set some variables to help
    int nCols, nRows;
    nCols = mesh->numCellsX + 2;
    nRows = mesh->numCellsY + 2;

    // On the BC array, we need to classify right and left as Dirichlet, 
    // and assign the values on BC_Value
    // All the Neumann condition have to be assigned on BC, 
    // but no change in BC_Value is required (since this setup has impermeable walls).

    int right, left, top, bottom;
    // set col values for right and left
    left = 0;
    right = nCols - 1;

    // set row values for top and bottom
    top = 0;
    bottom = nRows - 1;

    // set Dirichlet boundaries only if DC[i] != 0

    for(int i = 0; i < nRows; i++)
    {

        // right side
        BC[i*nCols + right] = 1;                    // Dirichlet
        BC_Value[i*nCols + right] = opts->CRight;

        // left side
        BC[i*nCols + left] = 1;                     // Dirichlet
        BC_Value[i*nCols + left] = opts->CLeft;

    }

    // set Neumann boundaries

    for(int j = 0; j < nCols; j++)
    {
        // top
        BC[top*nCols + j] = 2;
        
        // bottom
        BC[bottom*nCols + j] = 2;
    }

    return 0;
}



int SetBC_DeffSetup3D(options* opts, meshInfo* mesh, int* BC, double* BC_Value)
{
    /*
        Function SetBC_DeffSetup3D:
        Inputs:
            - pointer to options struct
            - pointer to mesh struct
            - pointer to BC classification array
            - pointer to BC_value array (value of BC for Neumann or Dirichlet)
        Output:
            - None
        
        The function will classify the entire BC array with 2 Dirichlet conditions on the left
        and right, while all other boundaries will be set to zero flux Neumann boundaries.
    */
    // Set some variables to help
    int nCols, nRows, nSlices;
    nCols = mesh->numCellsX + 2;
    nRows = mesh->numCellsY + 2;
    nSlices = mesh->numCellsZ + 2;
    // On the BC array, we need to classify right and left as Dirichlet, and assign the values
    // on BC_Value
    // All the Neumann condition have to be assigned on BC, but no change in BC_Value is required.

    int right, left, top, bottom, front, back;

    left = 0;
    right = nCols - 1;

    top = 0;
    bottom = nRows - 1;

    front = 0;
    back = nSlices - 1;

    // right and left boundaries (Dirichlet)

    for(int row = 0; row < nRows; row++)
    {
        for(int slice = 0; slice < nSlices; slice++)
        {
            long int index1 = slice*nRows*nCols + row*nRows + left;
            long int index2 = slice*nRows*nCols + row*nRows + right;
            // Left boundary
            BC[index1] = 1;         // Dirichlet flag
            BC_Value[index1] = opts->CLeft;
            // Rigth boundary
            BC[index2] = 1;
            BC_Value[index2] = opts->CRight;
        }
    }

    // Top and Bottom boundaries (Neumann)

    for(int slice = 0; slice < nSlices; slice++)
    {
        for(int col = 0; col < nCols; col++)
        {
            long int index1 = slice*nRows*nCols + top*nCols + col;
            long int index2 = slice*nRows*nCols + bottom*nCols + col;
            // Top
            BC[index1] = 2;
            // Bottom
            BC[index2] = 2;
        }
    }

    // Back and Front boundaries (Neumann)

    for(int row = 0; row < nRows; row++)
    {
        for(int col = 0; col < nCols; col++)
        {
            int index1 = front*nRows*nCols + row*nCols + col;
            int index2 = back*nRows*nCols + row*nCols + col;
            // Front
            BC[index1] = 2;
            // Back
            BC[index2] = 2;
        }
    }

    return 0;
}


int FloodFill2D_DeffSetup(meshInfo* mesh, int* BC, double* DC)
{
    /*
        FloddFill2D_DeffSetup function:
        Inputs:
            - pointer to mesh struct
            - pointer to array with BC's
            - pointer to array with DC's
        Outputs:
            - None
        
        The function will search the domain, and will set all DC values that are too
        low to a Neumann BC with zero flux. Non-participating media will also be flagged
        accordingly.
    */

    char* Domain = (char *)malloc(mesh->nElements*sizeof(char));

    // Initialize all the impermeable matter in the domain:

    for(long int index = 0; index < mesh->nElements; index++)
    {
        int row = index/mesh->numCellsX;
        int col = index - row*mesh->numCellsX;
        
        long int indexBC = (row + 1)*(mesh->numCellsX + 2) + (col + 1);
        if(DC[index] == 0)
        {
            Domain[index] = 0;
            BC[indexBC] = 2;
        }
        else
        {
            Domain[index] = -1;
        }
    }

    // Find permeable boundaries, add to open list

    std::set<coordPair> cList;

    int left = 0;
    int right = mesh->numCellsX - 1;

    for(int row = 0; row < mesh->numCellsY; row++)
    {
        // set right
        if(Domain[row*mesh->numCellsX + left] == -1)
        {
            Domain[row*mesh->numCellsX + left] = 0;
            cList.insert(std::pair(left, row));
        }
        // set right
        if(Domain[row*mesh->numCellsX + right] == -1)
        {
            Domain[row*mesh->numCellsX + right] = 0;
            cList.insert(std::pair(right, row));
        }
    }

    // Search full domain

    while(!cList.empty())
    {
        // pop first item on the list
        coordPair pop = *cList.begin();

        // remove from open list
        cList.erase(cList.begin());

        // read coordinates

        int col = pop.first;
        int row = pop.second;

        /*
            We need to check North, South, East, and West for more fluid:
            
            North = col + 0, row - 1
            South = col + 0, row + 1
            East  = col + 1, row + 0
            West  = col - 1, row + 0
        
            Note that diagonals are not considered a connection.
            This code assumes no periodic boundary conditions (currently).
        */
        int tempRow, tempCol;
        long int tempIndex;

        // North

        tempCol = col;

        if(row > 0)
        {
            tempRow = row - 1;
            tempIndex = tempRow*mesh->numCellsX + tempCol;
            if (Domain[tempIndex] == -1)
            {
                Domain[tempIndex] = 0;
                cList.insert(std::pair(tempCol, tempRow));
            }
        }

        // South

        tempCol = col;

        if(row < mesh->numCellsY - 1)
        {
            tempRow = row + 1;
            tempIndex = tempRow*mesh->numCellsX + tempCol;
            if (Domain[tempIndex] == -1)
            {
                Domain[tempIndex] = 0;
                cList.insert(std::pair(tempCol, tempRow));
            }
        }

        // West

        tempRow = row;

        if(col > 0)
        {
            tempCol = col - 1;
            tempIndex = tempRow*mesh->numCellsX + tempCol;
            if(Domain[tempIndex] == -1)
            {
                Domain[tempIndex] = 0;
                cList.insert(std::pair(tempCol, tempRow));
            }
        }

        // East

        tempRow = row;

        if(col < mesh->numCellsX - 1)
        {
            tempCol = col + 1;
            tempIndex = tempRow*mesh->numCellsX + tempCol;
            if(Domain[tempIndex] == -1)
            {
                Domain[tempIndex] = 0;
                cList.insert(std::pair(tempCol, tempRow));
            }
        }

        // end while
    }

    // Every flag that is still -1 means a non-participating media

    for(int index = 0; index < mesh->nElements; index++)
    {
        // Skip participating media
        if(Domain[index] != -1) continue;

        int row = index/mesh->numCellsX;
        int col = index - row*mesh->numCellsX;
        
        long int indexBC = (row + 1)*(mesh->numCellsX + 2) + (col + 1);

        // Set BC of non-participating media

        BC[indexBC] = -1;
    }
    
    // memory management
    free(Domain);


    return 0;
}


int FloodFill3D_DeffSetup(meshInfo* mesh, int* BC, double* DC)
{
    /*
        FloddFill3D_DeffSetup function:
        Inputs:
            - pointer to mesh struct
            - pointer to array with BC's
            - pointer to array with DC's
        Outputs:
            - None
        
        The function will search the domain, and will set all DC values that are too
        low to a Neumann BC with zero flux. Non-participating media will also be flagged
        accordingly.
    */

    char* Domain = (char *)malloc(mesh->nElements*sizeof(char));

    // Initialize all the impermeable matter in the domain:

    for(long int index = 0; index < mesh->nElements; index++)
    {
        int slice = index/(mesh->numCellsX*mesh->numCellsY);
        int row = (index - slice*mesh->numCellsX*mesh->numCellsY)/mesh->numCellsX;
        int col = (index - slice*mesh->numCellsX*mesh->numCellsY - row*mesh->numCellsX);
        long int indexBC = (slice + 1)*(mesh->numCellsX + 2)*(mesh->numCellsY + 2) +
                        (row + 1)*(mesh->numCellsX + 2) + (col + 1);
        if(DC[index] == 0)
        {
            Domain[index] = 1;
            BC[indexBC] = 2;    // set BC to Neumann
        } else
        {
            Domain[index] = -1;
        }
    }

    // Find Fluid in both boundaries, add to open list

    std::set<coord> cList;

    int left = 0;
    int right = mesh->numCellsX - 1;

    for(int row = 0; row < mesh->numCellsY; row++)
    {
        for(int slice = 0; slice < mesh->numCellsZ; slice++)
        {
            long int indexL = slice*mesh->numCellsX*mesh->numCellsY + row*mesh->numCellsX + left;
            long int indexR = slice*mesh->numCellsX*mesh->numCellsY + row*mesh->numCellsX + right;
            // set left
            if(Domain[indexL] == -1)
            {
                Domain[indexL] = 0;
                cList.insert(std::tuple(left, row, slice));
            }
            // set right
            if(Domain[indexR] == -1)
            {
                Domain[indexR] = 0;
                cList.insert(std::tuple(right, row, slice));
            }
        }
    }

    // Search Full Domain

    while(!cList.empty())
    {
        // pop first item on the list
        coord pop = *cList.begin();

        // remove from open list
        cList.erase(cList.begin());

        // get coordinates from the list
        int col     = std::get<0>(pop);
        int row     = std::get<1>(pop);
        int slice   = std::get<2>(pop);

        /*
            We need to check North, South, East, West, Back, and Front for more fluid:
            
            North = col + 0, row - 1, slice + 0
            South = col + 0, row + 1, slice + 0
            East  = col + 1, row + 0, slice + 0
            West  = col - 1, row + 0, slice + 0
            Front = col + 0, row + 0, slice - 1
            Back  = col + 0, row + 0, slice + 1
        
            Note that diagonals are not considered a connection.
            This code assumes no periodic boundary conditions (currently).
        */

        int tempRow, tempCol, tempSlice;
        long int tempIndex;
        
        // North

        tempCol = col;
        tempSlice = slice;

        if(row != 0)
        {
            tempRow = row - 1;
            tempIndex = tempSlice*mesh->numCellsX*mesh->numCellsY 
                        + tempRow*mesh->numCellsX + tempCol;
            if (Domain[tempIndex] == -1)
            {
                Domain[tempIndex] = 0;
                cList.insert(std::tuple(tempCol, tempRow, tempSlice));
            }
        }

        // South

        if(row != mesh->numCellsY - 1)
        {
            tempRow = row + 1;
            tempIndex = tempSlice*mesh->numCellsX*mesh->numCellsY 
                        + tempRow*mesh->numCellsX + tempCol;
            if (Domain[tempIndex] == -1)
            {
                Domain[tempIndex] = 0;
                cList.insert(std::tuple(tempCol, tempRow, tempSlice));
            }
        }

        // Front

        tempCol = col;
        tempRow = row;

        if(slice != 0)
        {
            tempSlice = slice - 1;
            tempIndex = tempSlice*mesh->numCellsX*mesh->numCellsY 
                        + tempRow*mesh->numCellsX + tempCol;
            if (Domain[tempIndex] == -1)
            {
                Domain[tempIndex] = 0;
                cList.insert(std::tuple(tempCol, tempRow, tempSlice));
            }
        }

        // Back

        if(slice != mesh->numCellsZ - 1)
        {
            tempSlice = slice + 1;
            tempIndex = tempSlice*mesh->numCellsX*mesh->numCellsY 
                        + tempRow*mesh->numCellsX + tempCol;
            if (Domain[tempIndex] == -1)
            {
                Domain[tempIndex] = 0;
                cList.insert(std::tuple(tempCol, tempRow, tempSlice));
            }
        }

        // West

        tempRow = row;
        tempSlice = slice;

        if(col != 0)
        {
            tempCol = col - 1;
            tempIndex = tempSlice*mesh->numCellsX*mesh->numCellsY 
                        + tempRow*mesh->numCellsX + tempCol;
            if (Domain[tempIndex] == -1)
            {
                Domain[tempIndex] = 0;
                cList.insert(std::tuple(tempCol, tempRow, tempSlice));
            }
        }

        // East

        if(col != mesh->numCellsX - 1)
        {
            tempCol = col + 1;
            tempIndex = tempSlice*mesh->numCellsX*mesh->numCellsY 
                        + tempRow*mesh->numCellsX + tempCol;
            if (Domain[tempIndex] == -1)
            {
                Domain[tempIndex] = 0;
                cList.insert(std::tuple(tempCol, tempRow, tempSlice));
            }
        }
        // repeat until cList is empty
    }

    // Every flag that is still -1 means a non-participating media

    for(int index = 0; index<mesh->nElements; index++)
    {
        if(Domain[index] != -1) continue;

        int slice = index/(mesh->numCellsX*mesh->numCellsY);
        int row = (index - slice*mesh->numCellsX*mesh->numCellsY)/mesh->numCellsX;
        int col = (index - slice*mesh->numCellsX*mesh->numCellsY - row*mesh->numCellsX);
        int indexBC = (slice + 1)*(mesh->numCellsX + 2)*(mesh->numCellsY + 2) +
                        (row + 1)*(mesh->numCellsX + 2) + (col + 1);

        BC[indexBC] = -1;
    }
    
    // memory management
    free(Domain);

    return 0;
}


/*

    Discretizations:

*/


int DiscSS2D_Simple(options*        opts,
                    meshInfo*       mesh,
                    int*            BC,
                    double*         BC_Value,
                    double*         DC,
                    double*         CoeffMatrix,
                    double*         RHS)
{
    /*
        Function DiscSS2D_Simple:
        Inputs:
            - pointer to options data structure
            - pointer to mesh data structure
            - pointer to integer array BC holding BC types
            - pointer to double array BC_Value holding BC values
            - pointer to double array DC holding diffusion coefficients
            - pointer to double array CoeffMatrix Coefficient Matrix
            - pointer to double array RHS holding right-hand side of discretized system.
        Output:
            - none.

        Function creates a discretization based on user entered information and boundary conditions,
        and it stores the discretized matrix in the array CoeffMatrix and the RHS on the RHS array.
        Boundary condition choice can be flexible, but this function is primarily for steady-state
        simulations.
    */

    // Set necessary variables

    int nCols;
    nCols = mesh->numCellsX;

    double dx, dy;
    dx = mesh->dx;
    dy = mesh->dy;

    int row, col;
    long int BC_index;
    double dw, de, ds, dn;

    for(long int i = 0; i < mesh->nElements; i++)
    {
        // dissolve index into rows and cols
        row = i / nCols;
        col = i - row*nCols;

        // get the equivalent index for BC's
        BC_index = (row + 1)*(nCols + 2) + (col + 1);
        
        // make sure CoeffMatrix and RHS are zero
        
        RHS[i] = 0;
        for(int k = 0; k < 5; k++)
        {
            CoeffMatrix[i*5 + k] = 0;
        }

        /*
            Correct for non-participating media, analogous to
            pressure-decoupled solid velocity correction:
            https://doi.org/10.1016/j.ijheatmasstransfer.2009.12.057
        */

        if(BC[BC_index] == -1)
        {
            // 1*phi = 0;
            CoeffMatrix[i*5 + 0] = 1;
            RHS[i] = 0;
            continue;
        }

        // Maybe that isn't necessary

        // ****************************************

        // Account for all boundaries

        // ****************************************

        // Check if this is a source/sink via Neumann BC

        if(BC[BC_index] != 0)
        {
            // this is a boundary, thus not part of the simulation
            // 1*phi = 0;
            CoeffMatrix[i*5 + 0] = 1;
            RHS[i] = 0;
            continue;
        }

        // This means participating fluid and not a wall

        /*
            Indexing for coeff marix:

            0 : P       i
            1 : W       i - 1
            2 : E       i + 1
            3 : S       i + nCols
            4 : N       i - nCols
        */

        // West

        if(BC[BC_index - 1] == 0)
        {
            // west is not a boundary, proceed normally
            dw = WeightedHarmonicMean(dx/2, dx/2, DC[i], DC[i - 1]);
            CoeffMatrix[i*5 + 1] = dw*(dy)/dx;
            CoeffMatrix[i*5 + 0] -= dw*(dy)/dx;
        }
        else if(BC[BC_index - 1] == 1)
        {
            // west is fixed concentration boundary
            dw = DC[i];
            CoeffMatrix[i*5 + 0] -= dw*(dy)/(dx/2);
            RHS[i] -= BC_Value[BC_index - 1]*dw*(dy)/(dx/2); 
        }
        else if(BC[BC_index - 1] == 2)
        {
            // Flux boundary (Neumann)
            RHS[i] -= BC_Value[BC_index - 1]*(dy);
        }   // other BC's not implemented yet

        // East

        if(BC[BC_index + 1] == 0)
        {
            // east is not a boundary, proceed normally
            de = WeightedHarmonicMean(dx/2, dx/2, DC[i], DC[i + 1]);
            CoeffMatrix[i*5 + 2] = de*(dy)/dx;
            CoeffMatrix[i*5 + 0] -= de*(dy)/dx;
        }
        else if(BC[BC_index + 1] == 1)
        {
            // west if fixed concentration
            de = DC[i];
            CoeffMatrix[i*5 + 0] -= de*(dy)/(dx/2);
            RHS[i] -= BC_Value[BC_index + 1]*de*(dy)/(dx/2);
        }
        else if(BC[BC_index + 1] == 2)
        {
            // Flux boundary (Neumann)
            RHS[i] -= BC_Value[BC_index + 1]*(dy);
        }

        // South

        if(BC[BC_index + (nCols + 2)] == 0)
        {
            // south is not a boundary
            ds = WeightedHarmonicMean(dy/2, dy/2, DC[i], DC[i + nCols]);
            CoeffMatrix[i*5 + 3] = ds*(dx)/dy;
            CoeffMatrix[i*5 + 0] -= ds*(dx)/dy;
        }
        else if(BC[BC_index + (nCols + 2)] == 1)
        {
            // Concentration BC (Dirichlet)
            ds = DC[i];
            CoeffMatrix[i*5 + 0] -= ds*(dx)/(dy/2);
            RHS[i] -= BC_Value[BC_index + (nCols + 2)]*ds*(dx)/(dy/2);
        }
        else if(BC[BC_index + (nCols + 2)] == 2)
        {
            // Flux BC (Neumann)
            RHS[i] -= BC_Value[BC_index + (nCols + 2)]*(dx);
        }

        // North

        if(BC[BC_index - (nCols + 2)] == 0)
        {
            // north is not a boundary
            dn = WeightedHarmonicMean(dy/2, dy/2, DC[i], DC[i - nCols]);
            CoeffMatrix[i*5 + 4] = dn*(dx)/dy;
            CoeffMatrix[i*5 + 0] -= dn*(dx)/dy;
        }
        else if(BC[BC_index - (nCols + 2)] == 1)
        {
            // Concentration BC (Dirichlet)
            dn = DC[i];
            CoeffMatrix[i*5 + 0] -= dn*(dx)/(dy/2);
            RHS[i] -= BC_Value[BC_index - (nCols + 2)]*dn*(dx)/(dy/2);
        }
        else if(BC[BC_index - (nCols + 2)] == 2)
        {
            // Flux BC (Neumann)
            RHS[i] -= BC_Value[BC_index - (nCols + 2)]*(dx);
        }

        // end
    }

    return 0;
}


int DiscSS3D_Simple(options*        opts,
                    meshInfo*       mesh,
                    int*            BC,
                    double*         BC_Value,
                    double*         DC,
                    double*         CoeffMatrix,
                    double*         RHS)
{
    /*
        Function DiscSS3D_Simple:
        Inputs:
            - pointer to options data structure
            - pointer to mesh data structure
            - pointer to integer array BC holding BC types
            - pointer to double array BC_Value holding BC values
            - pointer to double array DC holding diffusion coefficients
            - pointer to double array CoeffMatrix Coefficient Matrix
            - pointer to double array RHS holding right-hand side of discretized system.
        Output:
            - none.

        Function creates a discretization based on user entered information and boundary conditions,
        and it stores the discretized matrix in the array CoeffMatrix and the RHS on the RHS array.
        Boundary condition choice can be flexible, but this function is primarily for steady-state
        simulations.
    */
    int nCols, nRows;
    nCols = mesh->numCellsX;
    nRows = mesh->numCellsY;

    double dx, dy, dz;
    dx = mesh->dx;
    dy = mesh->dy;
    dz = mesh->dz;

    int row, col, slice;
    long int BC_index;
    double dw, de, ds, dn, df, db;
    for(long int i = 0; i < mesh->nElements; i++)
    {
        // read the index into slice, row, and col
        slice   = i/(nRows*nCols);
        row     = (i - slice*nRows*nCols)/nCols;
        col     = (i - slice*nRows*nCols - row*nCols);

        BC_index = (slice + 1)*(nCols + 2)*(nRows + 2) +
                    (row + 1)*(nCols + 2) + col + 1;
        // make sure RHS and CoeffMatrix are initialized
        RHS[i] = 0;
        for(int k = 0; k < 7; k++)
        {
            CoeffMatrix[i*7 + k] = 0;
        }
        /*
            Correct for non-participating media, analogous to
            pressure-decoupled solid velocity correction:
            https://doi.org/10.1016/j.ijheatmasstransfer.2009.12.057
        */
        if(BC[BC_index] == -1)
        {
            // 1*phi = 0;
            CoeffMatrix[i*7 + 0] = 1;
            RHS[i] = 0;
            continue;
        }

        // Maybe that isn't necessary

        // ****************************************

        // Account for all boundaries

        // ****************************************

        // Check if this is a source/sink via Neumann BC
        if(BC[BC_index] != 0)
        {
            // this is a boundary, thus not part of the simulation
            // 1*phi = 0;
            CoeffMatrix[i*7 + 0] = 1;
            RHS[i] = 0;
            continue;
        }

        // This means participating fluid and not a wall

        /*
            Indexing for coeff marix:

            0 : P       i
            1 : W       i - 1
            2 : E       i + 1
            3 : S       i + nCols
            4 : N       i - nCols
            5 : B       i + nCols * nRows
            6 : F       i - nCols * nRows

        */

        // West

        if(BC[BC_index - 1] == 0)
        {
            // west is not a bounary, proceed normally
            dw = WeightedHarmonicMean(dx/2, dx/2, DC[i], DC[i - 1]);
            CoeffMatrix[i*7 + 1] = dw*(dy*dz)/dx;
            CoeffMatrix[i*7 + 0] -= dw*(dy*dz)/dx;
        }
        else if(BC[BC_index - 1] == 1)
        {
            // west is fixed concentration boundary
            dw = DC[i];
            CoeffMatrix[i*7 + 0] -= dw*(dy*dz)/(dx/2);
            RHS[i] -= BC_Value[BC_index - 1]*dw*(dy*dz)/(dx/2); 
        }
        else if(BC[BC_index - 1] == 2)
        {
            // Flux boundary (Neumann)
            RHS[i] -= BC_Value[BC_index - 1]*(dy*dz);
        }   // other BC's not implemented yet

        // East

        if(BC[BC_index + 1] == 0)
        {
            // east is not a boundary
            de = WeightedHarmonicMean(dx/2, dx/2, DC[i], DC[i + 1]);
            CoeffMatrix[i*7 + 2] = de*(dy*dz)/dx;
            CoeffMatrix[i*7 + 0] -= de*(dy*dz)/dx;
        }
        else if(BC[BC_index + 1] == 1)
        {
            // fixed concentration BC
            de = DC[i];
            CoeffMatrix[i*7 + 0] -= de*(dy*dz)/(dx/2);
            RHS[i] -= BC_Value[BC_index + 1]*de*(dy*dz)/(dx/2);
        }
        else if(BC[BC_index] == 2)
        {
            // Flux boundary (Neumann)
            RHS[i] -= BC_Value[BC_index + 1]*(dy*dz);
        }   // other BC's not implemented yet

        // South

        if(BC[BC_index + (nCols + 2)] == 0)
        {
            // south is not a boundary
            ds = WeightedHarmonicMean(dy/2, dy/2, DC[i], DC[i + nCols]);
            CoeffMatrix[i*7 + 3] = ds*(dx*dz)/dy;
            CoeffMatrix[i*7 + 0] -= ds*(dx*dz)/dy;
        }
        else if(BC[BC_index + (nCols + 2)] == 1)
        {
            // Concentration BC (Dirichlet)
            ds = DC[i];
            CoeffMatrix[i*7 + 0] -= ds*(dx*dz)/(dy/2);
            RHS[i] -= BC_Value[BC_index + (nCols + 2)]*ds*(dx*dz)/(dy/2);
        }
        else if(BC[BC_index + (nCols + 2)] == 2)
        {
            // Flux BC (Neumann)
            RHS[i] -= BC_Value[BC_index + (nCols + 2)]*(dx*dz);
        }

        // North

        if(BC[BC_index - (nCols + 2)] == 0)
        {
            // north is not a boundary
            dn = WeightedHarmonicMean(dy/2, dy/2, DC[i], DC[i - nCols]);
            CoeffMatrix[i*7 + 4] = dn*(dx*dz)/dy;
            CoeffMatrix[i*7 + 0] -= dn*(dx*dz)/dy;
        }
        else if(BC[BC_index - (nCols + 2)] == 1)
        {
            // Concentration BC (Dirichlet)
            dn = DC[i];
            CoeffMatrix[i*7 + 0] -= dn*(dx*dz)/(dy/2);
            RHS[i] -= BC_Value[BC_index - (nCols + 2)]*dn*(dx*dz)/(dy/2);
        }
        else if(BC[BC_index - (nCols + 2)] == 2)
        {
            // Flux BC (Neumann)
            RHS[i] -= BC_Value[BC_index - (nCols + 2)]*(dx*dz);
        }

        // Back

        if(BC[BC_index + (nCols + 2)*(nRows + 2)] == 0)
        {
            // back is not a boundary
            db = WeightedHarmonicMean(dz/2, dz/2, DC[i], DC[i + nRows*nCols]);
            CoeffMatrix[i*7 + 5] = db*(dx*dy)/dz;
            CoeffMatrix[i*7 + 0] -= db*(dx*dy)/dz;
        }
        else if(BC[BC_index + (nCols + 2)*(nRows + 2)] == 1)
        {
            // Concentration BC (Dirichlet)
            db = DC[i];
            CoeffMatrix[i*7 + 0] -= db*(dx*dy)/(dz/2);
            RHS[i] -= BC[BC_index + (nCols + 2)*(nRows + 2)]*db*(dx*dy)/(dz/2);
        }
        else if(BC[BC_index + (nCols + 2)*(nRows + 2)] == 2)
        {
            // Flux BC (Neumann)
            RHS[i] -= BC_Value[BC_index + (nCols + 2)*(nRows + 2)]*(dx*dy);
        }

        // Front

        if(BC[BC_index - (nCols + 2)*(nRows + 2)] == 0)
        {
            // front is not a boundary
            df = WeightedHarmonicMean(dz/2, dz/2, DC[i], DC[i - nRows*nCols]);
            CoeffMatrix[i*7 + 6] = df*(dx*dy)/dz;
            CoeffMatrix[i*7 + 0] -= df*(dx*dy)/dz;
        }
        else if(BC[BC_index - (nCols + 2)*(nRows + 2)] == 1)
        {
            // Concentration BC (Dirichlet)
            df = DC[i];
            CoeffMatrix[i*7 + 0] -= df*(dx*dy)/(dz/2);
            RHS[i] -= BC[BC_index - (nCols + 2)*(nRows + 2)]*df*(dx*dy)/(dz/2);
        }
        else if(BC[BC_index - (nCols + 2)*(nRows + 2)] == 2)
        {
            // Flux BC (Neumann)
            RHS[i] -= BC_Value[BC_index - (nCols + 2)*(nRows + 2)]*(dx*dy);
        }

        // end
    }

    return 0;
}


/*

    GPU Space Management:

*/

int initGPU_2DSOR(double**      d_Coeff,
                double**        d_RHS,
                double**        d_Conc,
                double**        d_ConcTemp, 
                meshInfo*       mesh)
{
    /*
        Function initGPU_2DSOR:
        Inputs:
            - double pointer to d_Coeff, storing coeff matrix in GPU
            - double pointer to d_RHS, storing RHS vector in GPU
            - double pointer to d_Conc, the concentration array in GPU memory
            - double pointer to d_ConcTemp, where the concentration array will 
                be modified in GPU memory
            - pointer to meshInfo, holding general information about the mesh.
        Outputs:
            - None.
        
        The function will allocate the sufficient space for the arrays needed for
        the Standard Over-Relaxed Jacobi Method. It also initializes the arrays.
        Error calls are returned if it fails.
    */

    // Set device

    CHECK_CUDA( cudaSetDevice(0));

    // Allocate space

    CHECK_CUDA( cudaMalloc( (void**)&(*d_Coeff),     mesh->nElements * sizeof(double) * 5));
    CHECK_CUDA( cudaMalloc( (void**)&(*d_RHS),       mesh->nElements * sizeof(double)) );
    CHECK_CUDA( cudaMalloc( (void**)&(*d_Conc),      mesh->nElements * sizeof(double)) );
    CHECK_CUDA( cudaMalloc( (void**)&(*d_ConcTemp),  mesh->nElements * sizeof(double)) );

    // Set buffers

    CHECK_CUDA( cudaMemset((*d_Coeff),      0 , mesh->nElements * sizeof(double) * 5) );
    CHECK_CUDA( cudaMemset((*d_RHS),        0 , mesh->nElements * sizeof(double)) );
    CHECK_CUDA( cudaMemset((*d_Conc),       0 , mesh->nElements * sizeof(double)) );
    CHECK_CUDA( cudaMemset((*d_ConcTemp),   0 , mesh->nElements * sizeof(double)) );

    return 0;
}


int initGPU_3DSOR(double**      d_Coeff,
                double**        d_RHS,
                double**        d_Conc,
                double**        d_ConcTemp, 
                meshInfo*       mesh)
{
    /*
        Function initGPU_3DSOR:
        Inputs:
            - double pointer to d_Coeff, storing coeff matrix in GPU
            - double pointer to d_RHS, storing RHS vector in GPU
            - double pointer to d_Conc, the concentration array in GPU memory
            - double pointer to d_ConcTemp, where the concentration array will 
                be modified in GPU memory
            - pointer to meshInfo, holding general information about the mesh.
        Outputs:
            - None.
        
        The function will allocate the sufficient space for the arrays needed for
        the Standard Over-Relaxed Jacobi Method. It also initializes the arrays.
        Error calls are returned if it fails.
    */

    // Set device

    CHECK_CUDA( cudaSetDevice(0));

    // Allocate space

    CHECK_CUDA( cudaMalloc( (void**)&(*d_Coeff),     mesh->nElements * sizeof(double) * 7));
    CHECK_CUDA( cudaMalloc( (void**)&(*d_RHS),       mesh->nElements * sizeof(double)) );
    CHECK_CUDA( cudaMalloc( (void**)&(*d_Conc),      mesh->nElements * sizeof(double)) );
    CHECK_CUDA( cudaMalloc( (void**)&(*d_ConcTemp),  mesh->nElements * sizeof(double)) );

    // Set buffers

    CHECK_CUDA( cudaMemset((*d_Coeff),      0 , mesh->nElements * sizeof(double) * 7) );
    CHECK_CUDA( cudaMemset((*d_RHS),        0 , mesh->nElements * sizeof(double)) );
    CHECK_CUDA( cudaMemset((*d_Conc),       0 , mesh->nElements * sizeof(double)) );
    CHECK_CUDA( cudaMemset((*d_ConcTemp),   0 , mesh->nElements * sizeof(double)) );

    return 0;
}


int unInitGPU_SOR(double**        d_Coeff,
                double**            d_RHS,
                double**            d_Conc,
                double**            d_ConcTemp)
{
    /*
        Function unInitGPU_SOR:
        Inputs:
            - double pointer to d_Coeff, storing coeff matrix in GPU
            - double pointer to d_RHS, storing RHS vector in GPU
            - double pointer to d_Conc, the concentration array in GPU memory
            - double pointer to d_ConcTemp, where the concentration array will 
                be modified in GPU memory
        Outputs:
            - None.
        
        The function will free space in device memory.
    */

    CHECK_CUDA( cudaFree((*d_Coeff)));
    CHECK_CUDA( cudaFree((*d_RHS)));
    CHECK_CUDA( cudaFree((*d_Conc)));
    CHECK_CUDA( cudaFree((*d_ConcTemp)));
    
    return 0;
}

/*

    Solvers:

*/

int JI2D_SOR(double     *Coeff,
            double      *RHS,
            double      *Concentration,
            double      *d_Coeff,
            double      *d_RHS,
            double      *d_Conc,
            double      *d_ConcTemp,
            options     *opts,
            meshInfo    *mesh)
{
    /*
        Function JI2D_SOR:
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
        convergence criteria, and kernel coordination.
    */

    long int iterCount = 0;
    int threads_per_block = 128;
    int numBlocks = mesh->nElements / threads_per_block + 1;
    
    double pctChange = 1;
    int iterToCheck = 1000;

    // copy arrays into GPU

    CHECK_CUDA( cudaMemcpy( d_Conc      , Concentration, 
                sizeof(double) * mesh->nElements, cudaMemcpyHostToDevice) );

    CHECK_CUDA( cudaMemcpy( d_ConcTemp  , Concentration, 
                sizeof(double) * mesh->nElements, cudaMemcpyHostToDevice) );

    CHECK_CUDA( cudaMemcpy( d_RHS       , RHS, 
                sizeof(double) * mesh->nElements, cudaMemcpyHostToDevice) );
    
    CHECK_CUDA( cudaMemcpy( d_Coeff     , Coeff, 
                sizeof(double) * mesh->nElements * 5, cudaMemcpyHostToDevice) );
    
    // Create Array to store temp Conc

    double* TempConc = (double *)malloc(sizeof(double) * mesh->nElements);

    memcpy(TempConc, Concentration, sizeof(double) * mesh->nElements);

    // start the main loop

    while(iterCount < opts->MAX_ITER && pctChange > opts->ConvergeCriteria)
    {
        // call kernel

        JI_SOR2D_kernel<<<numBlocks, threads_per_block>>>(d_Coeff, d_ConcTemp, d_RHS, d_Conc,
                                        mesh->nElements, mesh->numCellsX, mesh->numCellsY);
        // check convergence

        if(iterCount % iterToCheck == 0 && iterCount != 0)
        {
            // copy array from device to host
            CHECK_CUDA( cudaMemcpy(Concentration, d_Conc,sizeof(double)*mesh->nElements, cudaMemcpyDeviceToHost) );
            
            // compare
            double sum = 0;
            long int count = 0;

            for(int i = 0;  i < mesh->nElements; i++)
            {
                if(Concentration[i] != 0)
                {
                    sum += fabs((Concentration[i] - TempConc[i])/Concentration[i]);
                    count++;
                }
            }
            // calculate the change
            pctChange = sum/count;
            // copy memory to temp conc
            memcpy(TempConc, Concentration, sizeof(double)*mesh->nElements);
        }
        
        // update d_Conc = d_ConcTemp

        CHECK_CUDA( cudaMemcpy(d_ConcTemp, d_Conc, sizeof(double)*mesh->nElements, cudaMemcpyDeviceToDevice) );
        
        // increment
        iterCount++;
    }

    // copy the solution

    CHECK_CUDA( cudaMemcpy(Concentration, d_ConcTemp, 
                    sizeof(double)*mesh->nElements, cudaMemcpyDeviceToHost) );

    // print success

    if(opts->verbose)
    {
        printf("Total iter = %d, pct change = %lf\n", iterCount, pctChange);
    }

    return 0;
}


int JI3D_SOR(double     *Coeff,
            double      *RHS,
            double      *Concentration,
            double      *d_Coeff,
            double      *d_RHS,
            double      *d_Conc,
            double      *d_ConcTemp,
            options     *opts,
            meshInfo    *mesh)
{
    /*
        Function JI3D_SOR:
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
        in 3D, with a standard over-relaxation applied. The function will manage data transfers,
        convergence criteria, and kernel coordination.
    */

    long int iterCount = 0;
    int threads_per_block = 128;
    int numBlocks = mesh->nElements / threads_per_block + 1;
    
    double pctChange = 1;
    int iterToCheck = 1000;

    // copy arrays into GPU

    CHECK_CUDA( cudaMemcpy( d_Conc      , Concentration, 
                sizeof(double) * mesh->nElements, cudaMemcpyHostToDevice) );

    CHECK_CUDA( cudaMemcpy( d_ConcTemp  , Concentration, 
                sizeof(double) * mesh->nElements, cudaMemcpyHostToDevice) );

    CHECK_CUDA( cudaMemcpy( d_RHS       , RHS, 
                sizeof(double) * mesh->nElements, cudaMemcpyHostToDevice) );
    
    CHECK_CUDA( cudaMemcpy( d_Coeff     , Coeff, 
                sizeof(double) * mesh->nElements * 7, cudaMemcpyHostToDevice) );
    
    // Create Array to store temp Conc

    double* TempConc = (double *)malloc(sizeof(double) * mesh->nElements);

    memcpy(TempConc, Concentration, sizeof(double) * mesh->nElements);

    // start the main loop

    while(iterCount < opts->MAX_ITER && pctChange > opts->ConvergeCriteria)
    {
        // call kernel

        JI_SOR3D_kernel<<<numBlocks, threads_per_block>>>(d_Coeff, d_ConcTemp, d_RHS, d_Conc,
                                        mesh->nElements, mesh->numCellsX, mesh->numCellsY);
        // check convergence

        if(iterCount % iterToCheck == 0 && iterCount != 0)
        {
            // copy array from device to host
            CHECK_CUDA( cudaMemcpy(Concentration, d_Conc,sizeof(double)*mesh->nElements, cudaMemcpyDeviceToHost) );
            
            // compare
            double sum = 0;
            long int count = 0;

            for(int i = 0;  i < mesh->nElements; i++)
            {
                if(Concentration[i] != 0)
                {
                    sum += fabs((Concentration[i] - TempConc[i])/Concentration[i]);
                    count++;
                }
            }
            // calculate the change
            pctChange = sum/count;
            // copy memory to temp conc
            memcpy(TempConc, Concentration, sizeof(double)*mesh->nElements);
        }
        
        // update d_Conc = d_ConcTemp

        CHECK_CUDA( cudaMemcpy(d_ConcTemp, d_Conc, sizeof(double)*mesh->nElements, cudaMemcpyDeviceToDevice) );
        
        // increment
        iterCount++;
    }

    // copy the solution

    CHECK_CUDA( cudaMemcpy(Concentration, d_ConcTemp, 
                    sizeof(double)*mesh->nElements, cudaMemcpyDeviceToHost) );

    // print success

    if(opts->verbose)
    {
        printf("Total iter = %d, pct change = %lf\n", iterCount, pctChange);
    }

    return 0;
}


int GS2D_OMP(double *Coeff, double *RHS, double *Concentration, options *opts, meshInfo *mesh)
{

    long int iterCount = 0;
    double sigma = 0;
    double pctChange = 1;
    int i;
    int iterToCheck = 100;
    int offset[5];
    // set array offsets
    offset[0] = 0;
    offset[1] = -1;
    offset[2] = 1;
    offset[3] = mesh->numCellsX;
    offset[4] = -mesh->numCellsX;

    double *Check = (double *)malloc(sizeof(double)*mesh->nElements);
    memcpy(Check, Concentration, sizeof(double)*mesh->nElements);

    if(opts->verbose)
    {
        printf("Starting Main Loop\n");
    }
    

    #pragma omp parallel private(i, sigma)

    while(pctChange > opts->ConvergeCriteria && iterCount < opts->MAX_ITER)
    {
        #pragma omp parallel for
        for(i = 0; i<mesh->nElements; i++)
        {
            sigma = 0;
            for(int j = 1; j < 5; j++)
            {
                if(Coeff[i*5 + j] == 0) continue;
                sigma += Coeff[i*5 + j] * Concentration[i + offset[j]];
            }
            Concentration[i] = 1.0/Coeff[i*5 + 0]*(RHS[i] - sigma);
        }

        iterCount++;
        long int count = 0;
        if(iterCount % iterToCheck == 0)
        {
            double sum = 0;
            #pragma omp parallel for reduction(+:sum)
            for(i = 0; i<mesh->nElements; i++)
            {
                if(Concentration[i] != 0)
                {
                    sum += fabs((Concentration[i] - Check[i])/Concentration[i]);
                    count++;
                }
            }
            pctChange = sum/count;
            memcpy(Check, Concentration, sizeof(double)*mesh->nElements);
        }
    }

    if(opts->verbose)
    {
        printf("Total iter = %d, pct change = %lf\n", iterCount, pctChange);
    }

    free(Check);
    return 0;
}



int GS3D_OMP(double *Coeff, double *RHS, double *Concentration, options *opts, meshInfo *mesh)
{

    int iterCount = 0;
    double sigma = 0;
    double pctChange = 1;
    int i;
    int iterToCheck = 100;
    int offset[7];
    // set array offsets
    offset[0] = 0;
    offset[1] = -1;
    offset[2] = 1;
    offset[3] = mesh->numCellsX;
    offset[4] = -mesh->numCellsX;
    offset[5] = mesh->numCellsX*mesh->numCellsY;
    offset[6] = -mesh->numCellsX*mesh->numCellsY;

    double *Check = (double *)malloc(sizeof(double)*mesh->nElements);
    memcpy(Check, Concentration, sizeof(double)*mesh->nElements);

    if(opts->verbose)
    {
        printf("Starting Main Loop\n");
    }

    #pragma omp parallel private(i, sigma)

    while(pctChange > opts->ConvergeCriteria && iterCount < opts->MAX_ITER)
    {
        #pragma omp parallel for
        for(i = 0; i<mesh->nElements; i++)
        {
            sigma = 0;
            for(int j = 1; j < 7; j++)
            {
                if(Coeff[i*7 + j] != 0){
                    sigma += Coeff[i*7 + j] * Concentration[i + offset[j]];
                }
            }
            Concentration[i] = 1.0/Coeff[i*7 + 0]*(RHS[i] - sigma);
        }

        iterCount++;
        if((iterCount % iterToCheck) == 0)
        {
            double sum = 0;
            long int count = 0;
            #pragma omp parallel for reduction(+:sum)
            for(i = 0; i<mesh->nElements; i++)
            {
                if(Concentration[i] != 0)
                {
                    sum += fabs((Concentration[i] - Check[i])/Concentration[i]);
                    count++;
                }
            }
            pctChange = sum/count;
            memcpy(Check, Concentration, sizeof(double)*mesh->nElements);
        }
        
    }

    if(opts->verbose)
    {
        printf("Total iter = %d, pct change = %lf\n", iterCount, pctChange);
    }
    

    free(Check);
    return 0;
}


/*

    Main simulation-modes control:

*/


int SteadyStateSim2D(options* opts)
{
    /*
        Function SteadyStateSim2D:
        Inputs:
            - pointer to options data structure
        Outputs:
            - none
        
        Function will control the simulation of the 2D structure in Steady-State
        operating conditions.
    */

    // Initialize required data structures

    meshInfo mesh;

    // For the 2D code, we can read the image straight up (no need for user entered info)
    char *simObject = nullptr;
    int readFlag = 0;

    readFlag = readImg2D(opts, &mesh, simObject);

    // return an error if the image wasn't read properly

    if(readFlag == 1) return 1;

    // set mesh parameters

    mesh.dx = (double)1.0/mesh.numCellsX;
    mesh.dy = (double)1.0/mesh.numCellsY;

    // Create arrays for BC's and DC's

    double  *DC = (double *)malloc(sizeof(double)*mesh.nElements);
    int     *BC = (int *)malloc(sizeof(int)*(mesh.numCellsY + 2) * (mesh.numCellsX + 2));
    double  *BC_Value = (double *)malloc(sizeof(double)*(mesh.numCellsY + 2) * (mesh.numCellsX + 2));

    // initialize arrays

    memset(DC, 0, sizeof(double)*mesh.nElements);
    memset(BC, 0, sizeof(int)*(mesh.numCellsY + 2) * (mesh.numCellsX + 2));
    memset(BC_Value, 0, sizeof(double)*(mesh.numCellsY + 2) * (mesh.numCellsX + 2));

    // Populate the array with the diffusion coefficients

    SetDC2D(opts, &mesh, DC, simObject);
    
    // Set Boundary Conditions

    SetBC_DeffSetup2D(opts, &mesh, BC, BC_Value);

    // set if D[i,j] < 10^-15, D[i,j] = 0 becomes a Neumann BC

    for(int index = 0; index<mesh.nElements; index++)
    {
        int row = index/(mesh.numCellsX);
        int col = index - row*mesh.numCellsX;

        int indexBC = (row + 1)*(mesh.numCellsX + 2) + (col + 1);

        if(DC[index] < 1e-15)
        {
            DC[index] = 0;
            BC[indexBC] = 2;    // set Neumann BC with zero flux
        }
    }

    // If any phase is impermeable, need to find all non-participating media

    FloodFill2D_DeffSetup(&mesh, BC, DC);

    // Allocate arrays for holding discretized equations

    double* CoeffMatrix     = (double *)malloc(mesh.nElements * 5 * sizeof(double));
    double* RHS             = (double *)malloc(mesh.nElements * sizeof(double));
    double* Concentration   = (double *)malloc(mesh.nElements * sizeof(double));
    
    // initialize the memory

    memset(CoeffMatrix  , 0, mesh.nElements * sizeof(double) * 5);
    memset(RHS          , 0, mesh.nElements * sizeof(double));
    memset(Concentration, 0, mesh.nElements * sizeof(double));

    // Linear initialize concentration

    for(int i = 0; i<mesh.nElements; i++)
    {
        int row = i/mesh.numCellsX;
        int col = i - row*mesh.numCellsX;
        Concentration[i] = ((double)col/mesh.numCellsX)*(opts->CRight - opts->CLeft) + opts->CLeft;
        if(DC[i] == 0) Concentration[i] = 0;
    }

    // Discretize equations

    DiscSS2D_Simple(opts, &mesh, BC, BC_Value, DC, CoeffMatrix, RHS);

    if(opts->useGPU == 0)
    {
        omp_set_num_threads(opts->nThreads);

        GS2D_OMP(CoeffMatrix, RHS, Concentration, opts, &mesh);
    } else
    {
        // Now we confirm that there is a match in GPUs available and user expectations
        
        int nDevices;
        cudaGetDeviceCount(&nDevices);
        
        if(nDevices < 1)
        {
            printf("No CUDA-capable GPU Detected! Exiting...\n");
            return 1;
        }else if(nDevices < opts->nGPU)
        {
            printf("User requested %d GPUs, but only %d were detected.\n", opts->nGPU, nDevices);
            printf("Proceeding with %d GPUs\n", nDevices);
            opts->nGPU = nDevices;
        }
        
        // Declare needed arrays

        double *d_Coeff     = NULL;
        double *d_RHS       = NULL;
        double *d_Conc      = NULL;
        double *d_ConcTemp  = NULL;

        // Initialize the GPU arrays

        initGPU_2DSOR(&d_Coeff, &d_RHS, &d_Conc, &d_ConcTemp, &mesh);

        // // Solve

        JI2D_SOR(CoeffMatrix, RHS, Concentration, d_Coeff,
                d_RHS, d_Conc, d_ConcTemp, opts, &mesh);
        
        // // Free GPU memory

        unInitGPU_SOR(&d_Coeff, &d_RHS, &d_Conc, &d_ConcTemp);
    }  

    FILE* OUT;

    OUT = fopen("ConDist2D.csv", "w");
    fprintf(OUT,"x,y,C\n");
    for(int i = 0; i<mesh.numCellsY; i++)
    {
        for(int j = 0; j<mesh.numCellsX; j++)
        {
            if(Concentration[i*mesh.numCellsX + j] != Concentration[i*mesh.numCellsX + j]) 
            {
                Concentration[i*mesh.numCellsX + j] = 0;
                printf("NaN Found at col %d, row %d\n", j,i);
            }
            
            fprintf(OUT,"%d,%d,%lf\n",j,i,Concentration[i*mesh.numCellsX + j]);
        }
    }

    fclose(OUT);

    // Memory management

    free(RHS);
    free(CoeffMatrix);
    free(Concentration);

    free(BC);
    free(BC_Value);
    free(DC);

    free(simObject);

    printf("Done?\n");

    return 0;
}


int SteadyStateSim3D(options* opts)
{
    /*
        Function SteadyStateSim3D:
        Inputs:
            - pointer to options data structure
        Outputs:
            - None.
        
        Function will control the entire simulation of a 3D structure in Steady-State
        operation.
    */

    // Initialize simulation data structures
   
    meshInfo mesh;

    // populate mesh info with available information

    mesh.numCellsX = opts->width*opts->MeshIncreaseX;
    mesh.numCellsY = opts->height*opts->MeshIncreaseY;
    mesh.numCellsZ = opts->depth*opts->MeshIncreaseZ;
    
    mesh.nElements = mesh.numCellsX*mesh.numCellsY*mesh.numCellsZ;

    mesh.dx = (double)1.0/mesh.numCellsX;
    mesh.dy = (double)1.0/mesh.numCellsY;
    mesh.dz = (double)1.0/mesh.numCellsZ;

    // Read structure
    
    char* simObject = (char *)malloc(opts->height*opts->width*opts->depth*sizeof(char));
    
    memset(simObject, 0, opts->height*opts->width*opts->depth*sizeof(char));    // initialized to pore-space

    readCSV3D(opts, simObject);

    // Declare and define BC's and DC's for the domain

    double *DC = (double *)malloc(sizeof(double)*mesh.nElements);
    int   *BC = (int *)  malloc(sizeof(int)*(mesh.numCellsX + 2) * 
                            (mesh.numCellsX + 2) * (mesh.numCellsX + 2));
    double *BC_Value = (double *)malloc(sizeof(double)*(mesh.numCellsX + 2) * 
                            (mesh.numCellsX + 2) * (mesh.numCellsX + 2));
    
    memset(DC, 0, mesh.nElements*sizeof(double));
    memset(BC, 0, (mesh.numCellsX + 2)*(mesh.numCellsY + 2)
                    *(mesh.numCellsZ + 2)*sizeof(int));
    memset(BC_Value, 0, (mesh.numCellsX + 2)*(mesh.numCellsY + 2)
                    *(mesh.numCellsZ + 2)*sizeof(double));

    // note BC array has space for ``ghost'' grid boundaries

    // Set DC's

    SetDC3D(opts, &mesh, DC, simObject);

    // Set BC's

    SetBC_DeffSetup3D(opts, &mesh, BC, BC_Value);
    
    // set if D[i,j,k] < 10^-15, D[i,j,k] = 0 becomes a Neumann BC

    for(int index = 0; index<mesh.nElements; index++)
    {
        int slice = index/(mesh.numCellsX*mesh.numCellsY);
        int row = (index - slice*mesh.numCellsX*mesh.numCellsY)/mesh.numCellsX;
        int col = (index - slice*mesh.numCellsX*mesh.numCellsY - row*mesh.numCellsX);
        int indexBC = (slice + 1)*(mesh.numCellsX + 2)*(mesh.numCellsY + 2) 
                        + (row + 1)*(mesh.numCellsX + 2) + (col + 1);
        if(DC[index] < 1e-15)
        {
            DC[index] = 0;
            BC[indexBC] = 2;    // set Neumann BC with zero flux
        }
    }

    // If any phase is impermeable, need to find all participating media

    FloodFill3D_DeffSetup(&mesh, BC, DC);

    // Allocate arrays for holding discretized equations

    double* CoeffMatrix     = (double *)malloc(mesh.nElements * 7 * sizeof(double));
    double* RHS             = (double *)malloc(mesh.nElements * sizeof(double));
    double* Concentration   = (double *)malloc(mesh.nElements * sizeof(double));
    
    // initialize the memory

    memset(CoeffMatrix  , 0, mesh.nElements * sizeof(double) * 7);
    memset(RHS          , 0, mesh.nElements * sizeof(double));
    memset(Concentration, 0, mesh.nElements * sizeof(double));

    // Linear initialize concentration

    for(int i = 0; i<mesh.nElements; i++)
    {
        int slice = i/(mesh.numCellsX*mesh.numCellsY);
        int row = (i - slice*mesh.numCellsX*mesh.numCellsY)/mesh.numCellsX;
        int col = (i - slice*mesh.numCellsX*mesh.numCellsY - row*mesh.numCellsX);
        if(DC[i] != 0)
        {
            Concentration[i] = ((double)col/mesh.numCellsX)*(opts->CRight - opts->CLeft) + opts->CLeft;
        }
    }
    // Discretize

    DiscSS3D_Simple(opts, &mesh, BC, BC_Value, DC, CoeffMatrix, RHS);

    // Solve!

    if(opts->useGPU == 0)
    {
        // CPU Solve

        omp_set_num_threads(opts->nThreads);

        GS3D_OMP(CoeffMatrix, RHS, Concentration, opts, &mesh);
    }else
    {
        // Now we confirm that there is a match in GPUs available and user expectations
        
        int nDevices;
        cudaGetDeviceCount(&nDevices);
        
        if(nDevices < 1)
        {
            printf("No CUDA-capable GPU Detected! Exiting...\n");
            return 1;
        }else if(nDevices < opts->nGPU)
        {
            printf("User requested %d GPUs, but only %d were detected.\n", opts->nGPU, nDevices);
            printf("Proceeding with %d GPUs\n", nDevices);
            opts->nGPU = nDevices;
        }
        
        // Declare needed arrays

        double *d_Coeff     = NULL;
        double *d_RHS       = NULL;
        double *d_Conc      = NULL;
        double *d_ConcTemp  = NULL;

        // Initialize the GPU arrays

        initGPU_3DSOR(&d_Coeff, &d_RHS, &d_Conc, &d_ConcTemp, &mesh);

        // Solve

        JI3D_SOR(CoeffMatrix, RHS, Concentration, d_Coeff,
                d_RHS, d_Conc, d_ConcTemp, opts, &mesh);
        
        // Free GPU memory

        unInitGPU_SOR(&d_Coeff, &d_RHS, &d_Conc, &d_ConcTemp);
    }
    

    FILE* OUT;

    OUT = fopen("ConDist3D.csv", "w");
    fprintf(OUT,"x,y,z,c\n");
    for(int i = 0; i<mesh.numCellsY; i++)
    {
        for(int j = 0; j<mesh.numCellsX; j++)
        {
            for(int k = 0; k<mesh.numCellsZ; k++){
                fprintf(OUT,"%d,%d,%d,%1.3e\n",j,i,k,Concentration[k*mesh.numCellsX*mesh.numCellsY + i*mesh.numCellsX + j]);
            }
        }
    }

    fclose(OUT);

    // Memory management

    free(RHS);
    free(CoeffMatrix);
    free(Concentration);

    free(BC);
    free(BC_Value);
    free(DC);

    free(simObject);

    return 0;
}

#endif