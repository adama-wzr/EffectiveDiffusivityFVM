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

typedef struct
{
	double *DC;                 // array with diffusion coefficients
    double numDC;               // number of diffusion coefficients
	int MeshIncreaseX;		    // Mesh refinement in x-direction
	int MeshIncreaseY;		    // Mesh refinement in y-direction
    int MeshIncreaseZ;          // Mesh refinement in the z-direction
	double CLeft;			    // Concentration of trace species in left boundary
	double CRight;			    // Concentration of trace species in right boundary
	long int MAX_ITER;		    // Max iterations
	double ConvergeCriteria;    // Convergence Criteria
	char *inputFilename;	    // Input filename
	char *outputFilename;	    // Output filename
	int printCmap;			    // print concentration map (true/false) flag
	char *CMapName;			    // Concentration map name
	int verbose;			    // verbose flag
	int BatchFlag;			    // Batch flag
	int NumImgBatch;		    // Number of images in the batch
    bool SteadyStateFlag;        // steady state simulation or time dependent ?
} options;

typedef struct
{
	int Width;
	int Height;
	int nChannels;
	double porosity;
	double SVF;
	double LVF;
	double gpuTime;
	unsigned char *target_data;
	double deff;
	bool PathFlag;
	double conv;
} simulationInfo;

typedef struct
{
	int numCellsX;
	int numCellsY;
	int nElements;
	double dx;
	double dy;
} meshInfo;

void test(void){
    printf("helper.cuh is working\n");
    return;
}

void readInputGeneral(char* FileName, options* opts){

	/*
		readInputFile Function:
		Inputs:
			- FileName: pointer to where the input file name is stored.
			- struct options: pass a struct with the options.
		Outputs: None

		Function reads the input file and stores the options in the opts struct.
	*/

    

    return;

#endif